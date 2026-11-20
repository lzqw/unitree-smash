#!/usr/bin/env python3
"""Fit effective string and solid restitution/friction in the ball predictor with signed native response supervision."""
import argparse
import json
from pathlib import Path

import torch
from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_paired_racket_solver import read_split
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def material_table(original, strings, params):
    table=original.clone()
    restitution=torch.where(strings,params[0],params[1])
    friction=torch.where(strings,params[2],params[3])
    table[:,2]=restitution
    # The measured native contract used here has equal static and dynamic friction.
    table[:,:2]=friction[:,None]
    return table


class PairedMaterialFit:
    def __init__(self, collection, split, prior, cfg, original, strings):
        rows,branches=read_split(collection/(split+'.json'),collection,split)
        self.settings,self.material,self.strings,self.prior=cfg,original,strings,prior
        self.native=torch.tensor([r['delta']['native'] for r in rows])
        self.repeat=torch.tensor([r['repeat']['native'] for r in rows])
        self.stable=torch.tensor([r['kind']=='racket_stable' for r in rows])
        self.streams={}
        self.spins={}
        seeds=sorted({r['seed'] for r in rows})
        for b,batch in branches.items():
            self.streams[b]=batch
            self.spins[b]=[torch.load(collection/b/f'spec_{s}/trace.pt',weights_only=True,mmap=True)
                ['raw']['substep_trace']['tensors']['shuttle_state_w'][0,:,10:13].clone() for s in seeds]

    @torch.no_grad()
    def evaluate(self, params):
        table=material_table(self.material,self.strings,params)
        predicted={}
        absolute=[]
        for b,batch in self.streams.items():
            ys=[]
            for (initial,outgoing,stream),spin in zip(batch,self.spins[b],strict=True):
                ball,_=tick(self.prior,initial,[v[0] for v in stream],[v[-1] for v in stream],
                    self.settings['fitted'],self.settings['solver']['pair_contact_offset_m'],
                    measured_path=[v[:-1] for v in stream],racket_material=table,initial_spin=spin,compound_racket=True)
                ys.append(ball[:,3:])
                absolute.append(ball[:,3:]-outgoing)
            predicted[b]=torch.cat(ys)
        d=predicted['plus']-predicted['minus']
        error=torch.cat(absolute)
        frozen=self.stable
        stats=dict(absolute_rmse=float(error.square().mean().sqrt()),
                   all_signed=response_metrics(self.native,d,self.repeat),
                   stable_signed=response_metrics(self.native[frozen],d[frozen],self.repeat[frozen]))
        # Compare derivatives under the same recorded motion. Retain low-response and switched rows
        # in the absolute and signed terms, without dividing by a near-zero native response.
        relative=(d-self.native)/self.native.norm(dim=-1,keepdim=True).clamp_min(.05)
        values=[error.flatten()/error.numel()**.5, relative.flatten()/relative.numel()**.5,
                relative[frozen].flatten()/relative[frozen].numel()**.5]
        return torch.cat(values).double(),stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('collection','material','calibration','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads(a.calibration.read_text())
    provenance=json.loads((a.collection/'nominal_a/provenance.json').read_text())
    teacher=Path(provenance['args'][provenance['args'].index('--teacher')+1])
    if sha(teacher)!=provenance['teacher_sha256']:
        raise ValueError('Changed fixture source identity')
    asset=torch.load(teacher,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
    contract=validate_contract(json.loads(a.material.read_text()),asset)
    table=torch.tensor(contract['effective'])
    strings=torch.tensor([name.startswith('Collider/strip_') for name in contract['component_names']])
    if (not torch.equal(table[:,0],table[:,1]) or
            any(len(table[m].unique(dim=0))!=1 for m in (strings,~strings))):
        raise ValueError('Expected the two measured material groups')
    original=torch.stack([table[strings][0,2],table[~strings][0,2],table[strings][0,1],table[~strings][0,1]])
    prior=BallImpulsePriorV1().eval()
    train=PairedMaterialFit(a.collection,'train',prior,cfg,table,strings)
    x=original.clone()
    residual,anchor=train.evaluate(x)
    def admissible(stats):
        return (stats['absolute_rmse']<=anchor['absolute_rmse'] and
                stats['all_signed']['absolute_rmse']<=anchor['all_signed']['absolute_rmse'])
    history=[]
    delta=torch.eye(4)*torch.tensor([.01,.01,.005,.01])
    for it in range(3):
        columns=[]
        for step in delta:
            upper,lower=(x+step).clamp(0,1),(x-step).clamp(0,1)
            plus,minus=train.evaluate(upper)[0],train.evaluate(lower)[0]
            columns.append((plus-minus)/(upper-lower).sum())
        jac=torch.stack(columns,1)
        dx=torch.linalg.solve(jac.T@jac+.02*torch.eye(4,dtype=jac.dtype),-jac.T@residual).float().clamp(-.12,.12)
        accepted=False
        for fraction in (1.,.5,.25,.125):
            proposal=(x+dx*fraction).clamp(0,1)
            after,stats=train.evaluate(proposal)
            history.append(dict(iteration=it,fraction=fraction,params=proposal.tolist(),train=stats,
                                objective=float(after.square().sum())))
            if admissible(stats) and after.square().sum()<residual.square().sum():
                x,residual=proposal,after
                accepted=True
                break
        if not accepted:
            break
    dev=PairedMaterialFit(a.collection,'development',prior,cfg,table,strings)
    before=dev.evaluate(original)[1]
    after=dev.evaluate(x)[1]
    v=after['stable_signed']
    checks=dict(absolute=after['absolute_rmse']<=before['absolute_rmse'],
        signed=after['all_signed']['absolute_rmse']<=before['all_signed']['absolute_rmse'],
        stable_improvement=v['absolute_rmse']<before['stable_signed']['absolute_rmse'],
        direction=v['cosine_median']>=.8 and v['cosine_p10']>=.5,gain=.5<=v['gain_median']<=2.)
    result=dict(parameters=['string_restitution','solid_restitution','string_friction','solid_friction'],
        source=original.tolist(),selected=x.tolist(),anchor_train=anchor,train_search=history,development=dict(before=before,after=after),
        checks=checks,eligible_for_reserved=all(checks.values()),material_sha256=sha(a.material),
        calibration_sha256=sha(a.calibration),split_hashes={s:sha(a.collection/(s+'.json')) for s in ('train','development')},
        scope='Effective model coefficients at fixed normal mass scale, native racket supplied, shuttle propagated from initial state',
        final_holdout_opened=False,native_material_changed=False,policy_updates=0,contact_gradients_admitted=False)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
