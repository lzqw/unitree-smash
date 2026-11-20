#!/usr/bin/env python3
"""Test native per-shape materials and propagated sphere spin on signed one-control-tick trials."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_paired_racket_solver import read_split
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


@torch.no_grad()
def evaluate(directory, audit, split, model, cfg, material, spin, slip, *, compound=False):
    rows, data = read_split(audit, directory, split)
    seeds = sorted({r['seed'] for r in rows})
    pred, absolute = {}, []
    for branch, batch in data.items():
        outputs = []
        for seed,(ball, truth, stream) in zip(seeds,batch,strict=True):
            tr=torch.load(directory/branch/f'spec_{seed}'/'trace.pt',mmap=True,weights_only=True)
            incoming_spin = tr['raw']['substep_trace']['tensors']['shuttle_state_w'][0,:,10:13] if spin else None
            endpoint,_ = tick(model,ball,[item[0] for item in stream],[item[-1] for item in stream],
                              cfg['fitted'],cfg['solver']['pair_contact_offset_m'],
                              torch.zeros(len(ball),dtype=torch.bool) if slip else None,
                              measured_path=[item[:-1] for item in stream],racket_material=material,initial_spin=incoming_spin,compound_racket=compound)
            outputs.append(endpoint[:,3:])
            absolute.append(endpoint[:,3:]-truth)
        pred[branch]=torch.cat(outputs)
    diff=pred['plus']-pred['minus']
    ref=torch.tensor([r['delta']['native'] for r in rows])
    repeat=torch.tensor([r['repeat']['native'] for r in rows])
    mask=torch.tensor([r['kind']=='racket_stable' for r in rows])
    robust=F.smooth_l1_loss((diff-ref)/ref.norm(dim=-1,keepdim=True).clamp_min(.05),torch.zeros_like(diff),reduction='none').mean(-1)
    return dict(all_signed=response_metrics(ref,diff,repeat),stable_signed=response_metrics(ref[mask],diff[mask],repeat[mask]),
                absolute_rmse=float(torch.cat(absolute).square().mean().sqrt()),balanced_loss=float((robust.mean()+robust[mask].mean())/2))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('collection','material','train','development','calibration','output'):
        cli.add_argument('--'+k,required=True,type=Path)
    a=cli.parse_args();torch.set_num_threads(3)
    cfg=json.loads(a.calibration.read_text())
    prov=json.loads((a.collection/'nominal_a'/'provenance.json').read_text())
    path=Path(prov['args'][prov['args'].index('--teacher')+1])
    if sha(path)!=prov['teacher_sha256']:
        raise ValueError('Native trial ancestry changed')
    asset=torch.load(path,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
    material=torch.tensor(validate_contract(json.loads(a.material.read_text()),asset)['effective'])
    model=BallImpulsePriorV1().eval()
    train={};settings={}
    for native_material in (False,True):
        for spin in (False,True):
            for slip in (True,False):
                key=f'material_{int(native_material)}_spin_{int(spin)}_initial_slip_{int(slip)}'
                settings[key]=(material if native_material else None,spin,slip)
                train[key]=evaluate(a.collection,a.train,'train',model,cfg,*settings[key])
    anchor='material_0_spin_0_initial_slip_1'
    passed=[k for k,v in train.items() if v['absolute_rmse']<=train[anchor]['absolute_rmse']]
    best=min(passed,key=lambda k:train[k]['balanced_loss'])
    dev={k:evaluate(a.collection,a.development,'development',model,cfg,*settings[k]) for k in dict.fromkeys((anchor,best))}
    ref=dev[anchor];v=dev[best];signed=v['stable_signed']
    checks=dict(absolute=v['absolute_rmse']<=ref['absolute_rmse'],all_signed=v['all_signed']['absolute_rmse']<=ref['all_signed']['absolute_rmse'],
                improved=signed['absolute_rmse']<ref['stable_signed']['absolute_rmse'],
                direction=signed['cosine_median']>=.8 and signed['cosine_p10']>=.5,gain=.5<=signed['gain_median']<=2)
    with a.output.open('x') as f:
        json.dump(dict(train=train,development=dev,selected=best,checks=checks,train_sha256=sha(a.train),
                       development_sha256=sha(a.development),material_sha256=sha(a.material),calibration_sha256=sha(a.calibration),
                       final_holdout_opened=False,eligible_for_reserved_panel=all(checks.values()),
                       scope='Native racket trajectory over a single tick, ball autoregressed after initialization',contact_bptt_admitted=False),f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
