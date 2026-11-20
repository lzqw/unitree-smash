#!/usr/bin/env python3
"""Calibrate tangential response on signed native moving-surface trials with family-separated validation."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_paired_body_transfer import response_metrics
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick


def read_split(report, root, split):
    audit=json.loads(report.read_text())
    design_path=root/'design.json'; design=json.loads(design_path.read_text())
    if audit['design_sha256']!=sha(design_path) or audit['split']!=split or audit['policy_training_admitted']:
        raise ValueError('Mismatched or unverified fixture audit')
    rows=sorted(audit['samples'],key=lambda r:(r['seed'],r['lane']))
    seeds=sorted(int(s) for s,k in design['assignments'].items() if k==split)
    if [(r['seed'],r['lane']) for r in rows]!=[(s,l) for s in seeds for l in range(design['lanes'])]:
        raise ValueError('Incomplete family or case coverage')
    branches={b:[] for b in ('minus','plus')}
    for b in branches:
        for s in seeds:
            path=root/b/f'spec_{s}'/'trace.pt'
            if sha(path)!=audit['archives'][str(path)]:raise ValueError('Fixture archive changed')
            trial=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            tr=trial['raw']['substep_trace']['tensors']
            stream=[tr[k] for k in ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')]
            initial=tr['shuttle_state_w'][0]
            origin=fixture_court_origin(trial,initial[:,:3],design)
            stream[0]=stream[0]-origin
            branches[b].append((torch.cat((initial[:,:3]-origin,initial[:,7:10]),-1),tr['shuttle_state_w'][-1,:,7:10],stream))
    return rows,branches


def evaluate(rows,branches,prior,settings):
    outputs=[]; absolute=[]
    for b in ('minus','plus'):
        result=[]
        for ball,label,stream in branches[b]:
            endpoint,_=tick(prior,ball,[v[0] for v in stream],[v[-1] for v in stream],settings['fitted'],
                settings['solver']['pair_contact_offset_m'],torch.zeros(len(ball),dtype=torch.bool),
                measured_path=[v[:-1] for v in stream])
            result.append(endpoint[:,3:])
            absolute.append(endpoint[:,3:]-label)
        outputs.append(torch.cat(result))
    delta=outputs[1]-outputs[0]
    native=torch.tensor([r['delta']['native'] for r in rows])
    noise=torch.tensor([r['repeat']['native'] for r in rows])
    stable=torch.tensor([r['kind']=='racket_stable' for r in rows])
    if not stable.any():raise ValueError('No stable contacts to calibrate')
    loss=F.smooth_l1_loss((delta-native)/native.norm(dim=-1,keepdim=True).clamp_min(.05),
                          torch.zeros_like(delta),reduction='none').mean(-1)
    return dict(case_count=len(rows),stable=int(stable.sum()),all_signed=response_metrics(native,delta,noise),
                stable_signed=response_metrics(native[stable],delta[stable],noise[stable]),
                absolute_rmse_mps=float(torch.cat(absolute).square().mean().sqrt()),
                balanced_response_loss=float(.5*loss.mean()+.5*loss[stable].mean()))


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','train','development','calibration','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    a=cli.parse_args();torch.set_num_threads(3)
    prior=BallImpulsePriorV1().eval();cfg=json.loads(a.calibration.read_text())
    design=json.loads((a.collection/'design.json').read_text())
    families=[set(design[n]) for n in ['calibration_families','development_families','reserved_final_families']]
    if any(families[i]&families[j] for i in range(3) for j in range(i)):raise ValueError('Family split overlap')
    rows,data=read_split(a.train,a.collection,'train')
    base=evaluate(rows,data,prior,cfg)
    grid=[];best=None
    for mu in (.08,cfg['fitted']['friction'],.25):
        for gain in (.2,2/7,.36):
            fit=dict(cfg,fitted=dict(cfg['fitted'],friction=mu,sticking_gain=gain))
            stats=base if (mu,gain)==(cfg['fitted']['friction'],2/7) else evaluate(rows,data,prior,fit)
            grid.append(dict(fitted=fit['fitted'],metrics=stats))
            if stats['absolute_rmse_mps']<=base['absolute_rmse_mps'] and (
                    best is None or stats['balanced_response_loss']<best[0]):
                best=stats['balanced_response_loss'],fit
    rows,data=read_split(a.development,a.collection,'development')
    before=evaluate(rows,data,prior,cfg);after=evaluate(rows,data,prior,best[1])
    metrics=after['stable_signed']
    checks=dict(absolute_forward=after['absolute_rmse_mps']<=before['absolute_rmse_mps'],
        signed_rmse=metrics['absolute_rmse']<before['stable_signed']['absolute_rmse'],
        all_signed=after['all_signed']['absolute_rmse']<=before['all_signed']['absolute_rmse'],
        direction=metrics['cosine_median']>=.8 and metrics['cosine_p10']>=.5,
        amplitude=.5<=metrics['gain_median']<=2)
    out=dict(source_calibration_sha256=sha(a.calibration),train_sha256=sha(a.train),development_sha256=sha(a.development),
             grid=grid,selected=best[1]['fitted'],development_baseline=before,development_candidate=after,
             checks=checks,eligible_for_reserved_evaluation=all(checks.values()),contact_bptt_admitted=False,
             policy_updates=0)
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
