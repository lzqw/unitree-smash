"""Calibrate a bounded transition from sticking to sliding on measured native friction impulses."""
import argparse
import json
from pathlib import Path

import torch

from calibrate_supermimic_toi_slip import (TOISlip, body_transfer, evaluate, friction_errors,
                                           integrated_friction, signed_training_capture)
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_native_racket_toi_head import fixture, free_rollout, g07
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


class PartialSlip(TOISlip):
    def __init__(self, friction, sticking, power):
        super().__init__(friction, sticking)
        self.power = power

    def toi_response(self, row, present):
        out = super().toi_response(row,present)
        if self.power is None:
            return out
        slip = row['relative_velocity'] - (row['relative_velocity']*row['normal']).sum(-1,keepdim=True)*row['normal']
        stick = self.sticking*slip.norm(dim=-1,keepdim=True)
        slide = self.friction*row['impulse_dv'].norm(dim=-1,keepdim=True)
        ratio = torch.minimum(stick,slide)/torch.maximum(stick,slide).clamp_min(1.e-12)
        out['impulse_residual_dv'] = out['impulse_residual_dv']*(1+ratio.pow(self.power)).pow(-1/self.power)
        return out


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('fixture','calibration','fit','training','development','body','online-body','stage','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    fit=read(args.fit)
    if (fit['calibration_sha256']!=sha(args.calibration) or fit['g07_sha256']!=sha(args.development/'status.json')
            or fit['g07_training']['status_sha256']!=sha(args.training/'status.json')
            or fit['fixture_sha256']!={s:sha(args.fixture/(s+'.json')) for s in ('train','development')}):
        raise ValueError('Unexpected calibration split')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    physical=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(physical)!=fit['physics_sha256']:
        raise ValueError('Shuttle physics mismatch')
    physics=read(physical)
    dataset=fixture(args.fixture,'train',prior)
    signed=integrated_friction(args.fixture,'train',physics,prior)
    gset,gobs,identity=signed_training_capture(args.training,args.development,prior,physics)
    panels=[(dataset,signed),(gset,gobs)]
    references=[friction_errors(TOISlip(0.,0.),data,obs) for data,obs in panels]
    choices=[]
    for row in fit['train_scores']:
        for power in (None,8,4,2,1):
            model=PartialSlip(row['friction'],row['sticking'],power)
            source_scores=[]
            for (data,obs),ref in zip(panels,references,strict=True):
                err=friction_errors(model,data,obs)
                terms=[err[k]/v for k,v in ref.items() if v is not None and v>0]
                source_scores.append(sum(terms)/len(terms))
            score=sum(source_scores)/len(source_scores)
            if power is None and abs(score-row['score'])>1.e-6:
                raise RuntimeError('Original hard-cap calibration does not reproduce')
            choices.append(dict(friction=model.friction,sticking=model.sticking,power=power,score=score))
    selected=min(choices,key=lambda v:v['score'])
    model=PartialSlip(**{k:selected[k] for k in ('friction','sticking','power')})
    development=fixture(args.fixture,'development',prior)
    _,_,noise,modes,seq,commands=g07(args.development,prior)
    oracle=free_rollout(model,prior,seq,modes,noise,commands)
    body=body_transfer(args.development,args.body,prior,model,modes,noise,seq,commands,args.online_body,args.stage)
    if body['replay_world_sha256']!=fit['online_body_racket']['replay_world_sha256']:
        raise ValueError('Body calibration differs')
    report=dict(calibration_sha256=sha(args.fit),training_sha256=identity['status_sha256'],
                selected=selected, best_by_power=[min((v for v in choices if v['power']==a),key=lambda r:r['score'])
                                                   for a in (None,8,4,2,1)],
                fixture_metrics=evaluate(prior,model,development), native_racket=oracle, body_racket=body,
                scope='Parameters fitted on measured training friction only. Development transfer, no final holdout or Actor updates.')
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
