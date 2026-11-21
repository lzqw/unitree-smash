"""Test one spatial coefficient for racket sticking strength with fixed contact, direction and friction cap."""
import argparse
import json
from pathlib import Path

import torch

from scripts.audit_supermimic_native_window_response import complete_window
from scripts.calibrate_supermimic_toi_slip import TOISlip, friction_errors, integrated_friction, signed_training_capture
from scripts.evaluate_supermimic_contact_rollout import sha
from scripts.fit_supermimic_native_racket_toi_head import fixture, free_rollout, g07
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


class SpatialSticking(TOISlip):
    def __init__(self, friction, sticking, slope, axes):
        super().__init__(friction, sticking)
        self.slope = slope
        self.axes = axes

    def toi_response(self, row, present):
        if self.slope == 0:
            return super().toi_response(row, present)
        local = row['local']
        radius_squared = (local[:, 1:3]/local.new_tensor(self.axes)).square().sum(-1,keepdim=True)
        # The bounded function is zero at the centre of the string bed.
        factor = 1 + self.slope * radius_squared/(1+radius_squared)
        n = row['normal']
        v = row['relative_velocity']
        tangent = v - (n*v).sum(-1,keepdim=True)*n
        cap = self.friction*row['impulse_dv'].norm(dim=-1,keepdim=True)
        scale = torch.minimum(self.sticking*factor, cap/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-12))
        active = present.reshape(-1).bool() & row['active'] & (row['object_id']==0)
        return dict(impulse_residual_active=active,impulse_residual_dv=-tangent*scale*active[:,None])


def fresh_training(collection, receipt, fit_hash, prior, physics):
    read=lambda p:json.loads(p.read_text())
    support=read(receipt)
    if (not support['training_only'] or support['policy_training_admitted'] or support['contact_receipt']
            or support['physics_fit_sha256']!=fit_hash or support['status_sha256']!=sha(collection/'status.json')):
        raise ValueError('Training source registration or score changed')
    if support['counts'].get('nominal_mismatch',0):
        raise ValueError('Repeated nominal contact branches differ on this native training panel')
    if not support['independent_stable_with_response']:
        raise ValueError('No repeatable signed response from stable racket cases in the new training source')
    data,groups,noise,_,_,commands=g07(collection,prior)
    folder=collection/read(collection/'status.json')['completed'][0]['spec']
    observations={}
    for branch in ('minus','plus'):
        run=_load_branch(folder,branch)
        traces=complete_window(run['raw'],run['spec'],prior.steps)
        if run['spec']['lanes']!=commands['lanes'] or run['spec']['contact_training_panel']['case_sha256']!=support['case_sha256']:
            raise ValueError('Native lane or launch identity differs from scored source')
        values=[]
        for tr in traces:
            if tr['physics_dt_s']!=prior.sub_dt or tr['contact_metadata']['friction_source']!='physx_filtered_racket_pair':
                raise ValueError('Changed clock or contact sensor')
            values.append(tr['tensors']['racket_friction_force_w'][1:].sum(0)*(prior.sub_dt/physics['mass_kg']))
        observations[branch]=torch.cat(values)
        if observations[branch].shape!=data[branch]['target'].shape or not observations[branch].isfinite().all():
            raise ValueError('Invalid signed friction data')
    return (data,groups,noise),observations


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','calibration','fixture','training','development','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--new-training',type=Path)
    parser.add_argument('--support',type=Path)
    args=parser.parse_args()
    if bool(args.new_training)!=bool(args.support):
        parser.error('Provide both the collected training source and its signed support score')
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    fit=read(args.fit)
    configs=Path(__file__).resolve().parents[1]/'configs'
    physics=read(configs/'shuttle_physics_v2.json')
    axes=read(configs/'full_racket_v4_r3.json')['assumptions']['strings_semiaxes_yz_m']
    development=read(args.development/'completed.json')
    if (sha(configs/'shuttle_physics_v2.json')!=fit['physics_sha256'] or sha(args.calibration)!=fit['calibration_sha256']
            or fit['g07_sha256']!=development['wide'] or sha(args.training/'status.json')!=fit['g07_training']['status_sha256']
            or sha(args.fixture/'train.json')!=fit['fixture_sha256']['train']):
        raise ValueError('Training sources or reference changed')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    def model(slope):
        return SpatialSticking(fit['selected']['friction'],fit['selected']['sticking'],slope,axes)
    signed,labels,_=signed_training_capture(args.training,args.development/'wide',prior,physics)
    panels={'fixture':(fixture(args.fixture,'train',prior),integrated_friction(args.fixture,'train',physics,prior)),
            'original_training':(signed,labels)}
    if args.new_training:
        panels['new_training']=fresh_training(args.new_training,args.support,sha(args.fit),prior,physics)
    refs={k:friction_errors(model(0.),*v) for k,v in panels.items()}
    grid=[]
    for slope in (0.,-.75,-.5,-.25,-.125,.125,.25,.5,.75):
        errors={k:friction_errors(model(slope),*v) for k,v in panels.items()}
        relative=[]
        for name,row in errors.items():
            ratios=[value/refs[name][k] for k,value in row.items() if refs[name][k] is not None and refs[name][k]>0]
            relative.append(sum(ratios)/len(ratios))
        grid.append(dict(slope=slope,source_errors=errors,score=sum(relative)/len(relative)))
    chosen=min(grid,key=lambda x:x['score'])
    results={}
    for scale in ('wide','half','quarter'):
        folder=args.development/scale
        if sha(folder/'status.json')!=development[scale]:
            raise ValueError('Development source changed')
        _,_,repeat,groups,seq,commands=g07(folder,prior)
        results[scale]=free_rollout(model(chosen['slope']),prior,seq,groups,repeat,commands,baseline_model=model(0.))
    if results['wide']['baseline']!=fit['g07']['candidate']:
        raise ValueError('Frozen first-impact model failed to reproduce')
    result=dict(fit_sha256=sha(args.fit),new_training_support_sha256=sha(args.support) if args.support else None,
                geometry_sha256=sha(configs/'full_racket_v4_r3.json'),train=grid,selected=chosen,development=results,
                actor_updates=0,v13_updates=0,contact_receipt=False,final_holdout_opened=False)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
