#!/usr/bin/env python3
"""Differentiate the full held G07 executed sequence and the corresponding measured racket motion."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from audit_supermimic_surface_path_jacobian import interpolate_motion
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,PhysicalWorldV2,racket,sha,tick
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def differentiate(branches,windows,body,prior,material,cfg,launch,positions,mode):
    raw=[x['raw'] for x in branches]
    ids=branches[0]['spec']['lanes']
    lanes=[ids[i] for i in positions]
    init=raw[0]['start']['before']['physical']
    state=encode_physical(init)[lanes]
    before=encode_physical(raw[1]['start']['before']['physical'])[lanes]
    torch.testing.assert_close(state,before,rtol=0,atol=0)
    previous,feet,elapsed,present=(init[k][lanes] for k in ('final_action','foot_contact','elapsed_s','ball_present'))
    feet=feet.float()
    spin=windows[0][0]['tensors']['shuttle_state_w'][0,positions,10:13]
    torch.testing.assert_close(spin,windows[1][0]['tensors']['shuttle_state_w'][0,positions,10:13],atol=0,rtol=0)
    alpha=state.new_full((len(lanes),),.5,requires_grad=True)
    opt=dict(racket_material=material,return_spin=True,compound_racket=True)
    clock=cfg['solver']['pair_contact_offset_m']
    seqs=[x['step_sequence'] for x in raw]
    for i,(minus,plus) in enumerate(zip(*seqs,strict=True)):
        if mode=='body_racket':
            targets=torch.lerp(minus['final_joint_targets'][lanes],plus['final_joint_targets'][lanes],alpha[:,None])
            def step(state,spin,target,previous,feet,elapsed):
                robot,logits,_=body(state,target,previous,feet,present,elapsed,launch[lanes])
                ball,_,w=tick(prior,state[:,49:],racket(state,body.geometry),racket(robot,body.geometry),
                              cfg['fitted'],clock,initial_spin=spin,**opt)
                return torch.cat((robot[:,:49],ball),-1),w,logits.sigmoid()
            state,spin,feet=checkpoint(step,state,spin,targets,previous,feet,elapsed,use_reentrant=False)
            previous=torch.lerp(minus['final_action'][lanes],plus['final_action'][lanes],alpha[:,None])
            elapsed=elapsed+prior.dt
        else:
            motion=[]
            for traces in windows:
                t=traces[i]['tensors']
                motion.append([t[k][:,positions] for k in ('racket_position_w','racket_quaternion_w',
                                                          'racket_linear_velocity_w','racket_angular_velocity_w')])
            # Actual endpoint samples are available. Each substep consumes its left endpoint.
            def step(ball,spin,alpha,paths=motion):
                stream=interpolate_motion(*paths,alpha)
                out,_,w=tick(prior,ball,[v[0] for v in stream],[v[-1] for v in stream],cfg['fitted'],clock,
                             measured_path=[v[:-1] for v in stream],initial_spin=spin,**opt)
                return out,w
            ball,spin=checkpoint(step,state[:,49:],spin,alpha,use_reentrant=False)
            state=torch.cat((state[:,:49],ball),-1)
    derivative=[]
    for j in range(3):
        g,=torch.autograd.grad(state[:,52+j].sum()+0*alpha.sum(),alpha,retain_graph=j<2)
        derivative.append(g.detach())
    return torch.stack(derivative,-1)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','response','calibration','material','body','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--batch',type=int,default=8)
    a=cli.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    report=read(a.response);status=read(a.collection/'status.json')
    if (status['status']!='COLLECTION_FINISHED' or sha(a.collection/'status.json')!=report['status_sha256']
            or (sha(a.calibration),sha(a.material),sha(a.body))!=(report['calibration_sha256'],report['material_sha256'],report['body_sha256'])
            or a.batch<=0):raise ValueError('Native run or transition identity mismatch')
    paths={row['branch']:a.collection/row['spec'] for row in status['completed']}
    for row in status['completed']:
        if sha(paths[row['branch']]/row['branch']/'native_probe.pt')!=row['probe_sha256']:
            raise ValueError('Physical trajectory changed')
    runs={b:_load_branch(folder,b) for b,folder in paths.items()}
    _same_contract(runs)
    minus,plus=runs['minus'],runs['plus']
    task=minus['task_contract'];spec=minus['spec']
    if task!=report['task_contract'] or report['prefix_sha256']!=minus['prefix_sha256']:
        raise ValueError('Different native G07 task or initial history')
    suite=Path(task['suite']['path'])
    if sha(suite)!=task['suite']['sha256']:raise ValueError('Case assignments changed')
    launch=torch.tensor([r['launch']['position_m']+r['launch']['velocity_mps'] for r in read(suite)['cases']])
    prior=BallImpulsePriorV1().eval().requires_grad_(False)
    body=PhysicalWorldV2(task['asset']['sha256']).eval().requires_grad_(False)
    body.load_state_dict(torch.load(a.body,weights_only=True,map_location='cpu')['model'],strict=True)
    mat=torch.tensor(validate_contract(read(a.material),task['asset']['sha256'])['effective'])
    cases=report['cases'];n=len(cases)
    if [c['lane'] for c in cases]!=spec['lanes']:raise ValueError('Case order changed')
    windows=[complete_window(r['raw'],spec,prior.steps) for r in (minus,plus)]
    t=str(spec['tick']);cfg=read(a.calibration)
    end=lambda key:torch.tensor([c['ticks'][t][key] for c in cases])
    native=end('native');noise=end('native_repeat')
    groups={};by_lane={}
    for mode in ('body_racket','measured_racket'):
        local=torch.cat([differentiate([minus,plus],windows,body,prior,mat,cfg,launch,
                           list(range(i,min(i+a.batch,n))),mode) for i in range(0,n,a.batch)])
        finite=local.isfinite().all(-1)
        by_lane[mode]=[v.tolist() if valid else None for v,valid in zip(local,finite)]
        groups[mode]={}
        for name in ['all',*sorted({c['group'] for c in cases})]:
            mask=torch.tensor([name=='all' or c['group']==name for c in cases])
            keep=mask&finite
            row=dict(rows=int(mask.sum()),nonfinite=int((mask&~finite).sum()))
            if keep.any():
                row['native_vs_local']=response_metrics(native[keep],local[keep],noise[keep])
                row['secant_vs_local']=response_metrics(end(mode)[keep],local[keep],torch.zeros_like(noise[keep]))
            groups[mode][name]=row
    with a.output.open('x') as f:
        json.dump(dict(lanes=spec['lanes'],local_response=by_lane,groups=groups,source_sha256=sha(a.response),
                       derivative='d(final shuttle velocity)/d(alpha), alpha=0.5 along recorded minus-to-plus trajectories',
                       actor_updates=0,contact_bptt_admitted=False),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
