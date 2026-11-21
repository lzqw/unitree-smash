"""Compare the Body plus calibrated TOI model derivative and secants on identical executed command paths."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_toi_slip import TOISlip, body_transfer
from evaluate_supermimic_contact_rollout import racket, sha
from fit_supermimic_native_racket_toi_head import g07, predicted_state, racket_transition_inputs
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def command_rollout(body,prior,slip,initial,launch,traces,lanes,alpha):
    state=encode_physical(initial)[lanes]
    previous,feet,elapsed,present=(initial[k][lanes] for k in ('final_action','foot_contact','elapsed_s','ball_present'))
    def command(minus,plus,key):
        return torch.lerp(minus[key][lanes],plus[key][lanes],alpha[:,None])
    for minus,plus in zip(traces['minus'],traces['plus'],strict=True):
        target=command(minus,plus,'final_joint_targets')
        robot,contact,next_present=body(state,target,previous,feet.float(),present,elapsed,launch)
        if not (present & next_present).all():
            raise ValueError('Expected a launched shuttle throughout the tested command prefix')
        endpoints=[torch.stack((first,last)) for first,last in zip(racket(state,body.geometry),racket(robot,body.geometry),strict=True)]
        row=racket_transition_inputs(prior,state[:,49:],endpoints)
        state=torch.cat((robot[:,:49],predicted_state(slip,prior,row)),1)
        previous,feet,elapsed,present=command(minus,plus,'final_action'),contact.sigmoid(),elapsed+prior.dt,next_present
    return state[:,52:55]


def derivative_summary(native,noise,signals,mask):
    results={}
    for name,value in signals.items():
        finite=mask & value.isfinite().all(-1)
        summary=dict(total=int(mask.sum()),nonfinite=int((mask & ~finite).sum()))
        if finite.any():
            summary['vs_native']=response_metrics(native[finite],value[finite],noise[finite])
            ref=signals['model_signed'][finite]
            summary['vs_model_signed']=response_metrics(ref,value[finite],torch.zeros_like(ref))
        results[name]=summary
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('body','world-replay','source-stage','physics-fit','calibration','development','output'):
        parser.add_argument('--'+k,type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    fit=read(a.physics_fit)
    design=read(a.development/'completed.json')
    replay=read(a.world_replay/'fit.json')
    weights=torch.load(a.world_replay/'online_world.pt',mmap=True,weights_only=True,map_location='cpu')
    parent=torch.load(a.body,mmap=True,weights_only=True,map_location='cpu')
    task=parent['source_manifest']['task_contract']
    stage=torch.load(a.source_stage,mmap=True,weights_only=False,map_location='cpu')
    if (fit['final_holdout_opened'] or fit['g07_sha256']!=design['wide'] or fit['calibration_sha256']!=sha(a.calibration)
            or sha(a.source_stage)!=replay['stage_checkpoint_sha256'] or tensor_digest(weights.items())!=replay['online_world_sha256']
            or stage['metadata']['physical_imagination']['checkpoint_sha256']['physical']!=sha(a.body)):
        raise ValueError('Unverified model or development collection')
    body=PhysicalWorldV2(task['asset']['sha256']).eval().requires_grad_(False)
    body.load_state_dict({k.removeprefix('physical_model.'):v for k,v in weights.items() if k.startswith('physical_model.')})
    suite=Path(task['suite']['path'])
    if sha(suite)!=task['suite']['sha256']:
        raise ValueError('Changed launches')
    launches=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in read(suite)['cases']])
    prior=BallImpulsePriorV1(normal_impulse_scale=read(a.calibration)['fitted']['normal_impulse_scale']).eval()
    slip=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    scales={}
    scope=None
    for divisor,name in ((1,'wide'),(2,'half'),(4,'quarter')):
        folder=a.development/name
        if sha(folder/'status.json')!=design[name]:
            raise ValueError('Development source changed')
        _,_,repeat,groups,seq,commands=g07(folder,prior)
        lanes=commands['lanes']
        source=folder/read(folder/'status.json')['completed'][0]['spec']
        runs={b:_load_branch(source,b) for b in ('minus','plus')}
        decl=dict(runs['plus']['spec'])
        amplitude=decl.pop('epsilon_request')*divisor
        identity=(decl,runs['plus']['seed'],amplitude,read(folder/'startup.json')['source_sha256'])
        if scope is None:
            scope=identity
        if (scope!=identity or decl['data_role']!='development_action_response' or runs['plus']['task_contract']!=task
                or not set(lanes)<=set(replay['excluded_lanes']) or set(lanes)&set(parent['calibration']['train_lanes'])):
            raise ValueError('Different case split, command direction, or physical contract')
        traces={b:r['raw']['step_sequence'] for b,r in runs.items()}
        initial=runs['minus']['raw']['start']['before']['physical']
        def velocity(x):
            return command_rollout(body,prior,slip,initial,launches[lanes],traces,lanes,x)
        alpha=torch.full((len(lanes),),.5,requires_grad=True)
        out=velocity(alpha)
        gradients=[]
        for component in range(3):
            gradients.append(torch.autograd.grad(out[:,component].sum(),alpha,retain_graph=component<2)[0])
        with torch.no_grad():
            derivative=torch.stack(gradients,1)
            secant=velocity(torch.ones_like(alpha))-velocity(torch.zeros_like(alpha))
            near=(velocity(torch.full_like(alpha,.55))-velocity(torch.full_like(alpha,.45)))/.10
            check=body_transfer(folder,a.body,prior,slip,groups,repeat,seq,commands,a.world_replay,a.source_stage)
            torch.testing.assert_close(secant,torch.tensor(check['predicted_delta_mps']),rtol=1e-4,atol=1e-5)
            native=torch.tensor(check['native_delta_mps'])
            noise=repeat[-len(lanes):]
            stable=torch.tensor([g=='racket_stable' for g in groups])
            signals=dict(autograd=derivative,central_difference=near,model_signed=secant)
            stats={k:derivative_summary(native,noise,signals,m) for k,m in (
                ('all',torch.ones_like(stable)),('stable',stable),('other_modes',~stable))}
            def vector(v):
                return v.tolist() if torch.isfinite(v).all() else None
            cases=[dict(lane=lane,mode=groups[i],native=vector(native[i]),noise=vector(noise[i]),
                        **{k:vector(v[i]) for k,v in signals.items()}) for i,lane in enumerate(lanes)]
            scales[name]=dict(stats=stats,cases=cases,expected_forward=check['metrics'])
    with a.output.open('x') as f:
        json.dump(dict(scales=scales,world_sha256=replay['online_world_sha256'],physics_fit_sha256=sha(a.physics_fit),
                       source_status_sha256=design,actor_updates=0,v13_updates=0,contact_receipt=False,
                       units='m/s per full signed executed interval at alpha=0.5; both executed targets and previous final-action context follow their recorded paths'),
                  f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
