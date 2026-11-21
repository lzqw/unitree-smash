"""Compare the Body plus calibrated TOI model derivative and secants on identical executed command paths."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_toi_slip import TOISlip, body_transfer
from evaluate_supermimic_contact_rollout import racket, sha
from fit_supermimic_native_racket_toi_head import free_rollout, g07, metrics, predicted_state, racket_transition_inputs
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest, load_physical_loop
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def command_rollout(body,prior,slip,initial,launch,traces,lanes,alpha,*,adapter=None,observed=None,channels=()):
    state=encode_physical(initial)[lanes]
    previous,feet,elapsed,present=(initial[k][lanes] for k in ('final_action','foot_contact','elapsed_s','ball_present'))
    def command(minus,plus,key):
        return torch.lerp(minus[key][lanes],plus[key][lanes],alpha[:,None])
    if observed is not None:
        if adapter is not None or torch.is_grad_enabled() or len(observed)!=len(traces['minus']):
            raise ValueError('Measured racket substitution requires a complete forward-only analytic comparison')
    for tick,(minus,plus) in enumerate(zip(traces['minus'],traces['plus'],strict=True)):
        target=command(minus,plus,'final_joint_targets')
        if adapter is None:
            robot,contact,next_present=body(state,target,previous,feet.float(),present,elapsed,launch)
            endpoints=[torch.stack((first,last)) for first,last in zip(racket(state,body.geometry),racket(robot,body.geometry),strict=True)]
            if observed is not None:
                for channel in channels:
                    endpoints[channel]=observed[tick][2][channel]
            row=racket_transition_inputs(prior,state[:,49:],endpoints)
            state=torch.cat((robot[:,:49],predicted_state(slip,prior,row)),1)
            feet=contact.sigmoid()
        else:
            state,feet,next_present=ball_transition(adapter.physical_model,adapter.ball_model,state,target,
                previous,feet.float(),present,elapsed,launch,action_model=adapter.action_ball_model)
        if not (present & next_present).all():
            raise ValueError('Expected a launched shuttle throughout the tested command prefix')
        previous,elapsed,present=command(minus,plus,'final_action'),elapsed+prior.dt,next_present
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
    for k in ('runtime-checkpoint','runtime-forward-guard','archive','safety-mapping'):
        parser.add_argument('--'+k,type=Path)
    a=parser.parse_args()
    runtime_options=(a.runtime_checkpoint,a.runtime_forward_guard,a.archive,a.safety_mapping)
    if any(runtime_options) and not all(runtime_options):
        parser.error('The full runtime transition requires a checkpoint, forward guard, archive and safety mapping')
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
    adapter=None
    runtime=None
    if a.runtime_checkpoint:
        checkpoint=torch.load(a.runtime_checkpoint,mmap=True,weights_only=False,map_location='cpu')
        meta=checkpoint['metadata']['physical_imagination']
        paths=meta['checkpoint_paths']
        _,adapter,_,runtime_launch,archive=load_physical_loop(archive=a.archive,safety_mapping=a.safety_mapping,
            physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
            action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device='cpu')
        world=tensor_digest(checkpoint['world'].items())
        guards=[json.loads(line) for line in a.runtime_forward_guard.read_text().splitlines()]
        if (not any(r['passed'] and r['world_sha256']==world and r['world_updates']==checkpoint['world_updates'] for r in guards)
                or meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256'] or archive['task_contract']!=task
                or set(checkpoint['config']['wm_holdout_lanes'])!=set(replay['excluded_lanes'])):
            raise ValueError('The requested online WM or its development split is not verified')
        torch.testing.assert_close(runtime_launch,launches,rtol=0,atol=0)
        adapter.load_state_dict(checkpoint['world'],strict=True)
        adapter.eval().requires_grad_(False)
        runtime=dict(world_sha256=world,world_updates=checkpoint['world_updates'],checkpoint_sha256=sha(a.runtime_checkpoint),
                     actor_sha256=tensor_digest(checkpoint['actor'].items()),action_ball_schema=getattr(adapter.action_ball_model,'schema',None))
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
            return command_rollout(body,prior,slip,initial,launches[lanes],traces,lanes,x,adapter=adapter)
        alpha=torch.full((len(lanes),),.5,requires_grad=True)
        out=velocity(alpha)
        gradients=[]
        for component in range(3):
            gradients.append(torch.autograd.grad(out[:,component].sum(),alpha,retain_graph=component<2)[0])
        with torch.no_grad():
            derivative=torch.stack(gradients,1)
            upper=velocity(torch.ones_like(alpha))
            lower=velocity(torch.zeros_like(alpha))
            secant=upper-lower
            near=(velocity(torch.full_like(alpha,.55))-velocity(torch.full_like(alpha,.45)))/.10
            if adapter is None:
                check=body_transfer(folder,a.body,prior,slip,groups,repeat,seq,commands,a.world_replay,a.source_stage)
                torch.testing.assert_close(secant,torch.tensor(check['predicted_delta_mps']),rtol=1e-4,atol=1e-5)
                native=torch.tensor(check['native_delta_mps'])
                forward=check['metrics']
            else:
                from fit_supermimic_native_probe_response import response
                pairs=dict(state=encode_physical(initial)[lanes],feet=initial['foot_contact'][lanes].float(),
                    previous_action=initial['final_action'][lanes],present=initial['ball_present'][lanes],
                    elapsed=initial['elapsed_s'][lanes],launch=launches[lanes])
                for branch,ticks in traces.items():
                    for dest,key in [('step_targets_','final_joint_targets'),('step_final_actions_','final_action')]:
                        pairs[dest+branch]=torch.stack([t[key][lanes] for t in ticks],dim=1)
                independent=response(adapter.action_ball_model,adapter.physical_model,adapter.ball_model,pairs)
                torch.testing.assert_close(secant,independent,rtol=1e-4,atol=1e-5)
                minus,plus=(seq[b][-1][1] for b in ('minus','plus'))
                native=plus-minus
                forward=dict(final_ball_velocity_rmse_mps=float(torch.cat((upper-plus,lower-minus)).square().mean().sqrt()))
            noise=repeat[-len(lanes):]
            comparison=None
            if adapter is not None:
                # Keep the exact deployed Body weights, four native prefixes, and metric reduction fixed.
                native_labels=({'minus':{'target':minus},'plus':{'target':plus}},groups,noise)
                hybrid={b:command_rollout(adapter.physical_model,prior,slip,initial,launches[lanes],traces,lanes,x)
                        for b,x in [('minus',torch.zeros_like(alpha)),('plus',torch.ones_like(alpha))]}
                native_racket=free_rollout(slip,prior,seq,groups,repeat,commands)
                comparison=dict(deployed=metrics(native_labels,{'minus':lower,'plus':upper}),
                                same_body_calibrated_contact=metrics(native_labels,hybrid),
                                measured_racket_calibrated_contact=native_racket['candidate'],
                                calibrated_body_signed=(hybrid['plus']-hybrid['minus']).tolist(),
                                native_racket_cases=native_racket['cases'])
                comparison['measured_channels']={}
                for label,channels in [('center',(0,)),('rotation',(1,)),('twist',(2,3)),('all',(0,1,2,3))]:
                    preds={}
                    for branch,command in [('minus',torch.zeros_like(alpha)),('plus',torch.ones_like(alpha))]:
                        preds[branch]=command_rollout(adapter.physical_model,prior,slip,initial,launches[lanes],
                            traces,lanes,command,observed=seq[branch],channels=channels)
                    result=metrics(native_labels,preds)
                    if label=='all':
                        torch.testing.assert_close(torch.as_tensor(result['absolute_rmse']),
                            torch.as_tensor(native_racket['candidate']['absolute_rmse']))
                        torch.testing.assert_close(preds['plus']-preds['minus'],
                            torch.tensor([row['candidate']['delta_mps'] for row in native_racket['cases']]),rtol=1e-4,atol=1e-5)
                    comparison['measured_channels'][label]=dict(metrics=result,
                        signed_mps=(preds['plus']-preds['minus']).tolist())
            stable=torch.tensor([g=='racket_stable' for g in groups])
            signals=dict(autograd=derivative,central_difference=near,model_signed=secant)
            stats={k:derivative_summary(native,noise,signals,m) for k,m in (
                ('all',torch.ones_like(stable)),('stable',stable),('other_modes',~stable))}
            def vector(v):
                return v.tolist() if torch.isfinite(v).all() else None
            cases=[dict(lane=lane,mode=groups[i],native=vector(native[i]),noise=vector(noise[i]),
                        **{k:vector(v[i]) for k,v in signals.items()}) for i,lane in enumerate(lanes)]
            scales[name]=dict(stats=stats,cases=cases,expected_forward=forward)
            if comparison is not None:
                scales[name]['matched_contact_inputs']=comparison
    with a.output.open('x') as f:
        json.dump(dict(scales=scales,world_sha256=runtime['world_sha256'] if runtime else replay['online_world_sha256'],
                       transition='checkpoint_full' if runtime else 'body_plus_calibrated_toi',runtime=runtime,
                       physics_fit_sha256=sha(a.physics_fit),
                       source_status_sha256=design,actor_updates=0,v13_updates=0,contact_receipt=False,
                       units='m/s per full signed executed interval at alpha=0.5; both executed targets and previous final-action context follow their recorded paths'),
                  f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
