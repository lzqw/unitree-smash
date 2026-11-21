"""Check the TOI racket objective along actual executed development command sequences."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_toi_racket_objective import flight_intercept, intercept_terms, racket_features
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def score(folder, expected, adapter, template, launch, bank, task, execution, held, *, training=False):
    read=lambda p: json.loads(p.read_text())
    status=read(folder/'status.json')
    if digest(folder/'status.json')!=expected or status['status']!='COLLECTION_FINISHED':
        raise ValueError('Development data changed')
    runs={}
    for r in status['completed']:
        p=folder/r['spec']
        if digest(p/r['branch']/'native_probe.pt')!=r['probe_sha256']:
            raise ValueError('Native trace changed')
        runs[r['branch']]=_load_branch(p,r['branch'])
    if set(runs)!=set(BRANCHES):
        raise ValueError('Missing signed or nominal branch')
    _same_contract(runs)
    base=runs['nominal_a']
    spec=base['spec']
    ids=spec['lanes']
    role='training_action_identification' if training else 'development_action_response'
    valid_lanes=(set(range(128))-held) if training else held
    if (spec['data_role']!=role or not set(ids)<=valid_lanes or
            base['task_contract']!=task or base['policy_execution']!=execution):
        raise ValueError('Unexpected training or execution source')
    if not training:
        bank.assert_excludes(ids)
    state=PhysicalImaginationStateV1.from_record(base['raw']['start'],executor_template=template,launch=launch,device='cpu')
    dt=task['control_dt_s']
    ge=adapter.physical_model.geometry
    T=len(base['raw']['step_sequence'])
    intercept=flight_intercept(state,64*dt)
    elapsed=state.clock_goal_context['elapsed'].flatten()
    ref=bank.bind(state.physical,'approach',geometry=ge,elapsed=elapsed,event_time=intercept['time'],require_toi=True)
    time=intercept['time']
    hit=intercept['valid'] & ref['matched']
    native={}
    cmds={}
    for name,run in runs.items():
        seq=run['raw']['step_sequence']
        previous=run['raw']['start']['before']['physical']['final_action'][ids]
        native[name]=torch.stack([state.physical[ids],*[step['next_state'][ids] for step in seq]])
        cmds[name]=(torch.stack([step['final_joint_targets'][ids] for step in seq]),
                    torch.stack([previous,*[step['final_action'][ids] for step in seq[:-1]]]))
        if len(seq)!=T:
            raise ValueError('Different executed horizon')
    def predicted(alpha):
        targets,previous=(torch.lerp(m,p,alpha) for m,p in zip(cmds['minus'],cmds['plus'],strict=True))
        before=base['raw']['start']['before']['physical']
        x=state.physical[ids]; feet=before['foot_contact'][ids].float(); present=before['ball_present'][ids]
        stream=[x]
        for j in range(T):
            x,feet,present=ball_transition(adapter.physical_model,adapter.ball_model,x,targets[j],previous[j],
                feet,present,elapsed[ids]+j*dt,launch[ids],action_model=adapter.action_ball_model)
            # Match detached-contact TOI training. Keep physical robot kinematics differentiable.
            x=torch.cat([x[:,:49],x[:,49:].detach()],-1)
            stream.append(x)
        return torch.stack(stream)
    selections=[]
    for i,l in enumerate(ids):
        for j in range(T):
            if hit[l] and elapsed[l]+j*dt < time[l] <= elapsed[l]+(j+1)*dt:
                selections.append((i,l,j))
                break
    if not selections:
        return dict(lanes=len(ids),matched_and_inside=0,cases=[])
    indices, selected_lanes, steps = torch.tensor(selections).T
    start = elapsed[selected_lanes] + steps*dt
    end = elapsed[selected_lanes] + (steps+1)*dt
    alpha_time = (time[selected_lanes]-start)/(end-start)
    def kinematics(xs):
        return torch.lerp(racket_features(xs[steps,indices],ge),racket_features(xs[steps+1,indices],ge),
                          alpha_time[:,None])
    def terms(features):
        return intercept_terms(features,intercept['position'][selected_lanes],ref['toi_racket'][selected_lanes])
    def total(components):
        a,b,c,d=components.unbind(-1)
        return a+b+c+d
    with torch.no_grad():
        native_features={b:kinematics(v) for b,v in native.items()}
        native_loss={b:total(terms(v)) for b,v in native_features.items()}
        m,p=(kinematics(predicted(torch.tensor(v))) for v in (0.,1.))
        native_m,native_p=(native_features[b] for b in ('minus','plus'))
        finite=total(terms(p))-total(terms(m))
        def exchanged(midpoint, delta):
            return total(terms(midpoint+.5*delta))-total(terms(midpoint-.5*delta))
        at_native_center=exchanged(.5*(native_m+native_p),p-m)
        with_native_delta=exchanged(.5*(m+p),native_p-native_m)
        observed_terms=terms(native_p)-terms(native_m)
    # The four components share a command sequence but each lane has its own scalar displacement.
    alpha=torch.full((1,len(ids),1),.5,requires_grad=True)
    components=terms(kinematics(predicted(alpha)))
    component_jvp=torch.stack([torch.autograd.grad(components[:,c].sum(),alpha,retain_graph=c<3)[0][0,:,0]
                               for c in range(4)],-1)
    grad=total(component_jvp)
    rows=[]
    for k,(i,l,j) in enumerate(selections):
        dv=float(native_loss['plus'][k]-native_loss['minus'][k])
        noise=abs(float(native_loss['nominal_a'][k]-native_loss['nominal_b'][k]))
        delta=float(finite[k]); jac=float(grad[i])
        target_change=float((cmds['plus'][0][:j+1,i]-cmds['minus'][0][:j+1,i]).abs().max())
        rows.append(dict(lane=l,tick=j+1,measured=dv,nominal_repeat=noise,model_difference=delta,model_derivative=jac,
                         at_native_midpoint=float(at_native_center[k]),with_native_difference=float(with_native_delta[k]),
                         measured_terms=observed_terms[k].tolist(),derivative_terms=component_jvp[i].tolist(),
                         target_change_rad=target_change,supported=target_change>0 and abs(dv)>max(1.e-7,3*noise)))
    supported=[r for r in rows if r['supported']]
    counts={}
    for key in ('model_difference','model_derivative','at_native_midpoint','with_native_difference'):
        counts[key]=dict(same_direction=sum(r[key]*r['measured']>0 for r in supported),
                         gain_median=statistics.median(abs(r[key]/r['measured']) for r in supported) if supported else None)
    return dict(lanes=len(ids),matched_and_inside=len(rows),above_noise=len(supported),scores=counts,cases=rows,
                probe_actor_sha256=tensor_digest(base['actor_state'].items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('decision','development','shadow','bank','output'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--training',type=Path,help='Registered training-only response panel for component calibration')
    p.add_argument('--arm-drive-fit',type=Path,help='Check a source-matched offline trajectory correction along the same native command sequences')
    args=p.parse_args();torch.set_num_threads(2)
    read=lambda f:json.loads(f.read_text())
    decision,shadow=read(args.decision),read(args.shadow)
    ckpath=Path(decision['checkpoint']);ck=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False)
    if (not shadow['passed'] or not decision['passed'] or digest(ckpath)!=decision['checkpoint_sha256']
            or shadow['actor_sha256']!=decision['actor_sha256'] or shadow['world_sha256']!=decision['world_sha256']
            or digest(args.bank)!=shadow['reference_bank_sha256']):
        raise ValueError('Shadow and forward evidence differ')
    meta=ck['metadata']['physical_imagination'];paths=meta['checkpoint_paths']
    argv=read(ckpath.parent/'startup.json')['argv'];opt=lambda key:Path(argv[argv.index('--'+key)+1])
    _,adapter,template,launch,manifest=load_physical_loop(archive=opt('archive'),safety_mapping=opt('safety-mapping'),
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device='cpu')
    ws=torch.load(Path(decision['candidate'])/'online_world.pt',weights_only=True,map_location='cpu',mmap=True)
    if tensor_digest(ws.items())!=decision['world_sha256'] or meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256']:
        raise ValueError('Different WM or execution')
    adapter.load_state_dict(ws); adapter.eval().requires_grad_(False)
    bank=TaskSpaceReferenceBankV1.from_payload(torch.load(args.bank,weights_only=True,map_location='cpu'))
    src=read(args.development/'completed.json')
    if args.arm_drive_fit:
        from supermimic_arm_drive_audit import arm_drive_hook
        calibration=read(args.arm_drive_fit)
        if (calibration['identity']['world']!=decision['world_sha256'] or calibration['final_holdout_opened']
                or calibration['identity']['development']!=src['wide']):
            raise ValueError('Different arm calibration source or development panel')
        coef=torch.tensor(calibration['trajectory_fit']['coefficients'])
        adapter.physical_model.register_forward_hook(arm_drive_hook(coef))
    result={s:score(args.development/s,src[s],adapter,template,launch,bank,manifest['task_contract'],
            ck['metadata']['policy_execution'],set(ck['config']['wm_holdout_lanes'])) for s in ('wide','half','quarter')}
    if args.training:
        status=digest(args.training/'status.json')
        reg=read(args.training.parent/'preregistration.json')
        startup=read(args.training/'startup.json')
        if (startup['source_sha256']!=reg['actor_checkpoint_sha256']
                or startup['specs'][0]['sha256']!=reg['spec_sha256']
                or set(reg['held_out_lanes'])!=set(ck['config']['wm_holdout_lanes'])):
            raise ValueError('Signed training case registration changed')
        result['training']=score(args.training,status,adapter,template,launch,bank,manifest['task_contract'],
            ck['metadata']['policy_execution'],set(ck['config']['wm_holdout_lanes']),training=True)
        result['training_status_sha256']=status
    result.update(shadow_sha256=digest(args.shadow),decision_sha256=digest(args.decision),actor_updates=0,
          scope='Fixed causal reference at the common prefix. Measured and modeled racket objectives along recorded command histories. Three scales share launch cases. This is not an Actor parameter gradient or a contact receipt.')
    if args.arm_drive_fit:
        result['arm_drive_fit_sha256']=digest(args.arm_drive_fit)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
