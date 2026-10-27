"""Replay bounded strike proposals on two actual native source input sets."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
sys.path.insert(0,str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    factory=load('public_momentum_strike',HERE/'strike_controller.py')
    source_factory_path=HERE.parent/'windowed_swing_20261027/window_controller.py'
    source_factory=load('momentum_preserved_source_dispatch',source_factory_path)
    run=HERE.parent/'early_lateral7_n128_01'
    inputs_path=run/'INPUTS.json'
    inputs=json.loads(inputs_path.read_text())
    defaults,limits=[torch.tensor(inputs[k],dtype=torch.float32) for k in
                     ['default_joint_position_rad','joint_limits_rad']]
    sources=[Path(__file__),HERE/'strike_controller.py',HERE/'free_floating.py',HERE/'ACTUAL_CHECK.json',
        source_factory_path,inputs_path,ROOT/'quadsmash/control/full_racket_residual_v4.py',
        ROOT/'quadsmash/control/full_racket_joint_preparation_v4.py',ROOT/'quadsmash/tasks/stand/coupled_env.py']
    results=[]
    for suffix in ['000','001']:
        policy_path,data_path=run/f'candidate_{suffix}.pt',run/f'public_and_actions_{suffix}.npz'
        source=torch.load(policy_path,map_location='cpu',weights_only=False)
        before=copy.deepcopy(source)
        with np.load(data_path) as data:
            public,native,valid=[data[k].copy() for k in ['observations','actions','valid_first_episode']]
            assert data['training_indices'].tolist()==[0,29,69,100,112,123]
        assert public.shape==(800,6,794)
        baseline=source_factory.make_source(source,6,'cpu')
        expected=[]
        old_time=None
        for obs in public:
            if old_time is not None:
                ids=np.flatnonzero(obs[:,775]<old_time-1e-6)
                if len(ids):
                    baseline.reset(torch.tensor(ids))
            expected.append(baseline.action({'policy':torch.from_numpy(obs)},defaults,limits).clone())
            old_time=obs[:,775].copy()
        for candidate in factory.candidates():
            kwargs={k:candidate[k] for k in ['gain','leg_velocity_delta_cap_rad_s']}
            controller=factory.MomentumStrikeController(source_factory.make_source(source,6,'cpu'),
                source['asset_sha256'],6,'cpu',**kwargs)
            old_time=previous=None
            max_native=max_delta=max_slew=max_arm_delta=max_leg_delta=max_request=max_velocity=0.
            max_h=max_p=0.
            active_rows=solved_rows=rejected_rows=infeasible_rows=0
            objective_before=[]
            objective_after=[]
            for obs,actual_native,keep,base in zip(public,native,valid,expected):
                reset=np.zeros(6,dtype=bool)
                if old_time is not None:
                    reset=obs[:,775]<old_time-1e-6
                    ids=np.flatnonzero(reset)
                    if len(ids):
                        controller.reset(torch.tensor(ids))
                values=torch.from_numpy(obs)
                untouched=values.clone()
                action=controller.action({'policy':values},defaults,limits)
                assert torch.equal(values,untouched)
                assert action.shape==(6,18) and torch.isfinite(action).all() and (abs(action)<=1.).all()
                target=limits[:,0]+.5*(action+1.)*(limits[:,1]-limits[:,0])
                base_target=limits[:,0]+.5*(base+1.)*(limits[:,1]-limits[:,0])
                difference=abs(target-base_target)
                max_arm_delta=max(max_arm_delta,float(difference[:,12:].max()))
                max_leg_delta=max(max_leg_delta,float(difference[:,:12].max()))
                max_delta=max(max_delta,float(abs(action-base).max()))
                if keep.any():
                    max_native=max(max_native,float(abs(action.numpy()[keep]-actual_native[keep]).max()))
                if previous is not None and (~reset).any():
                    max_slew=max(max_slew,float(abs(target[~reset,12:]-previous[~reset,12:]).max())/.01)
                if candidate['gain']:
                    diagnostic=controller.last_diagnostic
                    active=diagnostic['active']
                    assert not np.any(active & ((obs[:,772]<=.5) | (obs[:,67]<=.5) | (obs[:,57:61]>.5).any(1)))
                    height=1.80-np.sum((-obs[:,6:9])*obs[:,760:763],axis=1)
                    assert not np.any(active & ((height<.30) | (-obs[:,8]<=.25)))
                    assert diagnostic['original_complete_tool_guard_applied']
                    max_request=max(max_request,float(abs(diagnostic['requests']).max()))
                    active_rows+=int(active.sum())
                    rejected_rows+=int(diagnostic['guard_rejected'].sum())
                    assert torch.equal(action[~torch.tensor(active),:12],base[~torch.tensor(active),:12])
                    if candidate['leg_velocity_delta_cap_rad_s']==0.:
                        assert torch.equal(action[:,:12],base[:,:12])
                    for prediction in diagnostic['predictions']:
                        if not prediction['solved']:
                            infeasible_rows+=1
                            continue
                        solved_rows+=1
                        objective_before.append(prediction['before_weighted_objective'])
                        objective_after.append(prediction['after_weighted_objective'])
                        assert prediction['after_weighted_objective']<=prediction['before_weighted_objective']+1e-8
                        lane=prediction['lane']
                        dq=np.asarray(prediction['requested_joint_velocity_delta_rad_s'])
                        total=obs[lane,27:45]+dq
                        assert (abs(total[:12])<=30.+1e-7).all() and (abs(total[12:])<=5.+1e-7).all()
                        assert (abs(dq[:12])<=candidate['leg_velocity_delta_cap_rad_s']+1e-7).all()
                        max_velocity=max(max_velocity,float(abs(total).max()))
                        model=controller.requester.model
                        q=defaults.numpy()[None]+obs[lane:lane+1,9:27]
                        old=model.forward(q,obs[lane:lane+1,27:45],obs[lane:lane+1,:3],obs[lane:lane+1,3:6])
                        new=model.forward(q,total[None],np.array(prediction['predicted_root_linear_body_m_s'])[None],
                            np.array(prediction['predicted_root_angular_body_rad_s'])[None])
                        max_h=max(max_h,float(abs(new['angular_momentum']-old['angular_momentum']).max()))
                        max_p=max(max_p,float(abs(new['center_velocity']-old['center_velocity']).max()*model.mass.sum()))
                previous=target.clone()
                old_time=obs[:,775].copy()
            assert max_slew<=5.0005 and max_request<=1.+1e-7
            assert max_leg_delta<=.15+1e-5 and max_h<1e-6 and max_p<1e-6
            assert same_tree(source,before)
            if candidate['gain']==0.:
                assert max_delta==0.
            else:
                assert active_rows>0 and solved_rows>0 and max_delta>0.
            # Selective reset after the active stroke begins must preserve
            # other lanes' plan validity, slew state, and source histories.
            twin=factory.MomentumStrikeController(source_factory.make_source(source,6,'cpu'),
                source['asset_sha256'],6,'cpu',**kwargs)
            controller.reset()
            for obs in public[:281]:
                for c in [controller,twin]:
                    c.action({'policy':torch.from_numpy(obs)},defaults,limits)
            controller.reset(torch.tensor([0]))
            changed=public[281].copy()
            changed[0]=public[0,0]
            a=controller.action({'policy':torch.from_numpy(changed)},defaults,limits)
            b=twin.action({'policy':torch.from_numpy(public[281])},defaults,limits)
            assert torch.equal(a[1:],b[1:])
            assert same_tree(source,before)
            result=dict(source_candidate=int(suffix),candidate=candidate,actual_public_rows=4800,
                valid_first_episode_rows=int(valid.sum()),active_public_pre_hit_air_rows=active_rows,
                model_solved_rows=solved_rows,guard_rejected_rows=rejected_rows,infeasible_model_rows=infeasible_rows,
                maximum_normalized_action_change=max_delta,maximum_original_GPU_action_difference=max_native,
                maximum_arm_target_rate_rad_s=max_slew,maximum_leg_target_delta_rad=max_leg_delta,
                maximum_arm_target_delta_rad=max_arm_delta,maximum_residual_request=max_request,
                maximum_hypothetical_total_joint_velocity_rad_s=max_velocity,
                maximum_model_H_conservation_error=max_h,maximum_model_P_conservation_error=max_p,
                mean_local_objective_before=(float(np.mean(objective_before)) if objective_before else None),
                mean_local_objective_after=(float(np.mean(objective_after)) if objective_after else None),
                selective_reset_preserves_other_lanes=True,source_checkpoint_and_public_inputs_unchanged=True,
                original_complete_tool_guard_and_arm_slew_retained=True,
                baseline_original_tolerance_pass=(max_native<=5e-4 if candidate['gain']==0. else None))
            results.append(result)
            print(json.dumps(result),flush=True)
        sources.extend([policy_path,data_path])
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,
        both_baseline_replays_within_original_tolerance=all(r['baseline_original_tolerance_pass']
            for r in results if r['candidate']['gain']==0.),
        source_sha256={str(p):sha(p) for p in sources},
        actual_native_performance_claimed=False,native_execution=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    assert result['both_baseline_replays_within_original_tolerance']


if __name__=='__main__':
    main()
