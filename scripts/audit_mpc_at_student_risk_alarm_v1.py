#!/usr/bin/env python3
"""Query full recovery plans at first held-out alarms on real student traces.

Planning-only counterfactual; no state is stepped in Isaac. Uses actual recorded
root poses and public proprioception at the same time. A first alarm between
50 Hz pose samples is delayed by one 10 ms tick, never assigned a future pose.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState,rotation_wxyz
from scripts.parallel_landing_solver_v1 import ParallelLandingSolverV1
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--risk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-memory',choices=['first_alarm','observed_support'],default='first_alarm')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing planning evidence')
    risk=json.loads(a.risk.read_text());r=Path(risk['baseline'])
    for path,digest in risk['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Risk evidence source changed: '+path)
    read=lambda n:json.loads((r/n).read_text())
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    limits=np.asarray(native['joint_limits_rad']);geometry=read('geometry.json')
    root_index=geometry['body_names'].index('base')
    model=LoadedPinocchioModelV1()
    with np.load(r/'all_first_episodes.npz',allow_pickle=False) as z:
        poses=z['link_poses'].copy();clocks=z['episode_clock_s'].copy();valid=z['valid_first_episode'].copy()
    cfg=dict(family='full_interval',command_change_weight=1.,arm_target_rate_rad_s=10.,posture_target=default,
        interface=dict(limits=limits,stiffness=np.r_[np.full(12,25.),np.full(6,80.)],
            damping=np.r_[np.full(12,.5),np.full(6,4.)],effort=np.r_[np.full(12,23.5),np.full(6,100.)],
            velocity=np.r_[np.full(12,30.),np.full(6,5.)],friction=.65),
        qp_settings=asdict(SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
            swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
            nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,
            nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)),
        settings=asdict(LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
            one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,executor_replay=True,
            predict_touchdown_contacts=True,contact_mode_fallback=True)))
    requests=[];records=[];files=[Path(__file__),a.risk]
    background={};background_pose_error=0.;background_rotation_error=0.
    coverage={v['case']:v for v in risk['coverage']}
    for alarm in risk['case_results']:
        if not alarm['any_alarm']:continue
        case=alarm['case'];first_tick=int(round(alarm['first_alarm_s']*100));tick=2*((first_tick+1)//2)
        path=r/'fixed_policy_traces'/f'case_{case:03d}.npz';files.append(path)
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);actions=z['dispatched_actions'].copy()
        event=coverage[case]['first_nonfoot_s']
        record=dict(case=case,first_alarm_tick=first_tick,query_tick=tick,
            complete5=alarm['complete5'],ball5_except_recovery=alarm['ball5_except_recovery'],
            first_nonfoot_s=event,models={})
        if tick>=len(public) or not valid[tick//2,case] or (event is not None and .01*tick>=event):
            record['skipped']='pose unavailable before first event';records.append(record);continue
        obs=public[tick];pose=poses[tick//2,case,root_index]
        if abs(float(clocks[tick//2,case])-.01*tick)>2e-6 or abs(float(obs[775])-.01*tick)>2e-6:
            raise ValueError('Native pose and public sample clocks differ')
        rotation=rotation_wxyz(pose[3:]);q=pin.neutral(model.model)
        q[:3]=pose[:3];q[3:7]=pin.Quaternion(rotation).coeffs();q[model.q_indices]=default+obs[9:27]
        v=np.zeros(model.model.nv);v[:6]=obs[:6];v[model.v_indices]=obs[27:45]
        state=PublicState(q,v,rotation,q[:3].copy(),q[model.q_indices].copy(),
                          v[model.v_indices].copy(),obs[57:61]>.5,0.)
        state.previous_command_target=limits[:,0]+.5*(actions[tick-1]+1.)*(limits[:,1]-limits[:,0])
        reference=limits[:,0]+.5*(actions[tick]+1.)*(limits[:,1]-limits[:,0])
        snapshot=model.snapshot(q,v)
        actual_feet=poses[tick//2,case,[geometry['body_names'].index(name) for name in model.feet],:3]
        predicted_feet=snapshot['body_positions_world'][[model.names.index(name) for name in model.feet]]
        fk_error=float(np.max(abs(predicted_feet-actual_feet)))
        gravity_error=float(np.max(abs(rotation.T@np.array([0.,0.,-1.])-obs[6:9])))
        if fk_error>2e-4 or gravity_error>1e-5:raise ValueError('Pose/kinematics mismatch')
        record.update(foot_fk_max_error_m=fk_error,gravity_max_error=gravity_error,
            support_feet=int(state.contacts.sum()),lead_s=None if event is None else event-.01*tick,
            student_reference=reference.tolist())
        records.append(record)
        requests.append(dict(operation='observe',lane=case,state=state,reference=reference,
                             control_tick=tick,solve=True,force=True))
        if a.reference_memory=='observed_support':
            hit=False;active=False
            for earlier in range(tick):
                old=public[earlier]
                active=active or (hit and old[774]>.5)
                hit=hit or old[773]>.5
                if not active:continue
                previous_state=model.state_from_public(old,default)
                if earlier%2==0:
                    measured=poses[earlier//2,case,root_index]
                    pe=float(np.max(abs(previous_state.root_position_world-measured[:3])))
                    re=float(np.max(abs(previous_state.root_rotation_world-rotation_wxyz(measured[3:]))))
                    background_pose_error=max(background_pose_error,pe)
                    background_rotation_error=max(background_rotation_error,re)
                    if pe>2e-5 or re>2e-5:raise ValueError('Background public-state reconstruction mismatch')
                previous_state.previous_command_target=limits[:,0]+.5*(actions[earlier-1]+1.)*(limits[:,1]-limits[:,0])
                previous_reference=limits[:,0]+.5*(actions[earlier]+1.)*(limits[:,1]-limits[:,0])
                background.setdefault(earlier,[]).append(dict(operation='observe',lane=case,
                    state=previous_state,reference=previous_reference,control_tick=earlier,solve=False,force=True))
    queried=[r for r in records if 'skipped' not in r]
    for family in ['full_interval','feasible_command_cost']:
        cfg['family']=family;started=time.perf_counter()
        with ParallelLandingSolverV1(cfg,workers=4,timeout_s=300.) as pool:
            for _,batch in sorted(background.items()):pool.batch(batch)
            plans=pool.batch(requests)
        for row,plan in zip(queried,plans):
            stages=plan.get('horizon',[]);failed=next((s for s in stages if not s['accepted']),None)
            first=bool(stages and stages[0]['accepted'])
            target=np.asarray(stages[0]['joint_position_target']) if first else None
            row['models'][family]=dict(full_horizon=bool(plan['accepted']),first_feasible=first,
                stage_count=len(stages),failed_stage=None if failed is None else failed.get('stage'),
                rejection=plan.get('rejection'),terminal_risk=plan.get('terminal_risk'),
                command_cost_selection=plan.get('command_cost_selection'),
                first_target=None if target is None else target.tolist(),
                command_change_l2_rad=None if target is None else float(np.linalg.norm(target-row['student_reference'])))
        print(json.dumps(dict(family=family,queries=len(plans),seconds=time.perf_counter()-started)),flush=True)
    summary={}
    for group,selected in [('all',queried),('old_success',[v for v in queried if v['complete5']]),
        ('ball_qualified_nonfoot_failure',[v for v in queried if v['ball5_except_recovery'] and
                                          not v['complete5'] and v['first_nonfoot_s'] is not None])]:
        summary[group]=dict(cases=len(selected),models={family:dict(
            first_feasible=sum(v['models'][family]['first_feasible'] for v in selected),
            full_horizon=sum(v['models'][family]['full_horizon'] for v in selected))
            for family in ['full_interval','feasible_command_cost']})
    files += [model.path]+[r/n for n in ['native_command_inputs.json','geometry.json','all_first_episodes.npz']]
    files += [Path(m.__file__) for name,m in sys.modules.items() if name.startswith('scripts.') and getattr(m,'__file__',None)]
    output=dict(scope=__doc__,native_execution=False,teacher_rescue_established=False,
        risk_report=str(a.risk.resolve()),reference_memory=a.reference_memory,
        background_reference_updates=sum(len(v) for v in background.values()),
        background_reference_only_no_teacher_actions=True,
        background_pose_max_error_m=background_pose_error,
        background_rotation_matrix_max_error=background_rotation_error,
        contact_mode='original observed force-norm hysteresis; no additional vertical force threshold',
        missing_continuous_force_archive=True,load_aware_support=False,
        summary=summary,records=records,
        limitations=['Queries on historical student states; no physical correction executed.',
            'Held-out fold alarms are retrospective validation, not an installed deployment model.',
            'At most10ms delay to the next actually recorded root pose; never backdate that pose.',
            'A feasible teacher plan can still be inaccurate or worse than the student.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
