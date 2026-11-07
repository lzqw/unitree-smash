#!/usr/bin/env python3
"""Diagnose recorded landing horizons; no native control or relaxed execution.

Replay live reference memory from recorded observations/commands. At selected
ticks inspect rejected QP constraints with a separate feasibility LP. Dropping
constraints is diagnostic only. A free-arm QP counterfactual retains physical
and command limits but permits changing the prescribed arm target.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import asdict

import numpy as np
import osqp
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.command_history_support_qp_v1 import CommandHistorySupportQP
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
from scripts.implicit_support_prediction_v1 import predict_implicit_support


def feasibility(A,low,high,drop=()):
    keep=np.ones(len(low),dtype=bool);keep[list(drop)]=False
    upper=keep & np.isfinite(high);lower=keep & np.isfinite(low)
    result=linprog(np.zeros(A.shape[1]),A_ub=np.r_[A[upper],-A[lower]],
        b_ub=np.r_[high[upper],-low[lower]],bounds=[(None,None)]*A.shape[1],method='highs')
    return dict(status=int(result.status),feasible=bool(result.success),message=result.message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--implicit-replay-substeps',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous diagnostic')
    read=lambda name:json.loads((args.run/name).read_text())
    job=read('job.json');inputs=read('INPUTS.json');native=read('native_command_inputs.json')
    if job['status']!='SUCCEEDED' or inputs['training_indices']!=list(range(128)):
        raise ValueError('Require completed original TRAIN128')
    if '--landing-mpc-command-history-interface' not in job['command']:
        raise ValueError('Require command-history trial')
    limits=np.asarray(native['joint_limits_rad']);default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings=LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,
        contact_mode_fallback=True)
    mpc=LandingCommandHistoryMPCV1(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    implicit_checks=[]
    if args.implicit_replay_substeps:
        if args.implicit_replay_substeps<1:raise ValueError('Positive substep count required')
        def implicit_integrate(state,acceleration,plan=None):
            if plan is None:raise ValueError('Actual held command required for implicit diagnostic')
            target=mpc.admit_command_target(state,plan['joint_position_target'])
            result=predict_implicit_support(mpc.qp.model,interface,state,target,
                substeps=args.implicit_replay_substeps)
            implicit_checks.append(result['diagnostics'])
            following=result['state'];following.previous_command_target=target.copy()
            return following
        # This affects only CPU counterfactuals, never a live native controller.
        mpc._integrate=implicit_integrate
    free=CommandHistorySupportQP(interface,cfg,preserve_arm_command=False,arm_target_rate_rad_s=10.)
    records=read('landing_mpc_records.json');public={};hashes={}
    for case in sorted({r['case'] for r in records}):
        path=args.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:public[case]=unpack_public(z['packed_public'])
    with np.load(args.run/'all_first_episodes.npz') as z:
        poses=z['link_poses'];clocks=z['episode_clock_s'];valid=z['valid_first_episode']
    body_names=read('geometry.json')['body_names'];root_id=body_names.index(mpc.qp.model.record['root_body'])
    OriginalOSQP=osqp.OSQP
    captures=[]
    class CapturingOSQP:
        def __init__(self):self.solver=OriginalOSQP()
        def setup(self,**kw):
            self.data={k:kw[k].copy() for k in ('A','l','u')};return self.solver.setup(**kw)
        def solve(self,*a,**kw):
            result=self.solver.solve(*a,**kw)
            captures.append(dict(self.data,status=result.info.status));return result
        def __getattr__(self,name):return getattr(self.solver,name)
    original_solve=mpc.qp.solve;attempts=[]
    def capture_solve(state,target,**kw):
        memory=mpc.qp.reference_state();start=len(captures)
        plan=original_solve(state,target,**kw)
        attempts.append(dict(state=copy.deepcopy(state),target=np.asarray(target).copy(),
            memory=memory,plan=plan,matrix=(captures[-1] if len(captures)>start else None)))
        return plan
    mpc.qp.solve=capture_solve
    selected={1:{306,350,400},67:{300,400,600,760},95:{300,400,600,760}}
    output=[];seen=set();max_pose_error=0.
    try:
        osqp.OSQP=CapturingOSQP
        for r in records:
            case=r['case'];tick=r.get('control_tick')
            if r.get('episode_index')!=0 or tick is None:raise ValueError('Require first-episode tick provenance')
            obs=public[case][tick]
            if abs(float(obs[775])-.01*tick)>2e-6:raise ValueError('Trace/control clock mismatch')
            state=mpc.qp.model.state_from_public(obs,default)
            if tick%2==0:
                idx=tick//2
                if not valid[idx,case] or abs(float(clocks[idx,case])-.01*tick)>2e-6:
                    raise ValueError('No matching native root pose')
                error=float(np.max(abs(state.root_position_world-poses[idx,case,root_id,:3])))
                max_pose_error=max(max_pose_error,error)
                if error>2e-5:raise ValueError('Public reconstruction differs from native root pose')
            state.contacts=np.asarray(r['support_contacts'],dtype=bool)
            state.measured_force_world=np.asarray(r['measured_foot_force_world'])
            state.previous_command_target=np.asarray(r['previous_dispatched_target'])
            target=mpc.admit_command_target(state,np.asarray(r['requested_student_joint_reference']))
            mpc.observe(state,target,lane=case,control_tick=tick)
            if tick not in selected.get(case,set()):continue
            seen.add((case,tick));captures.clear();attempts.clear();implicit_checks.clear()
            plan=mpc.solve(state,target,lane=case,control_tick=tick,force=True)
            failure=next((s for s in plan.get('horizon',[]) if not s['accepted']),None)
            row=dict(case=case,tick=tick,recorded_failed_stage=r['failed_stage'],
                replay_failed_stage=None if failure is None else failure['stage'],
                replay_full_horizon_accepted=plan['accepted'],attempts=len(attempts))
            if implicit_checks:
                keys=['dynamics_contact_residual_max','joint_velocity_violation_rad_s',
                      'joint_position_violation_rad','normal_force_violation_n','friction_violation_n',
                      'motor_torque_violation_nm']
                row['implicit_prediction_violation_maxima']={k:max(x[k] for x in implicit_checks) for k in keys}
                row['all_implicit_prediction_checks_passed']=max(row['implicit_prediction_violation_maxima'].values())<=1e-5
            first=plan.get('horizon',[{}])[0]
            if first.get('accepted') and r.get('applied_first_step'):
                row['replayed_first_target_error_rad']=float(np.max(abs(
                    np.asarray(first['joint_position_target'])-np.asarray(r['admitted_joint_position_target_rad']))))
            if attempts and not attempts[-1]['plan']['accepted']:
                last=attempts[-1];s=last['state'];matrix=last['matrix'];n=int(s.contacts.sum())
                row.update(failed_contacts=s.contacts.tolist(),failed_joint_velocity=s.joint_velocity.tolist(),
                           failed_status=last['plan'].get('solver_status'))
                free.restore_reference_state(last['memory']);counter=free.solve(s,last['target'])
                row['same_state_free_arm']=dict(accepted=counter['accepted'],status=counter.get('solver_status'),
                    residuals=counter.get('residuals'))
                if matrix is not None:
                    A=matrix['A'].toarray();lo=matrix['l'];hi=matrix['u'];start=6+12*n
                    if len(lo)!=start+36:raise ValueError('Unexpected constraint grouping; do not infer labels')
                    groups=dict(floating_dynamics=range(6),contact_acceleration=range(6,6+3*n),
                        contact_friction=range(6+3*n,start),leg_torque=range(start,start+12),
                        arm_torque=range(start+12,start+18),leg_joint_bounds=range(start+18,start+30),
                        arm_joint_bounds=range(start+30,start+36))
                    row['linear_feasibility']=feasibility(A,lo,hi)
                    row['diagnostic_drop_one_group']={k:feasibility(A,lo,hi,v) for k,v in groups.items()}
            output.append(row)
    finally:
        osqp.OSQP=OriginalOSQP;mpc.qp.solve=original_solve
    expected={(c,t) for c,ts in selected.items() for t in ts}
    if seen!=expected:raise ValueError('Missing predeclared sample ticks: '+str(expected-seen))
    for f in ('landing_mpc_records.json','native_command_inputs.json','all_first_episodes.npz','job.json'):
        hashes[str(args.run/f)]=hashlib.sha256((args.run/f).read_bytes()).hexdigest()
    for f in ('implicit_support_prediction_v1.py','landing_command_history_mpc_v1.py',
              'command_history_support_qp_v1.py','landing_support_mpc_v1.py','landing_support_mpc_v2.py',
              'landing_support_mpc_v3.py','reduced_whole_body_support_qp_v1.py'):
        path=ROOT/'scripts'/f;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(scope=__doc__,source=str(args.run),native_new_execution=False,
        implicit_replay_substeps=args.implicit_replay_substeps,
        qp_settings=asdict(cfg),mpc_settings=asdict(settings),
        public_vs_native_root_position_error_max_m=max_pose_error,records=output,source_sha256=hashes)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(case=r['case'],tick=r['tick'],failed_stage=r['replay_failed_stage'],
        first_target_error=r.get('replayed_first_target_error_rad'),
        free_arm=r.get('same_state_free_arm',{}).get('accepted'),
        full_horizon=r['replay_full_horizon_accepted'],implicit_checks=r.get('all_implicit_prediction_checks_passed'),
        feasible_when_removed=[k for k,v in r.get('diagnostic_drop_one_group',{}).items() if v['feasible']]) for r in output]))


if __name__=='__main__':main()
