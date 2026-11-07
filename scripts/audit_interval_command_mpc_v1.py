#!/usr/bin/env python3
"""Replay recorded landing states through the matched implicit-command model.

CPU counterfactual only. Sampling is every twentieth recorded tick. All live reference ticks are replayed;
failed plans never update actual controller memory. No new native outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['osqp','active-set'],default='osqp')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Do not overwrite earlier evidence')
    read=lambda f:json.loads((args.run/f).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed native input required')
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings=LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,
        contact_mode_fallback=True)
    controller=LandingIntervalCommandMPCV1
    if args.backend=='active-set':
        from scripts.landing_active_set_interval_mpc_v1 import LandingActiveSetIntervalMPCV1
        controller=LandingActiveSetIntervalMPCV1
    mpc=controller(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    logs=read('landing_mpc_records.json');public={};hashes={};output=[]
    for case in sorted({r['case'] for r in logs}):
        path=args.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:public[case]=unpack_public(z['packed_public'])
    for row in logs:
        case,tick=row['case'],row['control_tick']
        if row.get('episode_index')!=0:raise ValueError('First episode only')
        obs=public[case][tick]
        if abs(float(obs[775])-.01*tick)>2e-6:raise ValueError('Clock mismatch')
        state=mpc.qp.model.state_from_public(obs,default)
        state.contacts=np.asarray(row['support_contacts'],dtype=bool)
        state.measured_force_world=np.asarray(row['measured_foot_force_world'])
        state.previous_command_target=np.asarray(row['previous_dispatched_target'])
        reference=mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
        mpc.observe(state,reference,lane=case,control_tick=tick)
        if tick%20:continue
        started=time.perf_counter()
        plan=mpc.solve(state,reference,lane=case,control_tick=tick,force=True)
        stages=plan.get('horizon',[])
        first=stages[0] if stages else {}
        failed=next((s for s in stages if not s['accepted']),None)
        checks=plan.get('implicit_prediction_checks',[])
        record=dict(case=case,tick=tick,
            native_recorded_failed_stage=row['failed_stage'],
            native_recorded_first_step_applied=row['applied_first_step'],
            candidate_first_step_accepted=bool(first.get('accepted')),
            candidate_horizon_accepted=bool(plan['accepted']),
            candidate_failed_stage=None if failed is None else failed['stage'],
            prediction_steps=len(checks),rejection=plan.get('rejection'),
            failed_qp_status=None if failed is None else failed.get('solver_status'),
            failed_qp_rejection=None if failed is None else failed.get('qp_rejection'),
            wall_seconds=time.perf_counter()-started,
            prediction_diagnostic_maxima={k:max(c[k] for c in checks) for k in checks[0]} if checks else {},
            prediction_stage_count=len(stages),
            stage_contact_transitions=[dict(stage=i,contacts=s['contacts'])
                for i,s in enumerate(stages) if i==0 or s['contacts']!=stages[i-1]['contacts']],
            terminal=plan.get('terminal_risk'),terminal_support_margin_ok=plan.get('support_margin_terminal_ok'))
        if first.get('accepted'):
            target=np.asarray(first['joint_position_target'])
            record['first_target_change_from_actual_rad']=float(np.max(abs(target-np.asarray(row['final_dispatched_joint_target_rad']))))
            record['first_residuals']=first['residuals']
        output.append(record)
        if len(output)%20==0:print(json.dumps(dict(processed=len(output),case=case,tick=tick)),flush=True)
    files=[args.run/n for n in ['landing_mpc_records.json','native_command_inputs.json','job.json','INPUTS.json']]
    files += [ROOT/'scripts'/n for n in ['audit_interval_command_mpc_v1.py',
        'interval_command_support_qp_v1.py','condensed_implicit_drive_v1.py','landing_interval_command_mpc_v1.py',
        'implicit_support_prediction_v1.py','continuous_support_qp_v1.py',
        'landing_support_mpc_v1.py','landing_support_mpc_v2.py','landing_support_mpc_v3.py',
        'landing_command_history_mpc_v1.py','command_history_support_qp_v1.py']]
    if args.backend=='active-set':
        files += [ROOT/'scripts'/n for n in ['landing_active_set_interval_mpc_v1.py','dense_active_set_qp_v1.py']]
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    summary=dict(samples=len(output),first_accepted=sum(r['candidate_first_step_accepted'] for r in output),
        horizon_accepted=sum(r['candidate_horizon_accepted'] for r in output),
        prediction_check_rejections=sum(not r['candidate_first_step_accepted'] and r['prediction_steps']>0 for r in output),
        wall_seconds=sum(r['wall_seconds'] for r in output))
    result=dict(scope=__doc__,native_new_execution=False,source=str(args.run),contract=mpc.contract(),
                summary=summary,records=output,source_sha256=hashes)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
