#!/usr/bin/env python3
"""Matched CPU comparison of full-command interval MPC and command-change cost.

Replays all observed control-memory ticks, solves on every 100th tick and first
active tick per case. Recorded states are not replaced by predicted states.
No native recovery gain is established by this diagnostic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_minimum_command_mpc_v1 import LandingMinimumCommandMPCV1
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    read=lambda n:json.loads((a.run/n).read_text())
    assert read('job.json')['status']=='SUCCEEDED'
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,swing_foot_weight=1.,
        swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,
        nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings=LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        executor_replay=True,predict_touchdown_contacts=True,contact_mode_fallback=True)
    controllers={name:cls(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
        for name,cls in [('unchanged',LandingFullCommandIntervalMPCV1),
                         ('minimum_command',LandingMinimumCommandMPCV1)]}
    for c in controllers.values():c.set_posture_target(default)
    public={};sources=[Path(__file__)];seen=set();output=[]
    for row in read('landing_mpc_records.json'):
        if row.get('episode_index')!=0:continue
        case,tick=row['case'],row['control_tick']
        if case not in public:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(path)
            with np.load(path,allow_pickle=False) as z:public[case]=unpack_public(z['packed_public'])
        obs=public[case][tick]
        assert abs(float(obs[775])-.01*tick)<2e-6
        selected=tick%100==0 or case not in seen
        seen.add(case);record=dict(case=case,tick=tick,models={})
        for name,c in controllers.items():
            state=c.qp.model.state_from_public(obs,default)
            state.contacts=np.asarray(row['support_contacts'],dtype=bool)
            state.measured_force_world=np.asarray(row['measured_foot_force_world'])
            state.previous_command_target=np.asarray(row['previous_dispatched_target'])
            reference=c.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
            c.observe(state,reference,lane=case,control_tick=tick)
            if not selected:continue
            start=time.perf_counter();plan=c.solve(state,reference,lane=case,control_tick=tick,force=True)
            stages=plan.get('horizon',[]);first=stages[0] if stages else {}
            failed=next((s for s in stages if not s['accepted']),None)
            item=dict(first_accepted=bool(first.get('accepted')),horizon_accepted=bool(plan['accepted']),
                failed_stage=None if failed is None else failed['stage'],
                stage_count=len(stages),rejection=plan.get('rejection'),
                terminal=plan.get('terminal_risk'),runtime_ms=1000*(time.perf_counter()-start))
            if first.get('accepted'):
                target=np.asarray(first['joint_position_target']);delta=target-reference
                item.update(command_change_l2_rad=float(np.linalg.norm(delta)),
                    command_change_max_rad=float(np.max(abs(delta))),target=target.tolist(),
                    first_residuals=first['residuals'])
            record['models'][name]=item
        if selected:
            output.append(record)
            if len(output)%25==0:print(json.dumps(dict(samples=len(output))),flush=True)
    common=[r for r in output if all(x['first_accepted'] for x in r['models'].values())]
    summary={}
    for name in controllers:
        rows=[r['models'][name] for r in output]
        changes=[r['models'][name]['command_change_l2_rad'] for r in common]
        summary[name]=dict(first_accepted=sum(r['first_accepted'] for r in rows),
            horizon_accepted=sum(r['horizon_accepted'] for r in rows),
            runtime_s=sum(r['runtime_ms'] for r in rows)/1000,
            common_command_change_l2_mean_rad=float(np.mean(changes)) if changes else None,
            common_command_change_l2_p95_rad=float(np.quantile(changes,.95)) if changes else None)
    violations=[dict(case=r['case'],tick=r['tick'],old=r['models']['unchanged']['command_change_l2_rad'],
        new=r['models']['minimum_command']['command_change_l2_rad']) for r in common
        if r['models']['minimum_command']['command_change_l2_rad']>r['models']['unchanged']['command_change_l2_rad']+1e-6]
    # For identical feasible convex QPs, adding a distance penalty should not
    # increase that distance. Flag mismatches instead of excusing them as gains.
    sources += [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    sources += [ROOT/'scripts'/n for n in ['landing_minimum_command_mpc_v1.py',
        'landing_full_command_interval_mpc_v1.py','landing_joint_damper_mpc_v1.py',
        'landing_arm_feasibility_mpc_v1.py','landing_active_set_interval_mpc_v1.py',
        'landing_interval_command_mpc_v1.py','interval_command_support_qp_v1.py',
        'condensed_implicit_drive_v1.py','implicit_support_prediction_v1.py',
        'continuous_support_qp_v1.py','loaded_pinocchio_model_v1.py']]
    result=dict(scope=__doc__,source=str(a.run),native_new_execution=False,samples=len(output),
        common_first_step_samples=len(common),summary=summary,command_distance_increases=violations,
        contracts={n:c.contract() for n,c in controllers.items()},records=output,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256','contracts']}))


if __name__=='__main__':main()
