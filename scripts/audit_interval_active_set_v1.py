#!/usr/bin/env python3
"""Compare active-set and OSQP on the identical first-stage physical QP.

On recorded states, compare the original QP and an active-set solve.
No native command is executed and no physical constraint is relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import osqp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    read=lambda name:json.loads((a.run/name).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed native input required')
    prior=json.loads(a.reference_audit.read_text())
    if Path(prior['source']).resolve()!=a.run.resolve():raise ValueError('Wrong reference audit')
    for path,expected in prior['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Changed audited source: '+path)
    selected={(r['case'],r['tick']):r for r in prior['records'] if r['candidate_failed_stage']==0}
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings=LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,contact_mode_fallback=True)
    mpc=LandingIntervalCommandMPCV1(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default);logs=read('landing_mpc_records.json');public={};paths=[];output=[]
    for case in sorted({r['case'] for r in logs}):
        path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';paths.append(path)
        with np.load(path) as z:public[case]=unpack_public(z['packed_public'])
    original_solver=osqp.OSQP;captured=[];factory=original_solver
    class CaptureSolver:
        def __init__(self):self.solver=factory()
        def setup(self,**kw):
            self.data={k:kw[k].copy() for k in ('P','q','A','l','u')}
            self.solver.setup(**kw)
        def solve(self):
            result=self.solver.solve();info=result.info
            self.data['info']={k:getattr(info,k,None) for k in
                ('status','iter','prim_res','dual_res','pri_res','dua_res','run_time')}
            self.data['independent_kkt']=getattr(self.solver,'last_diagnostics',None)
            captured.append(self.data);return result
    def compact(plan):
        return {k:plan.get(k) for k in ('accepted','solver_status','solver_iterations','rejection',
            'residuals','predicted_minimum_root_height_m','wall_runtime_s')}
    osqp.OSQP=CaptureSolver
    try:
        for row in logs:
            case,tick=row['case'],row['control_tick']
            if row['episode_index']!=0:raise ValueError('First episodes required')
            state=mpc.qp.model.state_from_public(public[case][tick],default)
            state.contacts=np.asarray(row['support_contacts'],dtype=bool)
            state.measured_force_world=np.asarray(row['measured_foot_force_world'])
            state.previous_command_target=np.asarray(row['previous_dispatched_target'])
            reference=mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
            mpc.observe(state,reference,lane=case,control_tick=tick)
            if (case,tick) not in selected:continue
            before=mpc._prediction_reference[case];effective=mpc._posture_reference(state,reference)
            mpc.qp.restore_reference_state(before);captured.clear();factory=original_solver
            mpc.qp.settings=cfg;baseline=mpc.qp.solve(state,effective);matrix=captured[-1]
            mpc.qp.restore_reference_state(before);factory=DenseActiveSetQP
            longer=mpc.qp.solve(state,effective);long_matrix=captured[-1]
            # The convergence comparison must retain exactly the same model,
            # objective, constraints and reference, not just the same case ID.
            for key in ('A','P'):
                if (matrix[key]!=long_matrix[key]).nnz:raise ValueError('Changed mathematical problem')
            for key in ('q','l','u'):np.testing.assert_array_equal(matrix[key],long_matrix[key])
            result=dict(case=case,tick=tick,old_native_applied=row['applied_first_step'],
                previous_audit_status=selected[(case,tick)]['failed_qp_status'],
                baseline=compact(baseline),baseline_solver=matrix['info'],
                active_set=compact(longer),active_set_solver=long_matrix['info'],
                active_set_kkt=long_matrix['independent_kkt'])
            output.append(result);print(json.dumps(result),flush=True)
            factory=original_solver;mpc.qp.settings=cfg
    finally:osqp.OSQP=original_solver
    if len(output)!=len(selected):raise ValueError('Missing selected rejected states')
    paths += [Path(__file__),ROOT/'scripts/dense_active_set_qp_v1.py',a.reference_audit]+[a.run/x for x in
        ('job.json','native_command_inputs.json','landing_mpc_records.json')]
    result=dict(scope=__doc__,native_new_execution=False,records=output,
        summary=dict(samples=len(output),baseline_accepted=sum(r['baseline']['accepted'] for r in output),
            active_set_accepted=sum(r['active_set']['accepted'] for r in output)),
        iteration_limit=cfg.solver_max_iterations,quadprog_version='0.1.13',
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
