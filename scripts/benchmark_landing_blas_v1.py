#!/usr/bin/env python3
"""Reproduce Isaac's late OpenBLAS override in an isolated CPU process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=[1,32],required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    # NumPy is normally imported before app construction. SciPy loads later,
    # after SimulationApp overrides the environment variable.
    import numpy as np
    os.environ['OPENBLAS_NUM_THREADS']=str(a.threads)
    from quadsmash.rl.episode_trace_v4 import unpack_public
    from scripts.landing_active_set_interval_mpc_v1 import LandingActiveSetIntervalMPCV1
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
    from scripts.landing_blas_runtime_v1 import configure_landing_blas_threads
    runtime=configure_landing_blas_threads(a.threads)
    read=lambda name:json.loads((a.run/name).read_text())
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
    mpc=LandingActiveSetIntervalMPCV1(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    records=read('landing_mpc_records.json')
    selected=[(3,360),(58,400),(115,700)]
    results=[];paths=[a.run/'native_command_inputs.json',a.run/'landing_mpc_records.json']
    for case,tick in selected:
        row=next(r for r in records if r['case']==case and r['control_tick']==tick)
        path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';paths.append(path)
        with np.load(path) as z:public=unpack_public(z['packed_public'])
        state=mpc.qp.model.state_from_public(public[tick],default)
        state.contacts=np.asarray(row['support_contacts'],dtype=bool)
        state.measured_force_world=np.asarray(row['measured_foot_force_world'])
        state.previous_command_target=np.asarray(row['previous_dispatched_target'])
        ref=mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
        mpc.observe(state,ref,lane=case,control_tick=tick)
        before=mpc._prediction_reference[case]
        times=[]
        for repeat in range(3):
            mpc.qp.restore_reference_state(before)
            start=time.perf_counter();plan=mpc.qp.solve(state,mpc._posture_reference(state,ref));times.append(time.perf_counter()-start)
        item=dict(case=case,tick=tick,seconds=times,accepted=bool(plan['accepted']),solver_status=plan['solver_status'])
        if plan['accepted']:
            item.update(target=plan['joint_position_target'].tolist(),residuals=plan['residuals'])
        results.append(item);print(json.dumps(item),flush=True)
    paths += [Path(__file__),ROOT/'scripts/landing_blas_runtime_v1.py',ROOT/'scripts/interval_command_support_qp_v1.py']
    result=dict(schema='landing-blas-benchmark-v1',scope=__doc__,threads=a.threads,runtime=runtime,
        records=results,native_execution=False,source_sha256={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
