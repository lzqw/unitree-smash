#!/usr/bin/env python3
"""Compare complete serial and process-pool plans on recorded native states."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.parallel_landing_solver_v1 import ParallelLandingSolverV1, build_controller, execute
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2

TIMING = {'wall_runtime_s','solver_runtime_s','runtime_ms','candidate_runtime_s',
          'baseline_runtime_s','total_runtime_s'}


def difference(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((difference(a[k],b[k]) for k in a if k not in TIMING),default=0.)
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        return max((difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,np.ndarray):
        np.testing.assert_allclose(a,b,rtol=0.,atol=1e-10)
        return float(np.max(abs(a.astype(float)-np.asarray(b,dtype=float)),initial=0.))
    if isinstance(a,(float,np.floating)):
        if not np.isfinite(a):
            assert a==b
            return 0.
        assert abs(float(a)-float(b))<=1e-10,(a,b)
        return abs(float(a)-float(b))
    assert a==b,(a,b)
    return 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--family',choices=['full_interval','feasible_command_cost'],default='full_interval')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    read=lambda n:json.loads((a.run/n).read_text())
    assert read('job.json')['status']=='SUCCEEDED'
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    cfg=dict(family=a.family,command_change_weight=1.,arm_target_rate_rad_s=10.,posture_target=default,
        interface=dict(limits=np.asarray(native['joint_limits_rad']),stiffness=np.r_[np.full(12,25.),np.full(6,80.)],
            damping=np.r_[np.full(12,.5),np.full(6,4.)],effort=np.r_[np.full(12,23.5),np.full(6,100.)],
            velocity=np.r_[np.full(12,30.),np.full(6,5.)],friction=.65),
        qp_settings=asdict(SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
            swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
            nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,
            nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)),
        settings=asdict(LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
            one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
            executor_replay=True,predict_touchdown_contacts=True,contact_mode_fallback=True)))
    model=LoadedPinocchioModelV1();public={};sources=[Path(__file__)];seen=set();batches={}
    for row in read('landing_mpc_records.json'):
        assert row['episode_index']==0
        case,tick=row['case'],row['control_tick']
        if case not in public:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(path)
            with np.load(path,allow_pickle=False) as z:public[case]=unpack_public(z['packed_public'])
        obs=public[case][tick];assert abs(float(obs[775])-.01*tick)<2e-6
        state=model.state_from_public(obs,default)
        state.contacts=np.asarray(row['support_contacts'],dtype=bool)
        state.measured_force_world=np.asarray(row['measured_foot_force_world'])
        state.previous_command_target=np.asarray(row['previous_dispatched_target'])
        request=dict(operation='observe',lane=case,control_tick=tick,state=state,
            reference=np.asarray(row['requested_student_joint_reference']),
            solve=tick%100==0 or case not in seen,force=True)
        seen.add(case);batches.setdefault(tick,[]).append(request)
    serial=build_controller(cfg);expected={};started=time.perf_counter()
    for tick,requests in sorted(batches.items()):
        expected[tick]=[execute(serial,copy.deepcopy(r)) for r in requests]
    serial_s=time.perf_counter()-started
    startup=time.perf_counter();pool=ParallelLandingSolverV1(cfg,workers=a.workers)
    startup_s=time.perf_counter()-startup;started=time.perf_counter();max_error=0.;plans=0;accepted=0
    try:
        for tick,requests in sorted(batches.items()):
            actual=pool.batch(requests)
            max_error=max(max_error,difference(expected[tick],actual))
            plans+=sum(r['solve'] for r in requests)
            accepted+=sum(bool(v['accepted']) for v in actual if v is not None)
        parallel_s=time.perf_counter()-started
        # Reset one lane and restart its clock. Another lane retains its final
        # reference memory; complete plans must still match the serial owner.
        lanes=sorted(seen);reset_case=lanes[0];other=lanes[1]
        reset=dict(operation='reset',lane=reset_case)
        execute(serial,reset);assert pool.batch([reset])==[None]
        original=next(r for requests in batches.values() for r in requests if r['lane']==reset_case)
        retained=next(r for requests in reversed(list(batches.values())) for r in requests if r['lane']==other)
        checks=[dict(original,solve=True),dict(retained,solve=True)]
        expected_reset=[execute(serial,copy.deepcopy(r)) for r in checks]
        reset_error=difference(expected_reset,pool.batch(checks))
        failed_closed=False
        try:pool.batch([dict(operation='unsupported',lane=reset_case)])
        except RuntimeError:failed_closed=pool.closed
        assert failed_closed
    finally:pool.close()
    assert all(p.poll() is not None for p in pool.processes)
    sources += [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    sources += [Path(m.__file__) for name,m in list(sys.modules.items())
                if name.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,passed=True,native_new_execution=False,native_controller_modified=False,
        family=a.family,workers=a.workers,lanes=len(seen),batches=len(batches),
        observed_ticks=sum(len(v) for v in batches.values()),solved_states=plans,
        full_horizon_accepted=accepted,maximum_plan_numeric_difference=max_error,
        per_lane_reset_difference=reset_error,worker_error_closes_pool=failed_closed,
        worker_processes_reaped=True,cuda_disabled_in_workers=True,
        serial_replay_seconds=serial_s,parallel_replay_seconds=parallel_s,
        process_startup_seconds=startup_s,replay_speedup=serial_s/parallel_s,
        ignored_fields=sorted(TIMING),
        limitations=['One serial-then-parallel timing pair; load/cache effects remain.',
                     'No native simulation outcome or throughput claim.',
                     'Worker failure is explicit; no unaudited fallback command is synthesized.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
