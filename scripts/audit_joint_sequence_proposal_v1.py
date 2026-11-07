#!/usr/bin/env python3
"""Offline multi-command proposals at all recorded first risk alarms.

Uses real frozen-student states. No native action is sent; a solved frozen
linear QP and a passed refreshed prediction are reported separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_joint_sequence_proposal_v1 import propose_sequence, verify_sequence
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.loaded_pinocchio_model_v1 import rotation_wxyz


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def query(item):
    cfg, row, obs, previous, measured_pose, default, limit, constraint_start = item
    controller=build_controller(cfg);model=controller.qp.model
    state=model.state_from_public(obs,default)
    pose_error=float(np.max(abs(state.root_position_world-measured_pose[:3])))
    rotation_error=float(np.max(abs(state.root_rotation_world-rotation_wxyz(measured_pose[3:]))))
    if pose_error>2e-5 or rotation_error>2e-5: raise ValueError('Actual recorded pose differs')
    state.previous_command_target=previous
    reference=np.asarray(row['student_reference'])
    started=time.perf_counter()
    proposal=propose_sequence(controller,state,reference,time_limit_s=limit,
                              constraint_start=constraint_start)
    verification=None
    if proposal['linear_proposal_solved']:
        verification=verify_sequence(controller,state,proposal['commands'])
    return plain(dict(case=row['case'],query_tick=row['query_tick'],
        complete5=row['complete5'],ball5_except_recovery=row['ball5_except_recovery'],
        first_nonfoot_s=row['first_nonfoot_s'],support_feet=int(state.contacts.sum()),
        baseline_full_horizon=row['models']['full_interval']['full_horizon'],
        pose_max_error_m=pose_error,rotation_max_error=rotation_error,
        proposal=proposal,verification=verification,wall_seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alarm-plans',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,nargs='*')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--solver-time-limit',type=float,default=20.)
    p.add_argument('--constraint-start',choices=['empty','control_endpoints'],default='control_endpoints')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier result')
    if not 1<=a.workers<=4:raise ValueError('1..4 CPU workers')
    plans=json.loads(a.alarm_plans.read_text())
    for path,digest in plans['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed evidence '+path)
    risk=json.loads(Path(plans['risk_report']).read_text());r=Path(risk['baseline'])
    native=json.loads((r/'native_command_inputs.json').read_text())
    geometry=json.loads((r/'geometry.json').read_text());root_index=geometry['body_names'].index('base')
    default=np.asarray(native['default_joint_position_rad']);limits=np.asarray(native['joint_limits_rad'])
    cfg=dict(family='full_interval',arm_target_rate_rad_s=10.,posture_target=default,
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
    with np.load(r/'all_first_episodes.npz',allow_pickle=False) as z:
        poses=z['link_poses'].copy();valid=z['valid_first_episode'].copy()
    selected=[x for x in plans['records'] if 'skipped' not in x and
              (a.cases is None or x['case'] in a.cases)]
    if a.cases is not None and sorted(x['case'] for x in selected)!=sorted(a.cases):
        raise ValueError('Some requested cases lack an actual first alarm')
    items=[];sources=[a.alarm_plans,Path(__file__),r/'all_first_episodes.npz',r/'native_command_inputs.json',r/'geometry.json',
                     ROOT/'configs/go2_piper_racket_landing_model_v1.json']
    for row in selected:
        case,tick=row['case'],row['query_tick'];path=r/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(path)
        with np.load(path,allow_pickle=False) as z:
            obs=unpack_public(z['packed_public'])[tick];action=z['dispatched_actions'][tick-1]
        if not valid[tick//2,case] or tick%2:raise ValueError('No matching measured pose')
        previous=limits[:,0]+.5*(action+1.)*(limits[:,1]-limits[:,0])
        items.append((cfg,row,obs,previous,poses[tick//2,case,root_index],default,a.solver_time_limit,a.constraint_start))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,items):
            rows.append(row)
            print(json.dumps(dict(case=row['case'],linear_solved=row['proposal']['linear_proposal_solved'],
                nonlinear_accepted=bool(row['verification'] and row['verification']['accepted']),
                qp_status=row['proposal'].get('solver_status'),seconds=row['wall_seconds'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    sources += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    summary=dict(queries=len(rows),linear_solved=sum(x['proposal']['linear_proposal_solved'] for x in rows),
        refreshed_prediction_accepted=sum(bool(x['verification'] and x['verification']['accepted']) for x in rows))
    result=dict(scope=__doc__,native_execution=False,new_native_successes=None,
        baseline=str(r),configuration=plain(cfg),summary=summary,records=rows,
        limitations=['Frozen proposal dynamics are only an initial guess; refreshed verification can reject.',
            'Fixed observed contact in the proposal; no liftoff or stepping sequence optimization.',
            'No native stability or self-collision certificate. A model pass needs full actual execution.',
            'This is offline search, not installed in the running controller.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
