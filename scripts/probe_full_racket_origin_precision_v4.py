#!/usr/bin/env python3
"""Measure native free flight and clone isolation at a declared origin spacing.

This engineering probe launches original TRAIN29 at reset in every replica,
before running0.5seconds. It is not a task episode, training score or evidence
of autonomous hitting. The complete robot, racket, net and native physics stay.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--env-spacing',type=float,choices=(0.,16.),required=True)
parser.add_argument('--seed',type=int,default=750601)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app

import numpy as np
from scipy.integrate import solve_ivp
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.tasks.stand.full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'INPUTS.json').exists()
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=json.loads(protocol_path.read_text());cases,indices=training_cases(ROOT,protocol)
    assert indices==list(range(128))
    assert sha(ROOT/protocol['asset']['path'])==protocol['asset']['sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protocol['physical_configs'].items())
    physics=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
    cfg=full_racket_cfg(128,protocol['physics_dt_s']);cfg.scene.env_spacing=args.env_spacing;cfg.seed=args.seed
    assert cfg.scene.filter_collisions and cfg.scene.replicate_physics
    assert abs(cfg.sim.dt*cfg.decimation-.01)<1e-12
    case=cases[29]
    # Check this diagnostic trajectory stays beyond the net before launch.
    start=np.r_[case['launch']['position_m'],case['launch']['velocity_mps']]
    wind=np.asarray(physics['wind_m_s']);drag=physics['quadratic_drag_k_per_m']
    def ode(_,state):
        relative=state[3:]-wind
        return np.r_[state[3:],np.array([0.,0.,-9.81])-drag*np.linalg.norm(relative)*relative]
    ideal=solve_ivp(ode,[0.,.5],start,rtol=1e-11,atol=1e-12,dense_output=True,max_step=.001,method='DOP853')
    assert ideal.success and ideal.y[0].min()>2. and ideal.y[2].min()>.1
    env=RallyTrainingEnvV4(cfg,cases=[case]*128,continue_after_miss=True)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        observation,_=wrapped.reset()
        default=env.robot.data.default_joint_pos[0,env.joint_ids]
        limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        action=(2*(default-limits[:,0])/(limits[:,1]-limits[:,0])-1).clamp(-1.,1.).repeat(128,1)
        assert not env.robot.is_fixed_base and torch.isfinite(action).all()
        ids=torch.arange(128,device=env.device)
        env.initialize_rally_launch(ids)
        env.sim.forward()
        paths=sorted(ROOT.glob('quadsmash/**/*.py'))+[Path(__file__)]
        atomic(args.output/'INPUTS.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(protocol_path),asset_sha256=protocol['asset']['sha256'],
            source_hashes={str(path.relative_to(ROOT)):sha(path) for path in paths},
            original_TRAIN_case_index=29,case=case,num_envs=128,seed=args.seed,
            original_court_spacing_m=16.,runtime_origin_spacing_m=args.env_spacing,
            actual_environment_origins_m=env.scene.env_origins.cpu().tolist(),
            direct_initial_launch=True,scope='0.5second free-flight/isolation engineering diagnostic; no task success measurement.',
            filter_collisions=cfg.scene.filter_collisions,replicate_physics=cfg.scene.replicate_physics,
            physics_dt_s=cfg.sim.dt,policy_dt_s=cfg.sim.dt*cfg.decimation,
            original_physical_configs_sha256=protocol['physical_configs'],formal_cases_read=False,formal_acceptance=False))
        positions=[(env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy().copy()]
        velocities=[env.shuttle.data.root_lin_vel_w.cpu().numpy().copy()]
        roots=[(env.robot.data.root_pos_w-env.scene.env_origins).cpu().numpy().copy()]
        forces=[];done_counts=[]
        for tick in range(50):
            observation,_,done,_=wrapped.step(action)
            positions.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy().copy())
            velocities.append(env.shuttle.data.root_lin_vel_w.cpu().numpy().copy())
            roots.append((env.robot.data.root_pos_w-env.scene.env_origins).cpu().numpy().copy())
            forces.append(env.shuttle_contact.data.net_forces_w[:,0].cpu().numpy().copy())
            done_counts.append(int(done.sum()))
            assert all(np.isfinite(values[-1]).all() for values in [positions,velocities,roots,forces])
            if (tick+1)%10==0 or stopping:
                atomic(args.output/'progress.json',dict(control_ticks=tick+1,actual_control_env_steps=(tick+1)*128,
                    current_shuttle_contact_force_max_n=float(np.linalg.norm(forces[-1],axis=-1).max()),
                    done_lanes=done_counts[-1],formal_acceptance=False))
                print(json.dumps(dict(tick=tick+1,done=done_counts[-1])),flush=True)
            if stopping:break
        p,v,body=np.stack(positions),np.stack(velocities),np.stack(roots)
        force=np.stack(forces);clock=np.arange(len(p))*.01
        np.savez_compressed(args.output/'native_probe.npz',ball_position_m=p,ball_velocity_mps=v,
            root_position_m=body,shuttle_contact_force_n=force,clock_s=clock,done_counts=done_counts,
            actual_environment_origins_m=env.scene.env_origins.cpu().numpy())
        if stopping:
            atomic(args.output/'interrupted.json',dict(actual_control_env_steps=(tick+1)*128,native_resume_claimed=False))
            return
        reference=ideal.sol(clock)[:3].T
        error=np.linalg.norm(p-reference[:,None,:],axis=-1)
        spread=np.linalg.norm(p-p[:,:1],axis=-1)
        root_spread=np.linalg.norm(body-body[:,:1],axis=-1)
        contacts=float(np.linalg.norm(force,axis=-1).max())
        numerical=sum(int(row['numerical_failures']) for row in env.rally_records)
        isolation=bool(contacts<=.001 and not any(done_counts) and root_spread.max()<=.001)
        result=dict(checked_at=datetime.now(timezone.utc).isoformat(),runtime_origin_spacing_m=args.env_spacing,
            actual_control_env_steps=6400,actual_native_samples=int(len(clock)*128),
            max_free_flight_position_error_m=float(error.max()),median_final_free_flight_position_error_m=float(np.median(error[-1])),
            max_replica_ball_position_spread_m=float(spread.max()),max_replica_root_position_spread_m=float(root_spread.max()),
            max_shuttle_contact_force_n=contacts,unexpected_done_count=sum(done_counts),numerical_failures=numerical,
            collision_isolation_diagnostic_passed=isolation,
            compact_precision_gate_passed=bool(isolation and numerical==0 and error.max()<=.001 and spread.max()<=.0001),
            scope='Same initial TRAIN29 free flight in all replicas; independent ball/robot overlap stress at spacing0. No autonomous task outcome inferred.',
            new_native_execution=True,task_performance_measured=False,formal_cases_read=False,formal_acceptance=False)
        atomic(args.output/'RESULT.json',result)
        atomic(args.output/'completed.json',dict(env_steps=6400))
        print(json.dumps(result),flush=True)
    finally:
        env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
