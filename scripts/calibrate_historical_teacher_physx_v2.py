#!/usr/bin/env python3
"""From-reset replay of frozen historical actions; NO planner or policy evaluation."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--precision-receipt',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.output.exists():raise ValueError('Fresh calibration output required')
args.output.mkdir(parents=True)
app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.control.full_racket_train_evaluation_v4 import training_noise
from scripts.native_isolation_microtrace_v1 import NativeIsolationMicrotraceV1


def main():
    actions=np.load(args.source/'public_actions.npy',allow_pickle=False)
    observed=np.load(args.source/'public_observations.npy',allow_pickle=False)
    if actions.shape!=(800,2,18) or observed.shape!=(800,2,794) or not np.isfinite(actions).all():
        raise ValueError('Original two-lane 800-tick frozen trace required')
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    suite=ROOT/protocol['training_suite']['path']
    if hashlib.sha256(suite.read_bytes()).hexdigest()!=protocol['training_suite']['sha256']:raise ValueError('Panel drift')
    cases=json.loads(suite.read_text())['cases'];cases=[cases[0],cases[64]]
    cfg=contact_recovery_origin0_cfg(2,protocol,args.precision_receipt);cfg.seed=882072;cfg.ground_size_m=(64.,64.,.1)
    noise=training_noise(2,dict(joint_position_rad=0.,joint_velocity_rad_s=0.,root_xy_m=0.,root_yaw_rad=0.),cfg.seed)
    env=FrozenPanelEnv(cfg,cases=cases,reset_noise=noise,contact_flight_recovery=protocol['recovery_association']['schema'])
    wrapper=RslRlVecEnvWrapper(env,clip_actions=None);micro=None;rows=[];buffers={}
    try:
        obs,_=wrapper.reset()
        initial=float(np.max(abs(obs['policy'].cpu().numpy()-observed[0])))
        (args.output/'INITIAL_STATE.json').write_text(json.dumps(dict(public_max_delta=initial,source_hashes={str(args.source/name):hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in ['public_actions.npy','public_observations.npy']},physics_dt_s=cfg.sim.dt,control_dt_s=env.step_dt),indent=2))
        if initial>1e-4:raise ValueError('From-reset initial state mismatch; no calibration continuation')
        micro=NativeIsolationMicrotraceV1(env,args.output,lanes=(0,1),first_frame=249*160+1,last_frame=263*160)
        limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        with torch.inference_mode():
            for tick in range(800):
                d=env.robot.data
                values=dict(public=obs['policy'],joint_position=d.joint_pos[:,env.joint_ids],joint_velocity=d.joint_vel[:,env.joint_ids],
                    applied_drive_torque=d.applied_torque[:,env.joint_ids],root_position=d.root_link_pos_w-env.scene.env_origins,
                    root_quaternion=d.root_link_quat_w,root_linear_velocity=d.root_lin_vel_w,root_angular_velocity=d.root_ang_vel_w,
                    foot_force=env.feet_contact.data.net_forces_w,contacts=env.observed_foot_contact,
                    previous_target=env.targets[:,env.joint_ids],episode_index=env.episode_index)
                for k,v in values.items():buffers.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
                request=torch.tensor(actions[tick],device=env.device)
                expected=limits[:,0]+.5*(request.clamp(-1,1)+1)*(limits[:,1]-limits[:,0])
                prior_episode=env.episode_index.clone()
                obs,_,done,_=wrapper.step(request)
                mask=env.episode_index==prior_episode
                delta=float((env.targets[:,env.joint_ids][mask]-expected[mask]).abs().max()) if mask.any() else 0.
                rows.append(dict(tick=tick,dispatch_delta_rad=delta,public_delta=float(np.max(abs(buffers['public'][-1]-observed[tick]))),done=done.cpu().tolist()))
                if delta>2e-6:raise ValueError('Recorded action dispatch mismatch')
                if (tick+1)%100==0:
                    (args.output/'progress.json').write_text(json.dumps(dict(ticks=tick+1,kind='history_physics_calibration',policy_evaluation=False)))
                    print('CALIBRATION_TICK',tick+1,flush=True)
        np.savez_compressed(args.output/'control_state.npz',**{k:np.asarray(v) for k,v in buffers.items()})
        (args.output/'CALIBRATION_COMPLETE.json').write_text(json.dumps(dict(ticks=800,initial_public_max_delta=initial,
            dispatch_max_delta_rad=max(r['dispatch_delta_rad'] for r in rows),public_max_delta=max(r['public_delta'] for r in rows),
            policy_evaluation=False,actor_forwards=0,optimizer_steps=0,physx_reference_valid=False,rows=rows),indent=2))
    finally:
        if micro is not None:micro.save()
        if buffers and not (args.output/'control_state.npz').exists():np.savez_compressed(args.output/'partial_state.npz',**{k:np.asarray(v) for k,v in buffers.items()})
        env.close()
try:main()
except BaseException:
    (args.output/'ERROR.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
