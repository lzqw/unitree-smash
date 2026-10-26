#!/usr/bin/env python3
"""Independent frozen200-case full-racket panel with actual reset disturbances."""
import argparse,csv,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--policy-lock',type=Path,required=True)
p.add_argument('--protocol',type=Path,default=ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json')
p.add_argument('--panel',choices=('A','B'),required=True)
p.add_argument('--precision-receipt',type=Path,required=True)
from scripts.origin0_launcher_parser_v4 import add_origin0_launcher_args
add_origin0_launcher_args(p,AppLauncher);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.evaluation_v4 import validate_frozen_inputs,summarize_panel,digest
from quadsmash.tasks.stand.full_racket_origin0_v4 import full_racket_origin0_cfg,validate_protocol,validate_admission
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from scripts.lock_full_racket_origin0_current_policy_v4 import require_ready_training
from quadsmash.usd_dependencies_v4 import audit_composed_asset


def noise_to_json(noise):
    return {key:value.tolist() for key,value in noise.items()}


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'INPUTS.json').exists() or (args.output/'completed.json').exists():
        raise ValueError('Use a fresh formal output; preserve every earlier attempt')
    preview_protocol=json.loads(args.protocol.read_text());validate_protocol(preview_protocol)
    preview_lock=json.loads(args.policy_lock.read_text())
    readiness=preview_lock['TRAIN128_readiness']
    if require_ready_training(ROOT,preview_protocol,args.checkpoint,Path(readiness['evaluation_root']))!=readiness:
        raise ValueError('Actual readiness changed before opening formal cases')
    protocol,lock=validate_frozen_inputs(ROOT,args.protocol,args.policy_lock,args.checkpoint)
    validate_protocol(protocol);validate_admission(args.precision_receipt)
    if lock.get('origin_precision_receipt_sha256')!=digest(args.precision_receipt) or lock.get('numerical_execution')!=protocol['numerical_execution']:
        raise ValueError('Origin numerical execution or precision admission differs from lock')
    own=str(Path(__file__).resolve().relative_to(ROOT))
    if lock['source_hashes'].get(own)!=digest(Path(__file__)) or lock.get('current_evaluator')!=own:
        raise ValueError('Current formal evaluator must be present in the immutable policy lock')
    if 'TRAIN128_readiness' not in lock:
        raise ValueError('Missing actual TRAIN128 readiness admission')
    suite=protocol['formal_suites'][args.panel];cases=json.loads((ROOT/suite['path']).read_text())['cases'];count=len(cases)
    rng=np.random.default_rng(protocol['initialization']['reset_seeds'][args.panel])
    amplitudes=protocol['initialization']['perturbations'][args.panel]
    noise={name:rng.uniform(-amplitudes[name],amplitudes[name],shape) for name,shape in
        [('joint_position_rad',(count,18)),('joint_velocity_rad_s',(count,18)),('root_xy_m',(count,2)),('root_yaw_rad',(count,))]}
    cfg=full_racket_origin0_cfg(count,protocol,args.precision_receipt);cfg.seed=protocol['initialization']['reset_seeds'][args.panel]
    cfg.episode_length_s=protocol['episode_seconds']
    if Path(cfg.robot.spawn.usd_path).resolve()!=(ROOT/protocol['asset']['path']).resolve():
        raise ValueError('Active environment asset differs from frozen protocol')
    if abs(cfg.sim.dt*cfg.decimation-protocol['policy_dt_s'])>1e-12:
        raise ValueError('Control period differs from frozen protocol')
    env=FrozenPanelEnv(cfg,cases=cases,reset_noise=noise);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    controller=make_rally_controller(torch.load(args.checkpoint,map_location='cpu',weights_only=False),count,env.device)
    try:
        dependency_audit=audit_composed_asset(ROOT,cfg.robot.spawn.usd_path)
        (args.output/'asset_dependencies.json').write_text(json.dumps(dependency_audit,indent=2)+'\n')
        if env.robot.is_fixed_base:raise ValueError('Formal robot must be floating and start from ground')
        observation,_=wrapped.reset();controller.reset()
        if env.scene.env_origins.any():raise ValueError('Formal actual origin0 replicas differ')
        (args.output/'origin_layout.json').write_text(json.dumps(dict(actual_environment_origins_m=env.scene.env_origins.cpu().tolist(),origin_precision_receipt_sha256=digest(args.precision_receipt)),indent=2)+'\n')
        export_robot_geometry(get_current_stage(),env.robot,args.output)
        first_initial={row['env_id']:row for row in env.initial_states}
        if len(first_initial)!=200:raise ValueError('Missing first ground reset states')
        for i,row in first_initial.items():
            if abs(row['root_state_local'][2]-protocol['initialization']['root_position_m'][2])>1e-6:
                raise ValueError('Initial root height differs from frozen ground reset')
        (args.output/'initial_states.json').write_text(json.dumps([first_initial[i] for i in range(count)],indent=2)+'\n')
        default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        (args.output/'INPUTS.json').write_text(json.dumps(dict(policy_sha256=lock['policy_sha256'],
            protocol_sha256=lock['protocol_sha256'],policy_lock_sha256=digest(args.policy_lock),
            evaluator_source_sha256=digest(Path(__file__)),panel=args.panel,reset_noise=noise_to_json(noise),
            origin_spacing_m=0.,origin_precision_receipt_sha256=digest(args.precision_receipt),
            default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),
            formal_cases_read=True,formal_acceptance=False),indent=2)+'\n')
        finished=torch.zeros(count,dtype=torch.bool,device=env.device);poses,trace=[],[]
        public,actions,public_valid=[],[],[]
        all_poses,all_balls,all_clocks,all_valid=[],[],[],[]
        ticks=round(protocol['episode_seconds']/protocol['policy_dt_s'])
        with torch.inference_mode():
            for tick in range(ticks):
                if tick%2==0:
                    data=env.robot.data
                    all_poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],data.body_link_quat_w),-1).cpu().numpy())
                    all_balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                    all_clocks.append(observation['policy'][:,775].cpu().numpy().copy())
                    all_valid.append((~finished).cpu().numpy().copy())
                    poses.append(torch.cat((data.body_link_pos_w[:2]-env.scene.env_origins[:2,None],data.body_link_quat_w[:2]),-1).cpu().numpy())
                    trace.append(dict(time_s=tick*.01,valid_first=(~finished[:2]).cpu().tolist(),
                        ball_position_m=(env.shuttle.data.root_pos_w[:2]-env.scene.env_origins[:2]).cpu().tolist(),
                        ball_velocity_mps=env.shuttle.data.root_lin_vel_w[:2].cpu().tolist()))
                action=controller.action(observation,default,limits)
                if action.shape!=(count,18) or not torch.isfinite(action).all() or (action.abs()>1).any():
                    raise ValueError('Nonfinite or unbounded formal native action')
                public.append(observation['policy'][:2].cpu().numpy().copy())
                actions.append(action[:2].cpu().numpy().copy())
                public_valid.append((~finished[:2]).cpu().numpy().copy())
                observation,_,done,_=wrapped.step(action);finished|=done.bool()
                ids=torch.where(done.bool())[0]
                if len(ids):controller.reset(ids)
                if (tick+1)%100==0:
                    progress=dict(control_ticks=tick+1,requested_ticks=ticks,first_episodes_complete=int(finished.sum()),
                        formal_acceptance=False,formal_eligible=True,panel=args.panel)
                    (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    (args.output/'episodes.partial.json').write_text(json.dumps(env.rally_records,indent=2)+'\n')
                    np.savez_compressed(args.output/'all_first_episodes.partial.npz',link_poses=np.stack(all_poses),
                        ball_position_m=np.stack(all_balls),episode_clock_s=np.stack(all_clocks),
                        valid_first_episode=np.stack(all_valid))
                    print(json.dumps(progress),flush=True)
                if bool(finished.all()):break
        np.savez_compressed(args.output/'all_first_episodes.npz',link_poses=np.stack(all_poses),
            ball_position_m=np.stack(all_balls),episode_clock_s=np.stack(all_clocks),valid_first_episode=np.stack(all_valid))
        np.savez_compressed(args.output/'public_and_actions_first2.npz',observations=np.stack(public),
            actions=np.stack(actions),valid_first_episode=np.stack(public_valid))
        first,jumps={},{}
        for row in env.rally_records:first.setdefault(row['env_id'],row)
        for row in env.episode_records:jumps.setdefault(row['env_id'],row)
        if len(first)!=200 or len(jumps)!=200:raise ValueError('Incomplete first-episode panel')
        rows=[dict(first[i],jump=jumps[i],case_id=cases[i]['case_id'],ground_init=True) for i in range(count)]
        protocol['_root']=str(ROOT);summary=summarize_panel(rows,protocol,args.panel)
        summary.update(policy_sha256=lock['policy_sha256'],protocol_sha256=lock['protocol_sha256'],
            policy_lock_sha256=digest(args.policy_lock),formal_acceptance=False,
            origin_spacing_m=0.,origin_precision_receipt_sha256=digest(args.precision_receipt),
            actual_control_env_steps=(tick+1)*count,
            statement='Single complete panel only; thresholds and400-case outcome require both panels.')
        (args.output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'contacts.json').write_text(json.dumps(env.contact_records,indent=2)+'\n')
        (args.output/'causal_plans.json').write_text(json.dumps(controller.logs,indent=2)+'\n')
        np.save(args.output/'link_poses_first2.npy',np.stack(poses))
        (args.output/'trajectory_first2.json').write_text(json.dumps(trace)+'\n')
        with (args.output/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
        (args.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        (args.output/'completed.json').write_text(json.dumps(dict(env_steps=(tick+1)*count))+'\n')
        print(json.dumps(summary),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
