#!/usr/bin/env python3
"""Run one fixed policy over the original TRAIN cases, optionally perturbed.

This deliberately shares the formal environment and original per-episode
criteria but never opens formal case files or emits a formal-panel score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--training-indices', type=int, nargs='+', default=None)
parser.add_argument('--noise', choices=('none', 'A', 'B'), default='none')
parser.add_argument('--seed', type=int, default=724201)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, training_noise, summarize_training
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.usd_dependencies_v4 import audit_composed_asset


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'INPUTS.json').exists():
        raise ValueError('Use a fresh output; never overwrite a previous diagnostic')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text())
    cases, indices = training_cases(ROOT, protocol, args.training_indices)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not checkpoint.get('native_execution'):
        raise ValueError('Evaluate an actually executed policy; unexecuted proposals require training first')
    if checkpoint['parameters'].ndim != 1 or not torch.isfinite(checkpoint['parameters']).all():
        raise ValueError('Require one fixed finite policy, not an optimizer population')
    if checkpoint['asset_sha256'] != protocol['asset']['sha256']:
        raise ValueError('Checkpoint belongs to a different physical model')
    if digest(ROOT / protocol['asset']['path']) != protocol['asset']['sha256']:
        raise ValueError('Frozen physical asset changed')
    for name, expected in protocol['physical_configs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen physical config changed: '+name)
    amplitudes = (dict(joint_position_rad=0., joint_velocity_rad_s=0., root_xy_m=0., root_yaw_rad=0.)
                  if args.noise == 'none' else protocol['initialization']['perturbations'][args.noise])
    count = len(cases)
    noise = training_noise(count, amplitudes, args.seed)
    atomic_json(args.output / 'INPUTS.json', dict(policy_sha256=digest(args.checkpoint),
        protocol_sha256=digest(protocol_path), training_suite=protocol['training_suite'],
        training_indices=indices, reset_noise_amplitudes=amplitudes, reset_noise_seed=args.seed,
        reset_noise={k: v.tolist() for k, v in noise.items()},
        scope='TRAIN-only one-policy diagnostic; no inherited performance or formal cases.',
        formal_cases_read=False, formal_acceptance=False))
    cfg = full_racket_cfg(count, protocol['physics_dt_s'])
    cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    if abs(cfg.sim.dt*cfg.decimation-protocol['policy_dt_s']) > 1e-12:
        raise ValueError('Control period differs from original protocol')
    env = FrozenPanelEnv(cfg, cases=cases, reset_noise=noise)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    controller = make_rally_controller(checkpoint, count, env.device)
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if env.robot.is_fixed_base:
            raise ValueError('Require a floating robot with an actual ground reset')
        atomic_json(args.output / 'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        observation, _ = wrapped.reset()
        controller.reset()
        initial = {r['env_id']: r for r in env.initial_states}
        if set(initial) != set(range(count)):
            raise ValueError('Missing first ground reset states')
        for i, row in initial.items():
            position = np.array(row['root_state_local'][:3])
            expected = np.r_[noise['root_xy_m'][i], protocol['initialization']['root_position_m'][2]]
            if np.max(abs(position-expected)) > 1e-5:
                raise ValueError('Native reset origin/height differs from the requested ground initialization')
        atomic_json(args.output / 'initial_states.json', [initial[i] for i in range(count)])
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        poses, balls, clocks, valid = [], [], [], []
        public, actions, public_valid = [], [], []
        requested = round(protocol['episode_seconds']/protocol['policy_dt_s'])
        with torch.inference_mode():
            for tick in range(requested):
                obs = observation['policy']
                action = controller.action(observation, default, limits)
                if action.shape != (count, 18) or not torch.isfinite(action).all() or (abs(action)>1).any():
                    raise ValueError('Nonfinite, invalid-width or unbounded native action')
                public.append(obs[:2].cpu().numpy().copy())
                actions.append(action[:2].cpu().numpy().copy())
                public_valid.append((~finished[:2]).cpu().numpy().copy())
                if tick % 2 == 0:
                    data = env.robot.data
                    pose = torch.cat((data.body_link_pos_w-env.scene.env_origins[:, None], data.body_link_quat_w), -1).cpu().numpy()
                    ball = (env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy()
                    clock = obs[:, 775].cpu().numpy().copy()
                    if not all(np.isfinite(a).all() for a in [pose, ball, clock]):
                        raise ValueError('Nonfinite actual body/ball/clock stream')
                    poses.append(pose);balls.append(ball);clocks.append(clock)
                    valid.append((~finished).cpu().numpy().copy())
                observation, _, done, _ = wrapped.step(action)
                finished |= done.bool()
                if (tick+1) % 100 == 0 or stopping or bool(finished.all()):
                    atomic_json(args.output / 'progress.json', dict(control_ticks=tick+1,
                        requested_ticks=requested, first_episodes_complete=int(finished.sum()),
                        training_indices=indices, formal_eligible=False, formal_acceptance=False))
                    atomic_json(args.output / 'episodes.partial.json', env.rally_records)
                    np.savez_compressed(args.output / 'all_first_episodes.partial.npz',
                        link_poses=np.stack(poses), ball_position_m=np.stack(balls),
                        episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
                    print(json.dumps(dict(tick=tick+1, first_complete=int(finished.sum()))), flush=True)
                if stopping or bool(finished.all()):
                    break
        np.savez_compressed(args.output / 'all_first_episodes.npz', link_poses=np.stack(poses),
            ball_position_m=np.stack(balls), episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
        np.savez_compressed(args.output / 'public_and_actions_first2.npz', observations=np.stack(public),
            actions=np.stack(actions), valid_first_episode=np.stack(public_valid))
        atomic_json(args.output / 'contacts.json', env.contact_records)
        atomic_json(args.output / 'causal_plans.json', controller.logs)
        first, jumps = {}, {}
        for row in env.rally_records:
            first.setdefault(row['env_id'], row)
        for row in env.episode_records:
            jumps.setdefault(row['env_id'], row)
        if stopping and (len(first) != count or len(jumps) != count):
            atomic_json(args.output / 'interrupted.json', dict(control_ticks=tick+1,
                actual_control_env_steps=(tick+1)*count, completed_first_episodes=len(first),
                requested_first_episodes=count, formal_acceptance=False,
                resume_semantics='Replay the same fixed policy/cases/noise from ground; no state restoration claimed.'))
            return
        if len(first) != count or len(jumps) != count:
            raise ValueError('Incomplete first-episode TRAIN diagnostic')
        rows = [dict(first[i], jump=jumps[i], case_id=cases[i]['case_id'], ground_init=True,
                     source_training_index=indices[i]) for i in range(count)]
        protocol['_root'] = str(ROOT)
        summary = summarize_training(rows, cases, indices, protocol)
        summary.update(policy_sha256=digest(args.checkpoint), protocol_sha256=digest(protocol_path),
            noise_level=args.noise, noise_seed=args.seed, actual_control_env_steps=(tick+1)*count)
        atomic_json(args.output / 'episodes.json', rows)
        atomic_json(args.output / 'metrics.json', summary)
        atomic_json(args.output / 'completed.json', dict(env_steps=(tick+1)*count))
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
