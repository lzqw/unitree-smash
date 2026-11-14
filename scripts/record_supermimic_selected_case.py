#!/usr/bin/env python3
"""Record one frozen full-racket case to MP4 without changing task physics.

This is a post-training recorder. It runs one selected TRAIN case in a fresh
Isaac process, uses the native G07 execution contract, and writes an RGB clip.
It is intentionally separate from the 128-environment learner so camera
rendering cannot change learning throughput or the training replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--training-index', type=int, required=True)
parser.add_argument('--seed', type=int, default=882072)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=360)
parser.add_argument('--execution-root', type=Path, default=Path('/home/autodrive/LZQW/unitree_smash'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.fps <= 0 or args.width <= 0 or args.height <= 0:
    parser.error('fps, width and height must be positive')
if args.training_index < 0 or args.training_index >= 128:
    parser.error('training-index must be in [0, 127]')
if args.output.exists() and any(args.output.iterdir()):
    raise ValueError(f'Output directory is not empty: {args.output}')
args.output.mkdir(parents=True, exist_ok=True)

app = AppLauncher(args).app

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
from quadsmash.rl.supermimic_task_contract import (
    make_task_cfg,
    resolve_task_contract,
    zero_panel_reset_noise,
)
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


checkpoint = args.checkpoint.resolve()
source = torch.load(checkpoint, map_location='cpu', weights_only=False)
contract, protocol, all_cases, receipt = resolve_task_contract(
    source, checkpoint, execution_root=args.execution_root
)
case = all_cases[args.training_index]
cfg = make_task_cfg(1, contract, protocol, receipt)
cfg.seed = args.seed
cfg.sim.device = 'cuda:0'
cfg.viewer.resolution = (args.width, args.height)
cfg.sim.render_interval = max(1, round((1.0 / args.fps) / cfg.sim.dt))
noise = zero_panel_reset_noise(1, args.seed)

env = None
writer = None
try:
    env = FrozenPanelEnv(
        cfg,
        cases=[case],
        reset_noise=noise,
        render_mode='rgb_array',
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    controller = make_g07_residual_controller(
        source, 1, env.device, execution_root=args.execution_root
    )
    observation, _ = wrapped.reset()
    controller.reset()
    default = env.robot.data.default_joint_pos[0, env.joint_ids]
    limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
    writer = imageio.get_writer(
        str(args.output / 'episode.mp4'),
        fps=args.fps,
        codec='libx264',
        quality=7,
        macro_block_size=8,
        output_params=['-movflags', '+faststart'],
    )
    requested_ticks = round(contract['episode_s'] / contract['control_dt_s'])
    render_stride = max(1, round(1.0 / (args.fps * contract['control_dt_s'])))
    frames = 0
    with torch.inference_mode():
        for tick in range(requested_ticks):
            action = controller.action(observation, default, limits)
            if action.shape != (1, 18) or not torch.isfinite(action).all():
                raise ValueError('Controller produced an invalid action')
            if tick % render_stride == 0:
                frame = env.render()
                if frame is None or not np.asarray(frame).any():
                    raise RuntimeError('Camera frame is empty; use --enable_cameras')
                frame = np.asarray(frame)
                writer.append_data(frame)
                frames += 1
            observation, _, done, _ = wrapped.step(action)
            if bool(done[0]):
                break
    writer.close()
    writer = None
    metadata = dict(
        schema='supermimic-selected-case-video-v1',
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        training_index=args.training_index,
        case_id=case.get('case_id'),
        seed=args.seed,
        control_dt_s=contract['control_dt_s'],
        episode_s=contract['episode_s'],
        requested_ticks=requested_ticks,
        frames=frames,
        fps=args.fps,
        resolution=[args.width, args.height],
        camera_mode='Isaac Lab rgb_array camera; one selected environment',
        physics_device='cuda:0',
        native_execution_contract=contract,
        event_records=env.rally_records,
        contact_records=env.contact_records,
        formal_acceptance=False,
        training_score_claim=False,
    )
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(json.dumps(metadata, default=str), flush=True)
finally:
    if writer is not None:
        writer.close()
    if env is not None:
        env.close()
    app.close(skip_cleanup=True)
