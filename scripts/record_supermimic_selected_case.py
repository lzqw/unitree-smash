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
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument(
    '--supermimic-checkpoint', type=Path,
    help='Optional supermimic_v1 actor checkpoint. --checkpoint remains the '
         'verified native G07 execution-contract checkpoint.',
)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--training-index', type=int, required=True)
parser.add_argument('--seed', type=int, default=882072)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=360)
parser.add_argument('--execution-root', type=Path, default=Path('/home/autodrive/LZQW/unitree_smash'))
parser.add_argument(
    '--offline-render', action='store_true',
    help='Run GPU PhysX without RTX cameras and save measured poses/ball trace for CPU MP4 rendering.',
)
parser.add_argument(
    '--capture-envs', type=int, default=32,
    help='Parallel duplicate environments used only by offline capture to amortize GPU PhysX overhead.',
)
parser.add_argument('--max-ticks', type=int, default=None,
                    help='Diagnostic cap; default runs the full task episode.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

def progress(message: str) -> None:
    print(f'[RECORDER] {message}', flush=True)

progress(f'parsed_args offline={args.offline_render} capture_envs={args.capture_envs} output={args.output}')

if args.fps <= 0 or args.width <= 0 or args.height <= 0:
    parser.error('fps, width and height must be positive')
if args.training_index < 0 or args.training_index >= 128:
    parser.error('training-index must be in [0, 127]')
if args.capture_envs < 1:
    parser.error('capture-envs must be positive')
if args.max_ticks is not None and args.max_ticks < 1:
    parser.error('max-ticks must be positive')
if args.output.exists() and any(args.output.iterdir()):
    raise ValueError(f'Output directory is not empty: {args.output}')
args.output.mkdir(parents=True, exist_ok=True)

app = AppLauncher(args).app
progress('app_ready')

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
progress('imports_ready')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


checkpoint = args.checkpoint.resolve()
progress(f'loading_checkpoint {checkpoint}')
source = torch.load(checkpoint, map_location='cpu', weights_only=False)
contract, protocol, all_cases, receipt = resolve_task_contract(
    source, checkpoint, execution_root=args.execution_root
)
case = all_cases[args.training_index]
capture_envs = args.capture_envs if args.offline_render else 1
progress(f'contract_ready capture_envs={capture_envs}')
cfg = make_task_cfg(capture_envs, contract, protocol, receipt)
cfg.seed = args.seed
cfg.sim.device = 'cuda:0'
cfg.viewer.resolution = (args.width, args.height)
cfg.sim.render_interval = max(1, round((1.0 / args.fps) / cfg.sim.dt))
noise = zero_panel_reset_noise(capture_envs, args.seed)

env = None
writer = None
try:
    progress('before_env')
    env = FrozenPanelEnv(
        cfg,
        cases=[case] * capture_envs,
        reset_noise=noise,
        render_mode=None if args.offline_render else 'rgb_array',
    )
    progress('env_ready')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    progress('wrapper_ready')
    controller = make_g07_residual_controller(
        source, capture_envs, env.device, execution_root=args.execution_root
    )
    progress('controller_ready')
    policy_meta = {
        'policy_kind': 'native_g07',
        'policy_checkpoint': str(checkpoint),
        'policy_checkpoint_sha256': digest(checkpoint),
    }
    if args.supermimic_checkpoint is not None:
        candidate_path = args.supermimic_checkpoint.resolve()
        candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
        if candidate.get('algorithm') != 'supermimic_v1':
            raise ValueError(
                f'Expected algorithm=supermimic_v1, got {candidate.get("algorithm")!r}'
            )
        actor_state = candidate.get('actor')
        if not isinstance(actor_state, dict):
            raise ValueError('SuperMimic checkpoint has no actor state_dict')
        # The native export already contains the exact gated base/recovery/
        # approach module.  Reuse that class and load the trainable actor
        # strictly after removing only the learner's ``actor.`` namespace.
        native_actor = torch.jit.load(
            io.BytesIO(source['residual_actor_jit']), map_location=env.device
        ).eval()
        exported_state = {
            key[len('actor.'):]: value
            for key, value in actor_state.items()
            if key.startswith('actor.')
        }
        expected_keys = set(native_actor.state_dict().keys())
        if set(exported_state) != expected_keys:
            missing = sorted(expected_keys - set(exported_state))
            extra = sorted(set(exported_state) - expected_keys)
            raise ValueError(f'SuperMimic/native actor schema mismatch; missing={missing}, extra={extra}')
        native_actor.load_state_dict(exported_state, strict=True)
        if not all(torch.isfinite(value).all() for value in exported_state.values()):
            raise ValueError('SuperMimic actor contains nonfinite parameters')
        controller.actor = native_actor
        policy_meta = {
            'policy_kind': 'supermimic_actor_on_native_g07_executor',
            'policy_checkpoint': str(candidate_path),
            'policy_checkpoint_sha256': digest(candidate_path),
            'execution_checkpoint': str(checkpoint),
            'execution_checkpoint_sha256': digest(checkpoint),
            'algorithm': candidate.get('algorithm'),
            'actor_updates': candidate.get('actor_updates'),
            'native_ticks': candidate.get('native_ticks'),
        }
    progress('before_reset')
    observation, _ = wrapped.reset()
    progress('after_reset')
    controller.reset()
    # The native G07 mixer accepts one 18D default vector and one 18x2 limit
    # table, then broadcasts them across capture lanes.  Passing batched
    # tensors here makes its arm limiter slice the lane dimension by mistake.
    default = env.robot.data.default_joint_pos[0, env.joint_ids]
    limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
    if not args.offline_render:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            str(args.output / 'episode.mp4'), fps=args.fps, codec='libx264', quality=7,
            macro_block_size=8, output_params=['-movflags', '+faststart'],
        )
    requested_ticks = round(contract['episode_s'] / contract['control_dt_s'])
    if args.max_ticks is not None:
        requested_ticks = min(requested_ticks, args.max_ticks)
    render_stride = max(1, round(1.0 / (args.fps * contract['control_dt_s'])))
    frames = 0
    measured_poses = []
    measured_trace = []
    with torch.inference_mode():
        for tick in range(requested_ticks):
            if tick % 100 == 0:
                progress(f'tick={tick}')
            tick_start = time.perf_counter()
            if tick == 0:
                progress('before_controller_action')
            action = controller.action(observation, default, limits)
            if tick == 0:
                progress(f'after_controller_action elapsed_s={time.perf_counter() - tick_start:.3f}')
            if action.shape != (capture_envs, 18) or not torch.isfinite(action).all():
                raise ValueError('Controller produced an invalid action')
            if tick % render_stride == 0:
                if args.offline_render:
                    measured_poses.append(torch.cat((
                        env.robot.data.body_link_pos_w[0],
                        env.robot.data.body_link_quat_w[0],
                    ), dim=1).cpu().numpy().copy())
                    ball = (env.shuttle.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy()
                    ball_v = env.shuttle.data.root_lin_vel_w[0].cpu().numpy()
                    measured_trace.append(dict(
                        time_s=float(tick * contract['control_dt_s']),
                        ball_xyz=ball.tolist(), ball_v=ball_v.tolist(),
                        airborne=bool(env.airborne_contact_seen[0]),
                        contact=bool(env.contact_seen[0]),
                    ))
                else:
                    frame = env.render()
                    if frame is None or not np.asarray(frame).any():
                        raise RuntimeError('Camera frame is empty; use --enable_cameras')
                    writer.append_data(np.asarray(frame))
                frames += 1
            step_start = time.perf_counter()
            observation, _, done, _ = wrapped.step(action)
            if tick == 0:
                progress(f'after_env_step elapsed_s={time.perf_counter() - step_start:.3f}')
            if bool(done[0]):
                progress(f'done_at_tick={tick}')
                break
    progress(f'loop_finished ticks={tick + 1} frames={frames}')
    if writer is not None:
        writer.close()
        writer = None
    if args.offline_render:
        progress('offline_extract_start')
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
        import trimesh

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        meshes, mesh_meta = {}, []
        robot_root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
            kind = prim.GetTypeName()
            if kind not in ('Mesh', 'Cube', 'Sphere', 'Cylinder', 'Capsule'):
                continue
            if '/collisions/' in str(prim.GetPath()).lower() or '/collision/' in str(prim.GetPath()).lower():
                continue
            ancestor = prim
            while ancestor.IsValid() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            if not ancestor.IsValid() or ancestor.GetName() not in env.robot.body_names:
                continue
            body = env.robot.body_names.index(ancestor.GetName())
            if kind == 'Mesh':
                shape = UsdGeom.Mesh(prim)
                vertices = np.asarray(shape.GetPointsAttr().Get(), dtype=np.float64)
                indices = list(shape.GetFaceVertexIndicesAttr().Get())
                counts = shape.GetFaceVertexCountsAttr().Get()
                faces, offset = [], 0
                for count in counts:
                    for j in range(1, int(count) - 1):
                        faces.append([indices[offset], indices[offset + j], indices[offset + j + 1]])
                    offset += int(count)
                faces = np.asarray(faces, dtype=np.int32)
            else:
                if kind == 'Cube':
                    shape = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
                elif kind == 'Sphere':
                    shape = trimesh.creation.icosphere(subdivisions=2, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
                else:
                    usd_shape = UsdGeom.Cylinder(prim) if kind == 'Cylinder' else UsdGeom.Capsule(prim)
                    shape = trimesh.creation.cylinder(
                        radius=float(usd_shape.GetRadiusAttr().Get()),
                        height=float(usd_shape.GetHeightAttr().Get()), sections=24,
                    )
                    axis = usd_shape.GetAxisAttr().Get()
                    if axis == 'X':
                        shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
                    elif axis == 'Y':
                        shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                vertices, faces = np.asarray(shape.vertices), np.asarray(shape.faces, dtype=np.int32)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(ancestor).GetInverse())
            vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3].astype(np.float32)
            key = f'm{len(mesh_meta)}'
            meshes[f'{key}_vertices'] = vertices
            meshes[f'{key}_faces'] = faces
            color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            mesh_meta.append(dict(key=key, body=body, path=str(prim.GetPath()), color=list(color[0]) if color else None))
        if not mesh_meta:
            raise RuntimeError('No renderable robot body meshes found for offline capture')
        np.savez_compressed(args.output / 'geometry.npz', **meshes)
        (args.output / 'geometry.json').write_text(json.dumps(
            dict(meshes=mesh_meta, body_names=env.robot.body_names), indent=2
        ) + '\n')
        np.save(args.output / 'link_poses.npy', np.stack(measured_poses))
        (args.output / 'trajectory.json').write_text(json.dumps(measured_trace, indent=2) + '\n')
        progress(f'offline_extract_done poses={len(measured_poses)}')
    metadata = dict(
        schema='supermimic-selected-case-video-v1',
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        policy=policy_meta,
        training_index=args.training_index,
        case_id=case.get('case_id'),
        seed=args.seed,
        control_dt_s=contract['control_dt_s'],
        episode_s=contract['episode_s'],
        requested_ticks=requested_ticks,
        frames=frames,
        fps=args.fps,
        resolution=[args.width, args.height],
        camera_mode='CPU offline renderer from measured GPU-PhysX poses' if args.offline_render else 'Isaac Lab rgb_array camera; one selected environment',
        offline_capture=bool(args.offline_render),
        capture_envs=capture_envs,
        pose_samples=len(measured_poses),
        physics_device='cuda:0',
        native_execution_contract=contract,
        event_records=env.rally_records,
        contact_records=env.contact_records,
        formal_acceptance=False,
        training_score_claim=False,
    )
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    progress('metadata_written')
    print(json.dumps(metadata, default=str), flush=True)
except BaseException as exc:
    (args.output / 'failure.txt').write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf-8')
    progress(f'failure {type(exc).__name__}: {exc}')
    raise
finally:
    progress('cleanup_start')
    if writer is not None:
        writer.close()
    if env is not None:
        env.close()
    app.close(skip_cleanup=True)
    progress('cleanup_done')
