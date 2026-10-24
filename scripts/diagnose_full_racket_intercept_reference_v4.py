#!/usr/bin/env python3
"""Audit prehit task references on saved public TRAIN states, without dynamics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.initial_intercept import predict_initial_intercept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot', 'checkpoint', 'joint-contract', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--proposed-plane', type=float, default=1.15)
    parser.add_argument('--proposed-speed', type=float, default=1.5)
    parser.add_argument('--proposed-calibration', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state_path = args.snapshot/'partial_search_state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    public_path, plans_path = [args.snapshot/name for name in ['near_contact_all.partial.npz', 'causal_plans.partial.json']]
    physics_path = ROOT/'configs/shuttle_physics_v2.json'
    physics = json.loads(physics_path.read_text())
    plans = json.loads(plans_path.read_text())
    with np.load(public_path) as saved:
        count = saved['observations'].shape[1]
    population = state['parameters'].numpy().repeat(count//len(state['parameters']), axis=0)
    event = np.full((2, count), 2.75)
    targets = np.zeros((2, count, 3))
    nominal = np.array(warm['seed']['desired_ball_center_m'])
    calibration = warm.get('event_calibration_s', 2.75-warm['result']['event_time_s'])
    for log in plans:
        ids = np.array(log['env_ids'])
        public = np.array(log['used_public_observations'])
        fresh = predict_initial_intercept(public, [1.1, 0, 1.8], [1., 0, 0],
            np.full(len(ids), args.proposed_plane), physics, dt=.001, max_seconds=1.5)
        proposed_event = public[:, 775]+fresh['arrival_time_s']+args.proposed_calibration
        for mode, times, positions, valid in [
            (0, np.array(log['calibrated_event_time_s']), np.array(log['position_m']), np.array(log['valid'])),
            (1, proposed_event, fresh['position_m'], fresh['valid'])]:
            valid = valid & np.isfinite(times) & (times >= 2.3) & (times <= 3.3)
            event[mode, ids] = np.where(valid, times, 2.75)
            targets[mode, ids] = np.where(valid[:, None], positions-nominal, 0.)
    reset_path = ROOT/'configs/full_racket_reset_v4.json'
    default = torch.tensor(json.loads(reset_path.read_text())['arm_joint_positions_rad'], dtype=torch.float32)
    model = FullRacketKinematicsV4().float()
    offset = torch.tensor([warm['seed'].get('face_sign', 1)*.0256, *warm['seed'].get('string_uv_m', [0., 0.])])
    old_velocity = np.array(warm['seed']['desired_point_velocity_mps'])
    proposed_velocity = old_velocity.copy()
    proposed_velocity[0] = args.proposed_speed
    results, masks = [], []
    with np.load(public_path) as saved, torch.inference_mode():
        for i, obs in enumerate(saved['observations']):
            joints = default+torch.tensor(obs[:, 21:27])
            point = model(joints, offset)[0].numpy()
            forward, up = obs[:, 763:766], -obs[:, 6:9]
            left = np.cross(up, forward)
            quat = saved['link_poses'][i, :, 0, 3:]
            rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
            root = saved['link_poses'][i, :, 0, :3]
            values = []
            for mode, velocity_world in enumerate([old_velocity, proposed_velocity]):
                target = obs[:, 760:763].copy()
                delta = targets[mode]
                target += delta[:, :1]*forward+delta[:, 1:2]*left+delta[:, 2:3]*up
                velocity = velocity_world[0]*forward+velocity_world[1]*left+velocity_world[2]*up
                relative = velocity-obs[:, :3]-np.cross(obs[:, 3:6], point)
                guided = target+velocity*np.clip(obs[:, 775]-event[mode], -.15, .10)[:, None]+.06*relative
                values.append(root+np.einsum('nij,nj->ni', rotation, guided))
            active = saved['valid_first_episode'][i] & (obs[:, 773] < .5) & ((obs[:, 772]+obs[:, 773]) > .5)
            active &= obs[:, 775] > event[0]-.25
            results.append(np.stack(values))
            masks.append(active)
    data, mask = np.stack(results), np.stack(masks)
    assert mask.any() and np.isfinite(data[mask[:, None].repeat(2, axis=1)]).all()
    def stats(mode):
        value = data[:, mode, :, 0][mask]
        return dict(samples=len(value), minimum=float(value.min()), median=float(np.median(value)),
                    maximum=float(value.max()), beyond_net_plane=int(np.sum(value > 1.5)),
                    beyond_joint_guard_reference=int(np.sum(value > 1.38)))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'prehit_task_references.npz', world_reference_m=data, baseline_active_unhit_mask=mask,
                        forecast_event_time_s=event, forecast_target_offset_world_m=targets)
    paths = [args.checkpoint, state_path, public_path, plans_path, reset_path, physics_path, Path(__file__),
             ROOT/'quadsmash/control/full_racket_teacher_v4.py', ROOT/'quadsmash/control/initial_intercept.py']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Original public TRAIN states and original causal ball observations at launch. Raw Cartesian task references before IK/net projection; alternative forecasts use the same past observations. No changed-state dynamics or success outcome.',
        original_forward_velocity_target_mps=float(old_velocity[0]), proposed_forward_velocity_target_mps=args.proposed_speed,
        proposed_intercept_plane_m=args.proposed_plane, original_raw_task_reference_x_m=stats(0),
        proposed_event_calibration_s=args.proposed_calibration, proposed_timing_residual_s=0.,
        proposed_raw_task_reference_x_m=stats(1), new_dynamics_executed=False, formal_acceptance=False,
        sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
