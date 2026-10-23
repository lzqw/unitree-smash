#!/usr/bin/env python3
"""Freeze physical initial serves and optional original ground perturbations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.initial_launch import solve_initial_launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=717201)
    parser.add_argument('--reset-distribution', choices=['none', 'a', 'b'], default='none')
    parser.add_argument('--include-nominal-control', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new immutable panel directory')
    args.output.mkdir(parents=True, exist_ok=True)
    teacher_path = ROOT/'reports/G07/compact_native_contact01/net_free_teacher214_v2.pt'
    launch_path = ROOT/'reports/G07/compact_native_contact01/launches.json'
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    reference = json.loads(launch_path.read_text())[37]
    point = np.array(reference['target_position_m'])
    rng = np.random.Generator(np.random.PCG64(args.seed))
    targets = np.tile(point, (args.count, 1))
    targets[:, 1] += rng.uniform(-.10, .10, args.count)
    targets[:, 2] += rng.uniform(-.03, .03, args.count)
    times = rng.uniform(4.60, 5.25, args.count)
    vx = rng.choice([-8., -12., -20., -30., -45., -70.], args.count)
    if args.include_nominal_control:
        targets[0] = point
        times[0] = reference['arrival_time_s']
        vx[0] = reference['shuttle_velocity'][0]
    scenarios = solve_initial_launches(targets, times, vx)
    reset_protocol = None
    if args.reset_distribution != 'none':
        reset_protocol = json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['panels'][0 if args.reset_distribution == 'a' else 1]
        reset_rng = np.random.Generator(np.random.PCG64(reset_protocol['seed']))
        dq = reset_rng.uniform(-reset_protocol['joint_offset_uniform_rad'], reset_protocol['joint_offset_uniform_rad'], (args.count, 18))
        qd = reset_rng.uniform(-reset_protocol['joint_velocity_uniform_rad_s'], reset_protocol['joint_velocity_uniform_rad_s'], (args.count, 18))
        yaw = reset_rng.uniform(-reset_protocol['yaw_uniform_rad'], reset_protocol['yaw_uniform_rad'], args.count)
        xy = reset_rng.uniform(-reset_protocol['xy_uniform_m'], reset_protocol['xy_uniform_m'], (args.count, 2))
        for index, row in enumerate(scenarios):
            row['robot_reset'] = dict(joint_offset_rad=dq[index].tolist(), joint_velocity_rad_s=qd[index].tolist(),
                                      yaw_rad=float(yaw[index]), xy_m=xy[index].tolist())
    panel = dict(version='g07-causal-long-prepare-development-v1', panel_id=args.output.name,
                 created_at=datetime.now(timezone.utc).isoformat(), seed=args.seed, formal_acceptance=False,
                 hardware_changed=False, net_geometry_version='environment-local-net-v2',
                 initial_state_only='Shuttle state written once at ground reset; no post-reset corrections',
                 controller_access='Public observation only; per-scenario arrival/target labels unavailable to policy',
                 ranges=dict(arrival_time_s=[4.60, 5.25], intercept_y_offset_m=[-.10, .10],
                             intercept_z_offset_m=[-.03, .03], initial_vx_mps=[-8., -12., -20., -30., -45., -70.]),
                 reset_distribution=reset_protocol, scenarios=scenarios)
    (args.output/'panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    calibration = dict(version='initial-public-ball-phase-calibration-v1',
                       source_training_launch_sha256=hashlib.sha256(launch_path.read_bytes()).hexdigest(),
                       source_training_launch_index=37,
                       intercept_x_m=float(point[0]), reference_intercept_y_m=float(point[1]),
                       reference_local_arrival_s=reference['arrival_time_s']-teacher['walk_seconds'],
                       minimum_prepare_s=3., maximum_prepare_s=4.5,
                       note='A single training calibration. No evaluation arrival or target labels are read by the controller.')
    (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
    print(json.dumps(dict(panel=str(args.output/'panel.json'), count=args.count,
                         sha256=hashlib.sha256((args.output/'panel.json').read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
