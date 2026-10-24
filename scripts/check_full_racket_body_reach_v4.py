#!/usr/bin/env python3
"""Conditional offline reach tests at a recorded TRAIN body pose.

Virtual base translations below are planning variables, never native states
or a proposal to teleport the robot. Every candidate needs ground-start
whole-body dynamics and the original complete-smash evaluation afterwards.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    source = json.loads(args.snapshot.read_text())
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = FullRacketGeometryV4()
    root = np.asarray(source['body_link_poses'][0])
    rotation = Rotation.from_quat(root[[4, 5, 6, 3]])
    rpy = rotation.as_euler('xyz')
    target = np.asarray(source['ball_position_m'])
    seed = np.asarray(checkpoint['seed']['joint_position_rad'])
    reference = seed + checkpoint['parameters'][8:14].numpy()
    lower = np.maximum(model.bounds[:, 0], seed-.25) + 1e-5
    upper = np.minimum(model.bounds[:, 1], seed+.25) - 1e-5
    normal = np.asarray(checkpoint['seed']['outward_contact_normal_world'])
    normal /= np.linalg.norm(normal)
    rng = np.random.default_rng(726408)
    starts = [np.clip(reference, lower, upper), *rng.uniform(lower, upper, (3, 6))]
    rows = []
    for uv_z in [0., .11]:
        assert uv_z <= model.half_extents[2]
        for dx in [0., .10, .20, .30]:
            for dz in [0., .05, .10, .15]:
                position = root[:3] + [dx, 0., dz]
                def residual(q):
                    point, n = model.contact_center(q, uv=(0., uv_z), base_position=position, base_rpy=rpy)
                    return np.r_[5*(point-target), n-normal, .001*(q-reference)]
                solutions = []
                for start in starts:
                    fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=300)
                    point, n = model.contact_center(fit.x, uv=(0., uv_z), base_position=position, base_rpy=rpy)
                    error, angle = np.linalg.norm(point-target), np.arccos(np.clip(n@normal, -1, 1))
                    solutions.append((error+.2*angle, error, angle, fit.x))
                _, error, angle, q = min(solutions, key=lambda x: x[0])
                rows.append(dict(virtual_base_dx_m=dx, virtual_base_dz_m=dz, string_uv_z_m=uv_z,
                                 position_error_m=float(error), normal_error_deg=float(np.degrees(angle)),
                                 joint_position_rad=q.tolist(), native_execution=False))
    arm_base = root[:3] + rotation.apply(model.arm_origin)
    center_tip = model.tool_offset + model.tool_rotation@np.array([.0256, 0., 0.])
    center_bound = sum(np.linalg.norm(p) for p, _ in model.joints) + np.linalg.norm(center_tip)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='Offline conditional reach of center/upper strings at one actual TRAIN base pose and hypothetical base translations. No translated state is executed.',
                  source_run=source['source_run'], source_env_id=source['env_id'], source_clock_s=source['recorded_clock_s'],
                  recorded_root_position_m=root[:3].tolist(), recorded_ball_position_m=target.tolist(),
                  distance_from_recorded_arm_base_m=float(np.linalg.norm(target-arm_base)),
                  triangle_upper_bound_for_center_point_m=float(center_bound),
                  bound_scope='Center-point-only necessary condition; upper strings can extend farther. Not a global impossibility claim.',
                  original_physical_joint_limits_unchanged=True, original_pose_search_offsets_rad=.25,
                  rows=rows, native_execution=False, formal_acceptance=False,
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [args.snapshot, args.checkpoint, Path(__file__)]})
    args.output.mkdir(parents=True)
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([r for r in rows if r['position_error_m'] < .005 and r['normal_error_deg'] < 2]), flush=True)


if __name__ == '__main__':
    main()
