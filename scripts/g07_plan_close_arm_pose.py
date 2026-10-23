#!/usr/bin/env python3
"""Conditional CPU arm IK with racket/net separation, never trajectory proof."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.piper_geometry import PiperGeometry


def separating_gap(center, rotation, half_extents):
    """Largest signed gap over the 15 OBB/AABB separating axes."""
    world = np.eye(3)
    axes = [*world, *rotation.T]
    axes.extend(np.cross(a, b) for a in world for b in rotation.T)
    delta = center - np.array([1.5, 0., .762])
    gaps = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        axis = axis / length
        gaps.append(abs(delta @ axis) - abs(rotation.T @ axis) @ half_extents
                    - abs(axis) @ np.array([.01, 1., .762]))
    return float(max(gaps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root-x', type=float, default=1.1)
    parser.add_argument('--height', type=float, default=.75)
    parser.add_argument('--pitch', type=float, default=0.)
    parser.add_argument('--roll', type=float, default=0.)
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--strike-x', type=float, default=1.4)
    parser.add_argument('--strike-z', type=float, default=1.65)
    parser.add_argument('--normal-z', type=float, default=.5)
    parser.add_argument('--uv-limit', type=float, default=.095)
    parser.add_argument('--starts', type=int, default=12)
    parser.add_argument('--seed', type=int, default=716201)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = PiperGeometry()
    target, desired_normal = np.array([args.strike_x, 0., args.strike_z]), np.array([np.sqrt(1-args.normal_z**2), 0., args.normal_z])
    base, rpy = [args.root_x, 0., args.height], [args.roll, args.pitch, args.yaw]
    lower = np.r_[geometry.bounds[:, 0], [-args.uv_limit, -args.uv_limit]]
    upper = np.r_[geometry.bounds[:, 1], [args.uv_limit, args.uv_limit]]
    seed = np.r_[[0., 1.53-args.pitch, -2.58, 0., -.96, 0.], [0., 0.]]
    rng = np.random.default_rng(args.seed)

    def residual(x):
        point, normal = geometry.contact_center(x[:6], x[6:], base_position=base, base_rpy=rpy)
        center, rotation = geometry.forward(x[:6], base_position=base, base_rpy=rpy)
        gap = separating_gap(center, rotation, geometry.half_extents)
        return np.r_[20*(point-target), 2*(normal-desired_normal),
                     20*max(0., .035-gap), .001*x[6:]]

    results = []
    for index in range(args.starts):
        start = seed if index == 0 else seed + rng.normal(0., .4, 8)
        fit = least_squares(residual, np.clip(start, lower+1e-7, upper-1e-7),
                            bounds=(lower, upper), max_nfev=350)
        point, normal = geometry.contact_center(fit.x[:6], fit.x[6:], base_position=base, base_rpy=rpy)
        center, rotation = geometry.forward(fit.x[:6], base_position=base, base_rpy=rpy)
        results.append(dict(arm_pose=fit.x[:6].tolist(), uv_m=fit.x[6:].tolist(),
                            sphere_center_m=point.tolist(), normal=normal.tolist(),
                            racket_center_m=center.tolist(),
                            racket_net_axis_separation_m=separating_gap(center, rotation, geometry.half_extents),
                            error_m=float(np.linalg.norm(point-target)),
                            normal_error=float(np.linalg.norm(normal-desired_normal)),
                            score=float(np.linalg.norm(residual(fit.x)))))
    best = min(results, key=lambda row: row['score'])
    report = dict(kind='Conditional CPU IK only; no dynamics, self-collision, whole-arm/net or tracking guarantee',
                  root_position_m=base, root_rpy=rpy, seed=args.seed,
                  target_m=target.tolist(), target_normal=desired_normal.tolist(),
                  best=best, results=results)
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    torch.save(dict(arm_pose=torch.tensor(best['arm_pose'], dtype=torch.float32), result=report),
               args.output/'pose_seed.pt')
    print(json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
