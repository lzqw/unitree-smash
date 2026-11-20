#!/usr/bin/env python3
"""Conditional impulse error at measured onset, stratified by native contact labels."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from calibrate_supermimic_pilot_oracle import BallImpulsePriorV1, _quat_rotate_inverse, surface_velocity, sliding_speed_change


def summary(entries):
    predicted, native, pn, nn, pt, nt = torch.tensor(entries, dtype=torch.float64).unbind(1)
    nonzero = native > 1e-10
    def median(x):
        return float(x.quantile(.5)) if x.numel() else None
    return dict(count=len(entries), measured_impulse_median_ns=median(native),
                predicted_impulse_median_ns=median(predicted),
                amplitude_ratio_median=median(predicted[nonzero]/native[nonzero]),
                absolute_relative_magnitude_error_median=median((predicted[nonzero]-native[nonzero]).abs()/native[nonzero]),
                normal_vector_error_mean_ns=float(pn.mean()), tangential_vector_error_mean_ns=float(pt.mean()),
                measured_normal_magnitude_median_ns=median(nn), measured_tangent_magnitude_median_ns=median(nt))


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'calibration', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--initial-slip', action='store_true')
    args = cli.parse_args()
    pilot = torch.load(args.pilot, weights_only=True, map_location='cpu')
    fit = json.loads(args.calibration.read_text())
    if pilot['split']['reference_pilot_sha256'] != fit['pilot_sha256']:
        raise ValueError('Model and revised held-out families do not match')
    rows = [r for r in pilot['rows'] if r['contact_object'] == 'racket']
    s = [r['toi_pre_substep'] for r in rows]
    def tensor(key):
        return torch.tensor([item[key] for item in s])[None]
    prior = BallImpulsePriorV1().eval()
    ball = tensor('ball_state_w')
    quat = tensor('racket_quaternion_wxyz')
    local = _quat_rotate_inverse(quat, ball[..., :3]-tensor('racket_position_w'))
    state = dict(local=local, q=quat, linear=tensor('racket_linear_velocity_w'),
                 angular=tensor('racket_angular_velocity_w'), v=ball[..., 7:10])
    lane = torch.arange(len(rows))
    _, n, _, vn, tangent = surface_velocity(state, prior, lane*0, lane)
    mass = .0052
    scale = fit['fitted']['normal_impulse_scale']
    normal = -mass*(1+prior.restitution)*scale*vn.clamp(max=0)
    pn = normal[:, None]*n
    jt = torch.minimum((2/7)*mass*tangent.norm(dim=-1), normal*fit['fitted']['friction'])
    if args.initial_slip:
        slide = sliding_speed_change(prior.preview.racket_gap(local[0]), vn, tangent,
                                     prior.sub_dt, fit['fitted']['friction'])
        jt = torch.maximum(jt, torch.where(mass*slide/prior.sub_dt > .001, mass*slide, 0.))
    pt = -jt[:, None]*F.normalize(tangent, dim=-1)
    nn = torch.tensor([r['pair_normal_impulse_w_ns'] for r in rows])
    nt = torch.tensor([r['pair_friction_impulse_w_ns'] for r in rows])
    output = torch.stack(((pn+pt).norm(dim=-1), (nn+nt).norm(dim=-1),
                          (pn-nn).norm(dim=-1), nn.norm(dim=-1), (pt-nt).norm(dim=-1), nt.norm(dim=-1)), 1)
    buckets = defaultdict(list)
    for row, entry in zip(rows, output.tolist()):
        split = row['split']
        buckets[split].append(entry)
        if split == 'test':
            tags = [row['contact_region'], 'part/'+str(row['racket_part']),
                    'normal_present' if row['native_normal_force_event'] else 'friction_only',
                    'first' if row['contact_order'] == 1 else 'later']
            tags.extend(b for b in row['bucket'].split(':') if b.startswith(('vn', 'vt', 'phase')))
            for tag in tags:
                buckets['test/'+tag].append(entry)
    report = dict(scope='Conditional impulse at measured onset. Truth integrates the complete response interval; tangential labels include native friction-only intervals.',
                  pilot_sha256=hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
                  calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                  initial_slip=args.initial_slip, stats={key:summary(values) for key, values in sorted(buckets.items())})
    with args.output.open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
