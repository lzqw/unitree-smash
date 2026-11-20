#!/usr/bin/env python3
"""Fit a causal substep racket-motion predictor on the registered unloaded calibration families."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_supermimic_contact_rollout import sha
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse, _quat_rotate


RACKET_FIELDS = ('racket_position_w', 'racket_quaternion_w', 'racket_linear_velocity_w',
                 'racket_angular_velocity_w')


def multiply_quaternions(a, b):
    wa, va = a[..., :1], a[..., 1:]
    wb, vb = b[..., :1], b[..., 1:]
    return torch.cat((wa * wb - (va * vb).sum(-1, keepdim=True),
                      wa * vb + wb * va + torch.linalg.cross(va, vb)), dim=-1)


def motion_features(packet, teacher):
    raw = packet['raw']['substep_trace']['tensors']
    q = raw['racket_quaternion_w'][0]
    inp = packet['velocity_intervention']
    native = inp['native_velocity_readback']['after']
    initial_q = teacher['raw']['start']['before']['physical']['joint_position'][packet['parent_lane']]
    # A uniform velocity shift changes world translation only for a freely flying articulation.
    # The shape and twist evolution use velocities relative to the root.
    dv = native['racket_link'] - native['root_com'][:, :3]
    g = torch.zeros_like(dv)
    g[:, 2] = -9.81
    return torch.cat((initial_q, inp['joint_targets'] - initial_q, native['dof'],
                      *[_quat_rotate_inverse(q, v) for v in (dv, native['root_com'][:, 3:], native['racket_angular'], g)]), -1)


def motion_residuals(packet):
    trace = packet['raw']['substep_trace']
    c, q, v, w = [trace['tensors'][k][:-1] for k in RACKET_FIELDS]
    t = torch.arange(len(c))[:, None, None] * trace['physics_dt_s']
    frame = q[:1].expand_as(q)
    inv = torch.cat((frame[..., :1], -frame[..., 1:]), -1)
    rel = multiply_quaternions(inv, q)
    rel = torch.where(rel[..., :1] < 0, -rel, rel)
    return torch.cat((_quat_rotate_inverse(frame, c - c[:1] - t*v[:1]),
                      rel[..., 1:], _quat_rotate_inverse(frame, v-v[:1]),
                      _quat_rotate_inverse(frame, w-w[:1])), -1).transpose(0, 1)


def predict_motion(model, features, start, times):
    c, q, v, w = start
    x = (features.double() - model['mean']) / model['scale']
    residual = (x @ model['weights'] + model['intercept']).float().reshape(len(c), len(times), 12).transpose(0, 1)
    frame = q[None].expand(len(times), -1, -1)
    pos, xyz, vel, omega = residual.split(3, dim=-1)
    rel = torch.cat(((1-xyz.square().sum(-1, keepdim=True)).clamp_min(0).sqrt(), xyz), -1)
    rotation = torch.nn.functional.normalize(multiply_quaternions(frame, rel), dim=-1)
    return [c + times[:, None, None]*v + _quat_rotate(frame,pos), rotation,
            v + _quat_rotate(frame,vel), w + _quat_rotate(frame,omega)]


@torch.no_grad()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('wide','unloaded','transfer','output'):
        cli.add_argument('--'+name, type=Path, required=True)
    a = cli.parse_args()
    torch.set_num_threads(3)
    transfer = json.loads(a.transfer.read_text())
    design = json.loads((a.wide/'design.json').read_text())
    prov = json.loads((a.unloaded/'nominal_a'/'provenance.json').read_text())
    teacher_path = Path(prov['args'][prov['args'].index('--teacher')+1])
    if (transfer['split'] != 'train' or sha(teacher_path) != design['source_teacher_sha256']
            or prov['teacher_sha256'] != design['source_teacher_sha256']):
        raise ValueError('Calibration families and original native teacher required')
    teacher = torch.load(teacher_path,weights_only=True,mmap=True)
    xs, ys, hashes = [], [], {}
    for seed, split in design['assignments'].items():
        if split != 'train':
            continue
        path = a.unloaded/'nominal_a'/f'spec_{seed}'/'trace.pt'
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != transfer['unloaded_hashes'][str(path)]:
            raise ValueError('Unregistered training trajectory')
        packet = torch.load(path, weights_only=True, mmap=True)
        if packet['schema'] != 'native-unloaded-racket-oracle-v1' or packet['seed'] != int(seed):
            raise ValueError('Unloaded native trajectories required')
        xs.append(motion_features(packet,teacher))
        ys.append(motion_residuals(packet).flatten(1))
    x, y = torch.cat(xs).double(), torch.cat(ys).double()
    mean, scale, intercept = x.mean(0), x.std(0).clamp_min(.01), y.mean(0)
    x = (x-mean)/scale
    ridge = .05
    weights = torch.linalg.solve(x.T @ x + len(x)*ridge*torch.eye(x.shape[1],dtype=x.dtype), x.T @ (y-intercept))
    if not weights.isfinite().all():
        raise ArithmeticError('Nonfinite motion fit')
    model = dict(schema='unloaded-articulation-motion-readout-v1', design_sha256=sha(a.wide/'design.json'),
                 teacher_sha256=sha(teacher_path), train_archives=hashes, mean=mean, scale=scale,
                 intercept=intercept, weights=weights, ridge=ridge, calibration_rows=len(x),
                 physics_dt_s=packet['raw']['substep_trace']['physics_dt_s'])
    with a.output.open('xb') as stream:
        torch.save(model, stream)


if __name__ == '__main__':
    main()
