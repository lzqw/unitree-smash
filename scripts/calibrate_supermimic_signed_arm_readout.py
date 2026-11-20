#!/usr/bin/env python3
"""Fit the existing arm position and velocity readout to natural motion and paired executed targets."""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_training_probe_replay import training_windows


def arm_rows(model, data, ticks, columns, launches):
    state = data['state'][ticks, columns]
    features = model.features(state, *(data[k][ticks, columns] for k in
                             ('targets', 'previous_action', 'feet', 'present', 'elapsed')), launches)
    hidden = model.body.net[:-1]((features - model.input_mean) / model.input_scale)
    layer = model.body.net[-1]
    # The physical ordering is leg12, arm6, for both q and qdot.
    arm = torch.arange(25, 31)
    rows = torch.cat((arm, arm + 18))
    velocity = state[:, arm + 18]
    prior = torch.cat((state[:, arm] + model.dt * velocity, velocity), dim=1)
    scale = model.delta_scale[rows]
    native = (data['next_state'][ticks, columns][:, rows] - prior - model.delta_mean[rows]) / scale
    error = native - layer(hidden)[:, rows]
    return torch.cat((hidden, torch.ones_like(hidden[:, :1])), dim=1), error, native


class Moments:
    def __init__(self, width, outputs):
        self.gram = torch.zeros(width, width, dtype=torch.float64)
        self.cross = torch.zeros(width, outputs, dtype=torch.float64)
        self.energy = torch.zeros(outputs, dtype=torch.float64)
        self.sse = torch.zeros(outputs, dtype=torch.float64)
        self.n = 0

    def add(self, x, errors, targets):
        x, e, y = (v.double() for v in (x, errors, targets))
        self.gram.addmm_(x.T, x)
        self.cross.addmm_(x.T, e)
        self.energy += y.square().sum(0)
        self.sse += e.square().sum(0)
        self.n += len(x)

    def rmse(self, update):
        sse = self.sse - 2 * (self.cross * update).sum(0) + (update * (self.gram @ update)).sum(0)
        return (sse.clamp_min(0) / self.n).sqrt()


def solve(natural, signed, ridge):
    # Normalize by the calibration-only per-channel signal power. A 1% RMS floor
    # keeps a nearly unexcited paired output from overwhelming natural motion.
    a = .75 / natural.n
    nat_energy = (natural.energy / natural.n).clamp_min(1.e-12)
    sig_energy = (signed.energy / signed.n).clamp_min(1.e-4 * nat_energy)
    b = .25 / signed.n * (nat_energy / sig_energy)
    lhs = a * natural.gram + ridge * torch.eye(natural.gram.shape[0], dtype=natural.gram.dtype)
    return torch.stack([torch.linalg.solve(lhs + w * signed.gram, a * n + w * s)
                        for w, n, s in zip(b, natural.cross.T, signed.cross.T, strict=True)], dim=1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('body', 'packed', 'training-collection', 'output'):
        parser.add_argument('--' + k, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(args.body, map_location='cpu', weights_only=True)
    meta = ck['calibration']
    task = ck['source_manifest']['task_contract']
    packed = torch.load(args.packed, weights_only=True, mmap=True)
    if digest(args.packed) != meta['packed_sha256'] or packed['source_manifest']['task_contract'] != task:
        raise ValueError('Different natural archive or physics contract')
    suite = Path(task['suite']['path'])
    if digest(suite) != task['suite']['sha256']:
        raise ValueError('Different launch suite')
    launch = torch.tensor([c['launch']['position_m'] + c['launch']['velocity_mps']
                           for c in json.loads(suite.read_text())['cases']])
    lanes = set(meta['train_lanes'])
    if lanes & set(meta['excluded_lanes']):
        raise ValueError('Training lanes intersect the holdout')
    windows, receipt = training_windows(args.training_collection, task, contract(), set(range(len(launch))) - lanes)
    if receipt not in meta.get('sequence_receipts', [meta.get('sequence_receipt')]):
        raise ValueError('Paired data differ from the parent calibration')
    model = PhysicalWorldV2(task['asset']['sha256']).eval()
    model.load_state_dict(ck['model'])
    layer = model.body.net[-1]
    rows = list(range(25, 31)) + list(range(43, 49))
    natural, signed = [Moments(layer.in_features + 1, len(rows)) for _ in range(2)]
    data = packed['tensors']
    eligible = torch.zeros(len(launch), dtype=torch.bool)
    eligible[sorted(lanes)] = True
    ticks, cols = torch.where(eligible & ~(data['terminated'] | data['timeout']))
    for t, c in zip(ticks.split(512), cols.split(512), strict=True):
        natural.add(*arm_rows(model, data, t, c, launch[c]))
    for ids, branches in windows:
        minus, plus = (branches[key] for key in ('minus', 'plus'))
        t, c = torch.where(torch.ones_like(minus['terminated'], dtype=torch.bool))
        for times, columns in zip(t.split(512), c.split(512), strict=True):
            launches = launch[torch.tensor(ids)[columns]]
            m, p = (arm_rows(model, d, times, columns, launches) for d in (minus, plus))
            signed.add(*(y - x for y, x in zip(p, m, strict=True)))
    ridge = .1
    step = solve(natural, signed, ridge)
    if not torch.isfinite(step).all():
        raise ValueError('Nonfinite readout')
    # Integer-list indexing returns a copy; assign explicitly and verify after serialization.
    layer.weight[rows] += step[:-1].T.to(layer.weight)
    layer.bias[rows] += step[-1].to(layer.bias)
    info = dict(parent_sha256=digest(args.body), natural_rows=natural.n, paired_rows=signed.n,
                paired_weight=.25, ridge=ridge, readout_rows=rows, training_collection=receipt,
                physical_units=['rad']*6+['rad/s']*6, actor_updates=0)
    for key, stats in [('natural', natural), ('paired', signed)]:
        scale = model.delta_scale[rows]
        info[key + '_rmse_before'] = (stats.rmse(torch.zeros_like(step)) * scale).tolist()
        info[key + '_rmse_after'] = (stats.rmse(step) * scale).tolist()
    output = args.output
    output.mkdir(exist_ok=False, parents=True)
    ck.update(model=model.state_dict(), calibration=dict(meta, signed_arm_readout=info), policy_training_admitted=False)
    pending = output / 'world.pending'
    torch.save(ck, pending)
    restored = PhysicalWorldV2(task['asset']['sha256']).eval()
    restored.load_state_dict(torch.load(pending, weights_only=True)['model'])
    torch.testing.assert_close(restored.body.net[-1].weight, layer.weight, rtol=0, atol=0)
    pending.replace(output / 'world.pt')
    (output / 'fit.json').write_text(json.dumps(dict(info, sha256=digest(output/'world.pt')), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
