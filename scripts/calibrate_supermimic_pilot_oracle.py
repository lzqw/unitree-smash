#!/usr/bin/env python3
"""Calibrate first racket impact against native held-out short contact trials."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate, _quat_rotate_inverse


def first(mask):
    missing = mask.shape[0]
    return torch.where(mask.any(0), mask.int().argmax(0), missing)


def quantile(values, q):
    return float(torch.quantile(values.float(), q)) if values.numel() else None


def trajectory(archive, prior):
    trace = archive['raw']['substep_trace']
    t = trace['tensors']
    state = t['shuttle_state_w']
    p, v = prior.advance_trajectory(state[0, :, :3], state[0, :, 7:10])
    p, v = p.transpose(0, 1), v.transpose(0, 1)
    if p.shape != state[..., :3].shape or abs(trace['physics_dt_s'] - prior.sub_dt) > 1e-12:
        raise ValueError('Incompatible native clock or measurement window')
    center, q = t['racket_position_w'], t['racket_quaternion_w']
    local = _quat_rotate_inverse(q, p - center)
    gap = prior.preview.racket_gap(local)
    touch = t['racket_force_w'].norm(dim=-1) > .001
    true_index = first(touch)
    lane = torch.arange(state.shape[1])
    limit = state.shape[0]
    idx = torch.arange(limit)[:, None]
    end = (first((idx >= true_index) & ~touch) - 1).clamp(0, limit - 1)
    return dict(gap=gap, local=local, q=q, v=v, center=center,
                linear=t['racket_linear_velocity_w'], angular=t['racket_angular_velocity_w'],
                truth=true_index, truth_out=state[end, lane, 7:10], end=end,
                multi=((touch[1:] & ~touch[:-1]).sum(0) > 1))


def predict(t, prior, bias):
    gap = t['gap']
    # Start-of-window overlaps are left-censored. An approaching surface
    # can still generate a first impulse at the next solver step.
    crossing = (gap[1:] <= bias) & (gap[1:] < gap[:-1])
    predicted = first(crossing) + 1
    lane = torch.arange(gap.shape[1])
    index = predicted.clamp(max=len(gap) - 1)
    local = t['local'][index, lane]
    nlocal, part = prior.racket_surface(local)
    q = t['q'][index, lane]
    n = _quat_rotate(q, nlocal)
    arm = _quat_rotate(q, local - prior.preview.radius * nlocal)
    surface = t['linear'][index, lane] + torch.linalg.cross(t['angular'][index, lane], arm)
    incoming = t['v'][index, lane]
    rel = incoming - surface
    vn = (rel * n).sum(-1)
    tangent = rel - vn[:, None] * n
    normal_dv = -(1 + prior.restitution) * vn.clamp(max=0)
    return dict(pred=predicted, truth=t['truth'], cap=torch.full_like(index, len(gap)), vn=vn,
                n=n, normal_dv=normal_dv, tangent=tangent, incoming=incoming,
                truth_out=t['truth_out'], end=t['end'], part=part, multi=t['multi'])


def join(group):
    return {key: torch.cat([row[key] for row in group]) for key in group[0]}


def hits(d):
    return (d['pred'] < d['cap']) & (d['truth'] < d['cap']) & (d['end'] < d['cap'] - 1)


def score(d, scale, friction):
    pos, pred = d['truth'] < d['cap'], d['pred'] < d['cap']
    mask = hits(d)
    dvn = d['normal_dv'] * scale
    speed = d['tangent'].norm(dim=-1)
    dvt = torch.minimum(2/7 * speed, friction * dvn)
    v = d['incoming'] + dvn[:, None] * d['n'] - dvt[:, None] * F.normalize(d['tangent'], dim=-1)
    v, measured = v[mask], d['truth_out'][mask]
    direction = F.cosine_similarity(v, measured)
    ratio = v.norm(dim=-1) / measured.norm(dim=-1).clamp_min(1e-8)
    delta = (d['truth'] - d['pred']).abs()[mask]
    return dict(lanes=len(pos), true_impacts=int(pos.sum()), true_negatives=int((~pos).sum()),
                missed=int((pos & ~pred).sum()), false_positives=int((pred & ~pos).sum()),
                fpr=float((pred & ~pos).sum() / (~pos).sum().clamp_min(1)),
                matched_uncensored=len(delta), native_repeated_lanes=int(d['multi'].sum()),
                toi_median=quantile(delta,.5), toi_p95=quantile(delta,.95),
                direction_median=quantile(direction,.5), direction_p10=quantile(direction,.1),
                amplitude_ratio_median=quantile(ratio,.5), amplitude_ratio_p10=quantile(ratio,.1),
                amplitude_ratio_p90=quantile(ratio,.9),
                outgoing_velocity_rmse_mps=float((v-measured).square().sum(-1).mean().sqrt()),
                normal_outgoing_sign_errors=int((((v*d['n'][mask]).sum(-1) *
                                                  (measured*d['n'][mask]).sum(-1)) < 0).sum()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    pilot = torch.load(args.pilot, map_location='cpu', weights_only=True)
    source_split = {}
    source_hash = {}
    for row in pilot['rows']:
        if source_hash.setdefault(row['source'],row['source_sha256']) != row['source_sha256']:
            raise ValueError('Source identity changed inside pilot')
        if source_split.setdefault(row['source'],row['split']) != row['split']:
            raise ValueError('Source overlaps splits')
    prior = BallImpulsePriorV1().eval()
    by_split = {key: [] for key in ('train','validation','test')}
    for path, split in sorted(source_split.items()):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source_hash[path]:
            raise ValueError(f'Source archive hash mismatch: {path}')
        by_split[split].append(trajectory(torch.load(path,map_location='cpu',weights_only=True), prior))
    if not all(by_split.values()) or not pilot['split']['unseen_spec_holdout']:
        raise ValueError('Independent holdout missing')
    def outputs(split, bias):
        return join([predict(trial, prior, bias) for trial in by_split[split]])
    def timing_loss(trial, bias):
        gap = trial['gap']
        pred = first((gap[1:] <= bias) & (gap[1:] < gap[:-1])) + 1
        return (pred - trial['truth']).abs().clamp(max=16).sum()
    offsets = (-.0004,-.0002,0.,.0001,.0002,.0004,.0008)
    bias = min(offsets, key=lambda b: sum(float(timing_loss(t,b)) for t in by_split['train']))
    train = outputs('train',bias)
    mask = hits(train) & (train['normal_dv'] > .5)
    projection = ((train['truth_out']-train['incoming']) * train['n']).sum(-1)
    scale = float((projection[mask] / train['normal_dv'][mask]).median().clamp(.5,1.25))
    friction = min((.04,.06,.08,.10,.12,.16), key=lambda f: score(train,scale,f)['outgoing_velocity_rmse_mps'])
    report = dict(scope='First-impact native racket teacher forcing over one 160-substep trial. Timing errors cover matched uncensored trials; misses and false positives are separate.',
                  pilot_sha256=hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
                  fitted=dict(gap_contact_threshold_m=bias,normal_impulse_scale=scale,friction=friction),
                  effective_mass_kg_at_fixed_restitution=.0052*scale,
                  rebound_restitution_assumption=prior.restitution, metrics={},
                  policy_training_admitted=False, contact_bptt='detached', v13_residual_gradient='detached')
    for split in by_split:
        report['metrics'][split] = dict(baseline=score(outputs(split,0.),1.,.08),
                                        calibrated=score(outputs(split,bias),scale,friction))
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['metrics']['test'],indent=2))


if __name__ == '__main__':
    main()
