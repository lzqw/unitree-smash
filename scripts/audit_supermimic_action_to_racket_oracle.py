#!/usr/bin/env python3
"""Compare native and Body-WM action-to-racket responses on a paired probe.

This is an offline diagnostic.  It never changes a policy, emits a receipt, or
uses native labels as model inputs.  The native endpoint comes from the saved
PhysX probe branches.  The model endpoint comes from the same initial physical
state and the recorded final joint targets passed through PhysicalWorldV2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state


def _load(path: Path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _vector_metrics(predicted: torch.Tensor, native: torch.Tensor):
    pred_norm = predicted.norm(dim=-1)
    native_norm = native.norm(dim=-1)
    cosine = (predicted * native).sum(-1) / (
        pred_norm * native_norm).clamp_min(1e-12)
    ratio = pred_norm / native_norm.clamp_min(1e-12)
    return dict(
        predicted_norm=pred_norm.tolist(),
        native_norm=native_norm.tolist(),
        gain_ratio=ratio.tolist(),
        cosine=cosine.tolist(),
        median_gain_ratio=float(ratio.median()),
        median_cosine=float(cosine.median()),
        p10_cosine=float(torch.quantile(cosine, .1)),
    )


def _normal(value: torch.Tensor):
    rotation = value[:, 3:12].reshape(-1, 3, 3)
    return torch.nn.functional.normalize(rotation[:, :, 0], dim=-1)


def _racket_metrics(predicted, native):
    result = {}
    for name, sl in (('position', slice(0, 3)),
                     ('linear_velocity', slice(12, 15)),
                     ('angular_velocity', slice(15, 18))):
        result[name] = _vector_metrics(
            predicted[:, sl] - predicted.new_zeros(predicted[:, sl].shape),
            native[:, sl] - native.new_zeros(native[:, sl].shape))
    result['normal'] = _vector_metrics(_normal(predicted), _normal(native))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical-checkpoint', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True)
    parser.add_argument('--minus-run', type=Path, required=True)
    parser.add_argument('--plus-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    source = _load(args.physical_checkpoint)
    region = _load(args.region)
    minus = _load(args.minus_run)['raw']
    plus = _load(args.plus_run)['raw']
    lanes = [int(v) for v in region['lane_ids']]
    pairs = region['pairs']
    contract = source['source_manifest']['task_contract']
    model = PhysicalWorldV2(contract['asset']['sha256']).to(device)
    model.load_state_dict(source['model'], strict=True)
    model.eval().requires_grad_(False)
    state = pairs['state'].to(device)
    previous = pairs['previous_action'].to(device)
    feet = pairs['feet'].to(device)
    present = pairs['present'].to(device)
    elapsed = pairs['elapsed'].to(device)
    launch = pairs['launch'].to(device)

    predictions = {}
    native_states = {}
    for branch, run, target_key in (
            ('minus', minus, 'targets_minus'),
            ('plus', plus, 'targets_plus')):
        predicted, _, _ = model(
            state, pairs[target_key].to(device), previous, feet, present,
            elapsed, launch)
        predictions[branch] = racket_state(predicted, model.geometry)
        native = run['next_state'][lanes].to(device)
        native_states[branch] = racket_state(native, model.geometry)

    model_delta = predictions['plus'] - predictions['minus']
    native_delta = native_states['plus'] - native_states['minus']
    metrics = {}
    for name, sl in (('position', slice(0, 3)),
                     ('linear_velocity', slice(12, 15)),
                     ('angular_velocity', slice(15, 18))):
        metrics[name] = _vector_metrics(model_delta[:, sl], native_delta[:, sl])
    metrics['normal'] = _vector_metrics(
        _normal(predictions['plus']) - _normal(predictions['minus']),
        _normal(native_states['plus']) - _normal(native_states['minus']))

    target_diff = plus['final_joint_targets'][lanes] - minus['final_joint_targets'][lanes]
    final_diff = plus['final_normalized'][lanes] - minus['final_normalized'][lanes]
    request_diff = plus['request_after'][lanes] - minus['request_after'][lanes]
    result = dict(
        schema='supermimic-action-to-racket-oracle-v1',
        lane_ids=lanes,
        action_response=dict(
            request_mean_abs=float(request_diff.abs().mean()),
            final_action_mean_abs=float(final_diff.abs().mean()),
            final_target_mean_abs=float(target_diff.abs().mean()),
            final_target_max_abs=float(target_diff.abs().max()),
        ),
        metrics=metrics,
        samples=[dict(
            lane=lane,
            model_position_delta=model_delta[i, :3].detach().cpu().tolist(),
            native_position_delta=native_delta[i, :3].detach().cpu().tolist(),
            model_velocity_delta=model_delta[i, 12:15].detach().cpu().tolist(),
            native_velocity_delta=native_delta[i, 12:15].detach().cpu().tolist(),
            model_angular_delta=model_delta[i, 15:18].detach().cpu().tolist(),
            native_angular_delta=native_delta[i, 15:18].detach().cpu().tolist(),
        ) for i, lane in enumerate(lanes)],
        receipt_status='diagnostic_only_not_a_contact_receipt',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(
        output=str(args.output),
        position=metrics['position'],
        linear_velocity=metrics['linear_velocity'],
        angular_velocity=metrics['angular_velocity'],
        normal=metrics['normal'],
    ), indent=2))


if __name__ == '__main__':
    main()
