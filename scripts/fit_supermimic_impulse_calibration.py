#!/usr/bin/env python3
"""Fit the two-parameter analytic impulse calibration on native paired probes.

The contact-boundary bias is searched on a declared grid because the forward
contact event is discrete. The normal impulse scale is searched jointly on the
same bounded grid. Final native holdout samples are evaluated once after the
development choice and never participate in selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_ball_v1 import action_ball_from_checkpoint
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import (
    BallImpulseCalibrationV1, attach_hybrid_impulse_prior, hybrid_ball_manifest)
from fit_supermimic_native_probe_response import (  # type: ignore
    concat_pairs, load_rows, metrics, response, validate_probe_identity_reuse)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-checkpoint', type=Path, required=True)
    parser.add_argument('--ball-checkpoint', type=Path, required=True)
    parser.add_argument('--action-ball-checkpoint', type=Path, required=True)
    parser.add_argument('--train-bundle', type=Path, nargs='+', required=True)
    parser.add_argument('--validation-bundle', type=Path, nargs='+', required=True)
    parser.add_argument('--final-holdout-bundle', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gap-min-m', type=float, default=-0.004)
    parser.add_argument('--gap-max-m', type=float, default=0.004)
    parser.add_argument('--gap-count', type=int, default=9)
    parser.add_argument('--scale-min', type=float, default=0.75)
    parser.add_argument('--scale-max', type=float, default=1.25)
    parser.add_argument('--scale-count', type=int, default=11)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.gap_count < 1 or args.scale_count < 1 or args.gap_min_m > args.gap_max_m:
        raise ValueError('Invalid calibration grid')
    if args.scale_min <= 0 or args.scale_min > args.scale_max:
        raise ValueError('Invalid impulse-scale grid')
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        raise RuntimeError('GPU required for impulse calibration')
    device = torch.device(args.device)
    train_rows = load_rows(args.train_bundle, device)
    validation_rows = load_rows(args.validation_bundle, device)
    final_rows = load_rows(args.final_holdout_bundle, device) if args.final_holdout_bundle else []
    train_ids = [sid for row in train_rows for sid in row['sample_ids']]
    validation_ids = [sid for row in validation_rows for sid in row['sample_ids']]
    final_ids = [sid for row in final_rows for sid in row['sample_ids']]
    all_ids = train_ids + validation_ids + final_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Impulse calibration splits contain duplicate sample identities')
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(final_ids) or set(validation_ids) & set(final_ids):
        raise ValueError('Impulse calibration splits overlap')
    identity = validate_probe_identity_reuse((*train_rows, *validation_rows, *final_rows))
    train = concat_pairs(train_rows, device)
    validation = concat_pairs(validation_rows, device)
    final = concat_pairs(final_rows, device) if final_rows else None
    physical_source = torch.load(args.physical_checkpoint, map_location='cpu', weights_only=True)
    ball_source = torch.load(args.ball_checkpoint, map_location='cpu', weights_only=True)
    candidate_source = torch.load(args.action_ball_checkpoint, map_location='cpu', weights_only=True)
    if candidate_source.get('physical_sha256') != sha(args.physical_checkpoint):
        raise ValueError('Action candidate physical checkpoint mismatch')
    if candidate_source.get('legacy_ball_sha256') != sha(args.ball_checkpoint):
        raise ValueError('Action candidate legacy ball checkpoint mismatch')
    from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
    from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
    physical = PhysicalWorldV2(
        physical_source['source_manifest']['task_contract']['asset']['sha256']).to(device)
    physical.load_state_dict(physical_source['model'], strict=True)
    ball = BallResidualV2().to(device)
    ball.load_state_dict(ball_source['model'], strict=True)
    physical.eval().requires_grad_(False)
    ball.eval().requires_grad_(False)

    def evaluate(gap_bias, scale, pairs):
        model = action_ball_from_checkpoint(candidate_source).to(device)
        model.load_state_dict(candidate_source['model'], strict=True)
        attach_hybrid_impulse_prior(
            model, device,
            calibration={'gap_bias_m': float(gap_bias),
                         'normal_impulse_scale': float(scale)})
        model.eval().requires_grad_(False)
        with torch.no_grad():
            predicted = response(model, physical, ball, pairs)
            value = metrics(predicted, pairs)
        return value

    best = None
    history = []
    for gap in torch.linspace(args.gap_min_m, args.gap_max_m, args.gap_count).tolist():
        for scale in torch.linspace(args.scale_min, args.scale_max, args.scale_count).tolist():
            value = evaluate(gap, scale, validation)
            row = dict(gap_bias_m=float(gap), normal_impulse_scale=float(scale),
                       validation=value)
            history.append(row)
            key = (value['response_rmse'], value['relative_error_max'],
                   -value['response_cosine_min'])
            if best is None or key < best[0]:
                best = (key, row)
    if best is None:
        raise RuntimeError('Calibration grid produced no candidate')
    gap = best[1]['gap_bias_m']; scale = best[1]['normal_impulse_scale']
    selected = action_ball_from_checkpoint(candidate_source).to(device)
    selected.load_state_dict(candidate_source['model'], strict=True)
    attach_hybrid_impulse_prior(selected, device,
                                calibration={'gap_bias_m': gap,
                                             'normal_impulse_scale': scale})
    selected.eval().requires_grad_(False)
    with torch.no_grad():
        validation_final = metrics(response(selected, physical, ball, validation), validation)
        final_metrics = None if final is None else metrics(response(selected, physical, ball, final), final)
    output = dict(candidate_source)
    output['model'] = {key: value.detach().cpu() for key, value in candidate_source['model'].items()}
    output['hybrid_ball_transition'] = hybrid_ball_manifest(
        gap_bias_m=gap, normal_impulse_scale=scale)
    output['impulse_calibration'] = dict(
        schema=BallImpulseCalibrationV1.schema,
        gap_bias_m=gap,
        normal_impulse_scale=scale,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        final_holdout_sample_ids=final_ids,
        selection_rule='development response_rmse, then relative_error_max, then cosine floor',
        native_identity=identity,
        final_holdout_not_used_for_selection=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    result = dict(schema='ball-impulse-calibration-result-v1', output=str(args.output),
                  output_sha256=sha(args.output), selected=output['impulse_calibration'],
                  validation=validation_final, final_holdout=final_metrics,
                  grid_count=len(history), history=history)
    args.output.with_name('calibration_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
