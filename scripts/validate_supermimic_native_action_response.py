#!/usr/bin/env python3
"""Validate a measured native action-response bundle against the loaded WM.

This is the final native direction gate.  It consumes only four-branch
same-prefix PhysX evidence and predeclared holdout bounds.  It never fits a
model, fabricates a response, or changes an Actor checkpoint.  A receipt is
written only when event-aligned metrics and the measured native response both
pass their declared bounds.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_probe_region_v1 import paired_metrics
from quadsmash.rl.supermimic_evidence_binding_v1 import (
    json_sha, native_code_compatibility, runtime_identity,
)
from quadsmash.rl.supermimic_long_run_contract_v1 import verify_contact_receipt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value, name):
    if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
        raise ValueError(f'Nonfinite or missing metric: {name}')
    return float(value)


def _load_bounds(path):
    bounds = json.loads(Path(path).read_text())
    if bounds.get('schema') != 'contact-validation-bounds-v1':
        raise ValueError('Contact validation bounds schema mismatch')
    if bounds.get('thresholds_declared_before_validation') is not True:
        raise ValueError('Bounds must be declared before validation')
    limits = bounds.get('limits')
    if not isinstance(limits, dict) or not limits:
        raise ValueError('Missing native response limits')
    for key, value in limits.items():
        _finite_number(value, f'limits/{key}')
        if value < 0:
            raise ValueError(f'Negative limit: {key}')
    event_limits = bounds.get('event_limits')
    if not isinstance(event_limits, dict) or not event_limits:
        raise ValueError('Missing event-aligned metric limits')
    for key, value in event_limits.items():
        _finite_number(value, f'event_limits/{key}')
        if value < 0:
            raise ValueError(f'Negative event limit: {key}')
    minimum_samples = bounds.get('minimum_paired_samples')
    minimum_episodes = bounds.get('minimum_independent_validation_episodes')
    if type(minimum_samples) is not int or minimum_samples < 1:
        raise ValueError('Declare a positive minimum paired native sample count')
    if type(minimum_episodes) is not int or minimum_episodes < 1:
        raise ValueError('Declare a positive minimum independent validation episode count')
    train = set(bounds.get('training_episode_ids', []))
    valid = set(bounds.get('validation_episode_ids', []))
    if not train or not valid or train & valid:
        raise ValueError('Calibration and validation episodes must be disjoint')
    return bounds


def _event_metrics(path):
    payload = json.loads(Path(path).read_text())
    metrics = payload.get('event_aligned_errors', payload)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError('Missing event-aligned holdout errors')
    result = {}
    for name, value in metrics.items():
        result[str(name)] = _finite_number(value, f'event_aligned_errors/{name}')
    return result


def _response_errors(measured):
    cosine = measured['response_cosine']
    if not cosine:
        raise ValueError('Native response contains no samples')
    values = {
        'native_repeat_max': measured['native_repeat_max'],
        'response_rmse': measured['response_rmse'],
        'response_cosine_error_max': max(0.0, 1.0 - min(cosine)),
        'response_relative_error_max': max(measured['relative_error']),
    }
    return {name: _finite_number(value, name) for name, value in values.items()}


def validate_native_sample_ids(sample_ids, validation_episode_ids):
    """Require every measured probe identity to be declared held-out."""
    values = list(sample_ids)
    allowed = set(validation_episode_ids)
    if not values or not set(values) <= allowed:
        raise ValueError('Native probe samples are outside the declared validation episode set')


def build_receipt(*, bounds, event_errors, response_errors, models, contract,
                  probe_path, event_path, measured, identity):
    limits = bounds['limits']
    event_limits = bounds['event_limits']
    if set(event_errors) != set(event_limits):
        raise ValueError('Event-aligned metrics do not match declared bounds')
    for name, value in event_errors.items():
        limit = event_limits[name]
        if value > float(limit):
            raise ValueError(f'Event-aligned metric exceeds bound: {name}')
    for name, value in response_errors.items():
        limit = limits.get(name)
        if limit is None or value > float(limit):
            raise ValueError(f'Native response metric exceeds bound: {name}')
    sample_ids = measured.get('sample_ids', [])
    if measured.get('count') != len(sample_ids):
        raise ValueError('Native response sample identity count mismatch')
    if measured['count'] < bounds['minimum_paired_samples']:
        raise ValueError('Insufficient paired native response samples')
    if len(set(sample_ids)) < bounds['minimum_independent_validation_episodes']:
        raise ValueError('Insufficient independent native validation episodes')
    source_files = [
        dict(path=str(Path(probe_path).resolve().name), sha256=sha256(probe_path)),
        dict(path=str(Path(event_path).resolve().name), sha256=sha256(event_path)),
    ]
    receipt = dict(
        schema='supermimic-contact-validation-v1',
        model_sha256={key: sha256(path) for key, path in models.items()},
        task_contract_sha256=hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest(),
        training_episode_ids=sorted(bounds['training_episode_ids']),
        validation_episode_ids=sorted(bounds['validation_episode_ids']),
        thresholds_declared_before_validation=True,
        event_aligned_errors=event_errors,
        action_response_errors=response_errors,
        upper_bounds=dict(
            event_aligned_errors=bounds.get('event_limits', {}),
            action_response_errors={key: limits[key] for key in response_errors},
        ),
        native_action_probe=dict(
            same_prefix_verified=measured['native_repeat_max'] <= limits['native_repeat_max'],
            paired_samples=measured['count'],
            independent_episode_count=len(set(sample_ids)),
            response_direction_verified=measured['passed'],
            metrics=measured,
        ),
        runtime_identity=identity,
        evidence_files=source_files,
        scope='measured same-prefix PhysX action response and independent event-aligned holdout; no Actor update',
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'physical-checkpoint', 'ball-checkpoint', 'task-checkpoint', 'safety-mapping'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--probe-bundle', type=Path, required=True)
    parser.add_argument('--event-metrics', type=Path, required=True)
    parser.add_argument('--bounds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action-ball-checkpoint', type=Path)
    parser.add_argument('--g07-execution-root', type=Path)
    parser.add_argument('--head', choices=('approach', 'recovery'), required=True)
    parser.add_argument('--policy-version', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bounds = _load_bounds(args.bounds)
    event_errors = _event_metrics(args.event_metrics)
    bundle = torch.load(args.probe_bundle, map_location='cpu', weights_only=True)
    if bundle.get('schema') != 'native-paired-prefix-region-v1':
        raise ValueError('Probe bundle schema mismatch')
    if bundle.get('head') != args.head:
        raise ValueError('Probe bundle head mismatch')
    if bundle.get('runtime_identity') is None:
        raise ValueError('Probe bundle lacks runtime identity')
    model_paths = dict(physical=args.physical_checkpoint, ball=args.ball_checkpoint,
                       task=args.task_checkpoint)
    if args.action_ball_checkpoint is not None:
        model_paths['action_ball'] = args.action_ball_checkpoint
    actor, adapter, _, _, manifest = load_physical_loop(
        archive=args.archive, physical_checkpoint=args.physical_checkpoint,
        ball_checkpoint=args.ball_checkpoint, task_checkpoint=args.task_checkpoint,
        safety_mapping=args.safety_mapping, action_ball_checkpoint=args.action_ball_checkpoint,
        device=args.device, execution_root=args.g07_execution_root)
    current = runtime_identity(actor, adapter, manifest['task_contract'],
                               manifest['policy_execution'], args.policy_version)
    origin = bundle['runtime_identity']
    if origin.get('actor_state_sha256') != current['actor_state_sha256']:
        raise ValueError('Probe actor identity differs from loaded Actor')
    if origin.get('task_contract_sha256') != current['task_contract_sha256']:
        raise ValueError('Probe task contract differs from loaded runtime')
    if origin.get('policy_execution') != current['policy_execution']:
        raise ValueError('Probe execution contract differs from loaded runtime')
    code_scope = native_code_compatibility(origin.get('execution_code_sha256'),
                                           current['execution_code_sha256'])
    if not code_scope['compatible']:
        raise ValueError(f'Probe execution code is incompatible: {code_scope}')
    limits = bounds['limits']
    measured = paired_metrics(adapter, bundle['pairs'], args.head, dict(
        native_repeat_max=limits['native_repeat_max'],
        native_response_min=limits['native_response_min'],
        response_rmse_max=limits['response_rmse'],
        response_cosine_min=1.0-limits['response_cosine_error_max'],
        response_relative_error_max=limits['response_relative_error_max'],
        final_normalized_command_delta_max=limits.get('final_normalized_command_delta_max', float('inf')),
        final_joint_target_delta_rad_max=limits.get('final_joint_target_delta_rad_max', float('inf')),
        accepted_update_age_max=limits.get('accepted_update_age_max', float('inf')),
    ))
    measured['sample_ids'] = bundle.get('sample_ids', [])
    validate_native_sample_ids(measured['sample_ids'], bounds['validation_episode_ids'])
    response_errors = _response_errors(measured)
    result = dict(passed=False, runtime_identity=current, code_scope=code_scope,
                  event_aligned_errors=event_errors, response_errors=response_errors,
                  measured=measured, model_sha256={key: sha256(path) for key, path in model_paths.items()},
                  probe_bundle_sha256=sha256(args.probe_bundle), event_metrics_sha256=sha256(args.event_metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if measured['passed']:
        # Receipts reference evidence relative to their own directory.  Copy
        # immutable inputs into that directory so later verification does not
        # depend on a temporary path or silently follow a changed file.
        probe_copy=args.output.parent/'native_probe_bundle.pt'
        event_copy=args.output.parent/'event_aligned_metrics.json'
        if probe_copy.exists() or event_copy.exists():
            raise FileExistsError('Receipt evidence destination already exists')
        shutil.copy2(args.probe_bundle,probe_copy)
        shutil.copy2(args.event_metrics,event_copy)
        receipt = build_receipt(bounds=bounds, event_errors=event_errors,
                                response_errors=response_errors, models=model_paths,
                                contract=manifest['task_contract'], probe_path=probe_copy,
                                event_path=event_copy, measured=measured, identity=current)
        receipt_path = args.output.parent / '.contact_validation_receipt_check.json'
        if receipt_path.exists():
            raise FileExistsError(receipt_path)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        verify_contact_receipt(receipt_path, model_paths, manifest['task_contract'])
        receipt_path.unlink()
        result['passed'] = True
        result['receipt'] = receipt
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        return 0
    result['failure_reason'] = measured['failures']
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    raise RuntimeError('Native response direction validation failed: ' + ','.join(measured['failures']))


if __name__ == '__main__':
    main()
