#!/usr/bin/env python3
"""Fit an action-conditioned ball head against measured native pair probes.

The split is declared by the caller before fitting.  This utility never chooses
lanes from the measured response, never creates counterfactual labels, and never
issues a promotion receipt.  It is a bounded WM calibration step only.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_ball_v1 import (
    action_ball_from_checkpoint, migrate_temporal_v7, migrate_temporal_v8,
    migrate_temporal_v9, migrate_temporal_v10, migrate_temporal_v11,
)
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_evidence_binding_v1 import NATIVE_REQUIRED_CODE


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(paths, device):
    rows = []
    for path in paths:
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if bundle.get('schema') != 'native-paired-prefix-region-v1':
            raise ValueError(f'Unsupported probe bundle schema: {path}')
        pairs = bundle.get('pairs', {})
        required = ('state', 'targets_minus', 'targets_plus', 'previous_action',
                    'feet', 'present', 'elapsed', 'launch', 'native_minus',
                    'native_plus')
        if any(key not in pairs for key in required):
            raise ValueError(f'Incomplete native probe bundle: {path}')
        for key in required:
            if not torch.is_tensor(pairs[key]) or not torch.isfinite(pairs[key]).all():
                raise ValueError(f'Invalid native probe tensor {key}: {path}')
        rows.append(bundle)
    if not rows:
        raise ValueError('At least one probe bundle is required')
    return rows


def concat_pairs(rows, device):
    keys = ('state', 'targets_minus', 'targets_plus', 'previous_action',
            'feet', 'present', 'elapsed', 'launch', 'native_minus', 'native_plus')
    result={key: torch.cat([row['pairs'][key].to(device) for row in rows], dim=0)
            for key in keys}
    temporal_keys=tuple(key for key,value in rows[0]['pairs'].items()
                        if key.startswith('step_') and torch.is_tensor(value))
    if temporal_keys and any(
        tuple(key for key,value in row['pairs'].items()
              if key.startswith('step_') and torch.is_tensor(value)) != temporal_keys
        for row in rows):
        raise ValueError('Temporal probe tensor schemas differ')
    for key in temporal_keys:
        result[key]=torch.cat([row['pairs'][key].to(device) for row in rows], dim=0)
    for key in ('temporal_horizon','action_tick','event_tick'):
        values=[row['pairs'].get(key) for row in rows]
        if any(value is not None for value in values):
            if any(value is None for value in values):
                raise ValueError('Temporal probe metadata missing: '+key)
            if key == 'temporal_horizon':
                if len(set(values)) != 1:
                    raise ValueError('Temporal probe horizon differs')
                result[key] = values[0]
            else:
                # Absolute event ticks are provenance for each source bundle.
                # They need not match when every bundle stores the same causal
                # rollout length and action/event ordering.
                result[key] = values[0] if len(set(values)) == 1 else None
    return result


def validate_probe_identity_reuse(rows):
    """Allow native labels to be reused across prediction-code revisions.

    This fitter calibrates a prediction model against immutable native labels.
    It therefore requires the actor, task contract, policy execution contract,
    policy version, and every native execution source to match.  Prediction
    and bundle-analysis source revisions are recorded as provenance and are
    allowed to differ.  This does not authorize a native receipt or Actor BPTT.
    """
    identities = [row.get('runtime_identity') for row in rows]
    if any(not isinstance(identity, dict) for identity in identities):
        raise ValueError('Native calibration bundle is missing runtime identity')
    base = identities[0]
    required = ('schema', 'actor_state_sha256', 'task_contract_sha256',
                'policy_execution', 'policy_version')
    for identity in identities[1:]:
        mismatches = [key for key in required if identity.get(key) != base.get(key)]
        if mismatches:
            raise ValueError('Native calibration identity mismatch: ' + ','.join(mismatches))
        recorded = base.get('execution_code_sha256', {})
        current = identity.get('execution_code_sha256', {})
        native_mismatches = [key for key in NATIVE_REQUIRED_CODE
                             if recorded.get(key) != current.get(key)]
        if native_mismatches:
            raise ValueError('Native execution identity mismatch: ' + ','.join(sorted(native_mismatches)))
    return dict(
        shared_identity={key: base.get(key) for key in required},
        native_execution_code_sha256={key: base.get('execution_code_sha256', {}).get(key)
                                      for key in sorted(NATIVE_REQUIRED_CODE)},
        bundle_identity_count=len(identities),
        prediction_code_sha256=[identity.get('execution_code_sha256', {}) for identity in identities],
        prediction_code_reuse=True,
        receipt_status='not_issued_requires_independent_native_direction_validation',
    )


def response(model, physical, ball, pairs):
    temporal='step_targets_minus' in pairs
    def rollout(name):
        if not temporal:
            common=(pairs['state'],pairs['previous_action'],pairs['feet'],
                    pairs['present'],pairs['elapsed'],pairs['launch'])
            return ball_transition(physical,ball,common[0],pairs['targets_'+name],
                                   *common[1:],action_model=model)[0]
        state=pairs['state'];previous=pairs['previous_action'];feet=pairs['feet']
        present=pairs['present'];elapsed=pairs['elapsed']
        targets=pairs['step_targets_'+name];actions=pairs['step_final_actions_'+name]
        for i in range(targets.shape[1]):
            state,contact,_=ball_transition(physical,ball,state,targets[:,i],previous,feet,present,elapsed,pairs['launch'],action_model=model)
            previous=actions[:,i];feet=contact[:,:feet.shape[1]];elapsed=elapsed+elapsed.new_tensor(.01)
        return state
    minus=rollout('minus');plus=rollout('plus')
    return plus[:, 52:55] - minus[:, 52:55]


def metrics(predicted, pairs):
    native = pairs['native_plus'][:, 52:55] - pairs['native_minus'][:, 52:55]
    norms = native.norm(dim=-1)
    cosine = (predicted * native).sum(-1) / (
        predicted.norm(dim=-1) * norms).clamp_min(1e-12)
    relative = (predicted - native).norm(dim=-1) / norms.clamp_min(1e-12)
    supported = norms > 1e-7
    if not supported.any():
        raise ValueError('Probe split has no non-zero native ball response')
    error = predicted[supported] - native[supported]
    native_supported = native[supported]
    predicted_supported = predicted[supported]
    zero_error = native_supported
    return dict(
        count=int(len(native)),
        supported_count=int(supported.sum()),
        response_rmse=float((predicted[supported] - native[supported]).square().mean().sqrt()),
        component_rmse=[float(value) for value in error.square().mean(0).sqrt()],
        native_component_rms=[float(value) for value in native_supported.square().mean(0).sqrt()],
        predicted_component_rms=[float(value) for value in predicted_supported.square().mean(0).sqrt()],
        zero_response_baseline_rmse=float(zero_error.square().mean().sqrt()),
        response_cosine=cosine.tolist(),
        response_cosine_min=float(cosine[supported].min()),
        relative_error=relative.tolist(),
        relative_error_max=float(relative[supported].max()),
        native_response_norm=norms.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-checkpoint', type=Path, required=True)
    parser.add_argument('--ball-checkpoint', type=Path, required=True)
    parser.add_argument('--action-ball-checkpoint', type=Path, required=True)
    parser.add_argument('--train-bundle', type=Path, nargs='+', required=True)
    parser.add_argument('--validation-bundle', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--direction-weight', type=float, default=1.0,
                        help='Weight of normalized native-response direction loss.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.updates != 256:
        raise ValueError('Native response calibration uses exactly 256 WM updates')
    if not torch.isfinite(torch.tensor(args.direction_weight)) or args.direction_weight <= 0:
        raise ValueError('--direction-weight must be finite and positive')
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        raise RuntimeError('GPU required for native response calibration')
    device = torch.device(args.device)
    torch.manual_seed(20261119)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.train_bundle, device)
    validation_rows = load_rows(args.validation_bundle, device)
    train_ids = [sid for row in train_rows for sid in row['sample_ids']]
    validation_ids = [sid for row in validation_rows for sid in row['sample_ids']]
    if not train_ids or not validation_ids:
        raise ValueError('Declared native calibration split cannot be empty')
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError('Duplicate native sample identity in calibration split')
    if set(train_ids) & set(validation_ids):
        raise ValueError('Native calibration train and validation identities overlap')
    identity_reuse = validate_probe_identity_reuse((*train_rows, *validation_rows))
    train = concat_pairs(train_rows, device)
    validation = concat_pairs(validation_rows, device)
    physical_source = torch.load(args.physical_checkpoint, map_location='cpu', weights_only=True)
    ball_source = torch.load(args.ball_checkpoint, map_location='cpu', weights_only=True)
    candidate_source = torch.load(args.action_ball_checkpoint, map_location='cpu', weights_only=True)
    contract = physical_source['source_manifest']['task_contract']
    if candidate_source.get('physical_sha256') != digest(args.physical_checkpoint):
        raise ValueError('Action-ball candidate belongs to a different physical model')
    if candidate_source.get('legacy_ball_sha256') != digest(args.ball_checkpoint):
        raise ValueError('Action-ball candidate belongs to a different legacy ball model')
    physical = PhysicalWorldV2(contract['asset']['sha256']).to(device)
    physical.load_state_dict(physical_source['model'], strict=True)
    ball = BallResidualV2().to(device)
    ball.load_state_dict(ball_source['model'], strict=True)
    physical.requires_grad_(False).eval()
    ball.requires_grad_(False).eval()
    temporal_fit = 'step_targets_minus' in train
    temporal_schemas = {
        'causal-racket-mode-action-sweep-v3',
        'temporal-causal-racket-mode-action-sweep-v7',
        'native-temporal-causal-racket-mode-target-action-sweep-v8',
        'native-temporal-causal-racket-mode-target-floor-action-sweep-v9',
        'native-temporal-causal-racket-mode-target-linear-floor-action-sweep-v10',
    }
    if temporal_fit and candidate_source.get('schema') in temporal_schemas:
        model = migrate_temporal_v11(candidate_source).to(device)
    else:
        model = action_ball_from_checkpoint(candidate_source).to(device)
        model.load_state_dict(candidate_source['model'], strict=True)
    model.train()
    native_head_only = model.__class__.__name__ == 'TemporalCausalRacketModeNativeResponseBallV11'
    if native_head_only:
        # V11 is a residual calibrator over the already fitted V10 candidate.
        # Moving the inherited geometry/command heads during this small native
        # fit can lower the aggregate loss while destroying the old response
        # on lanes absent from the calibration split.  Keep that base fixed
        # and learn only the newly introduced causal native-response head.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('native_response_head.'))
    command_head = getattr(model, 'command_head', None)
    if native_head_only:
        optimizer = torch.optim.Adam(
            model.native_response_head.parameters(), lr=args.lr)
    elif command_head is not None:
        # The executable-target channel sees small perturbations (often below
        # 1e-2 rad).  Give that bounded calibration branch its own declared
        # step size instead of letting the nonlinear residual head absorb all
        # of the native-response gradient.
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if not n.startswith('command_head.')],
             'lr': args.lr},
            {'params': command_head.parameters(), 'lr': max(args.lr, 5e-3)},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial = metrics(response(model, physical, ball, validation), validation)
    train_native = train['native_plus'][:, 52:55] - train['native_minus'][:, 52:55]
    history = []
    best_validation = initial
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_update = 0
    selection_rule = (
        'among holdout candidates with response_rmse <= zero_response_baseline_rmse '
        'and relative_error_max <= 1, maximize response_cosine_min, then minimize response_rmse; '
        'otherwise minimize response_rmse'
    )

    def selection_key(metrics_value):
        eligible = (
            metrics_value['response_rmse'] <= metrics_value['zero_response_baseline_rmse']
            and metrics_value['relative_error_max'] <= 1.0
        )
        if eligible:
            return (1, metrics_value['response_cosine_min'], -metrics_value['response_rmse'])
        return (0, -metrics_value['response_rmse'], metrics_value['response_cosine_min'])

    best_key = selection_key(best_validation)
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = response(model, physical, ball, train)
        native_norm = train_native.norm(dim=-1)
        supported = native_norm > 1e-7
        if not supported.any():
            raise ValueError('Training split has no non-zero native response')
        predicted_supported = predicted[supported]
        native_supported = train_native[supported]
        cosine = (predicted_supported * native_supported).sum(-1) / (
            predicted_supported.norm(dim=-1) * native_supported.norm(dim=-1)
        ).clamp_min(1e-8)
        point_loss = ((predicted_supported - native_supported) / 0.01).square().mean()
        direction_loss = (1.0 - cosine).mean()
        loss = point_loss + args.direction_weight * direction_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if update == 1 or update % 16 == 0 or update == args.updates:
            with torch.no_grad():
                holdout = metrics(response(model, physical, ball, validation), validation)
            history.append(dict(update=update, loss=float(loss.detach()), validation=holdout))
            holdout_key = selection_key(holdout)
            if holdout_key > best_key:
                best_validation = holdout
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                best_update = update
                best_key = holdout_key
    with torch.no_grad():
        # Use the best fixed-split calibration iterate rather than the final
        # Adam iterate.  This is still a candidate fit and never a native
        # receipt, but avoids reporting a known overfit tail as the result.
        model.load_state_dict(best_state, strict=True)
        final = metrics(response(model, physical, ball, validation), validation)
    state = dict(candidate_source)
    state.update(
        model={key: value.detach().cpu() for key, value in model.state_dict().items()},
        schema=model.schema,
        temporal_response_fit=bool(temporal_fit),
        updates=args.updates,
        native_response_calibration=dict(
            train_bundles=[str(path.resolve()) for path in args.train_bundle],
            validation_bundles=[str(path.resolve()) for path in args.validation_bundle],
            train_sample_ids=train_ids,
            validation_sample_ids=validation_ids,
            split_is_declared_before_fit=True,
            source_checkpoint_sha256=digest(args.action_ball_checkpoint),
            no_promotion_receipt=True,
            identity_reuse=identity_reuse,
            native_head_only=native_head_only,
            best_validation_update=best_update,
            selection_rule=selection_rule,
        ),
    )
    torch.save(state, args.output)
    result = dict(
        schema='native-response-calibration-result-v1',
        output=str(args.output.resolve()),
        output_sha256=digest(args.output),
        updates=args.updates,
        device=str(device),
        train_sample_count=len(train['state']),
        validation_sample_count=len(validation['state']),
        validation_initial=initial,
        validation_final=final,
        history=history,
        elapsed_s=time.perf_counter() - started,
        passed=None,
        receipt_status='not_issued_requires_independent_native_direction_validation',
    )
    (args.output.parent / 'calibration_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

