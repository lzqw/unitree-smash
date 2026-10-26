#!/usr/bin/env python3
"""Verify fresh learner export against recorded origin0 commands; no simulator."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent/'unitree_smash'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from quadsmash.control.full_racket_origin0_residual_v4 import (
    make_rally_controller, frozen_origin0_controller, validate_frozen_baseline)
from quadsmash.rl.full_racket_origin0_residual_ppo_v4 import export_residual_policy
from quadsmash.rl.frozen_origin0_source_v4 import verify_frozen_origin0_source
from scripts.initialize_fresh_body_residual_v4 import initialize_fresh_body_actor
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    HERE = args.output_dir.resolve()
    assert not HERE.exists(), 'Preserve prior CPU exports and evidence; use a fresh directory'
    HERE.mkdir(parents=True)
    fixture = args.source.resolve()
    base = torch.load(fixture, map_location='cpu', weights_only=False); before = copy.deepcopy(base)
    validate_frozen_baseline(base)
    source_admission = (None if args.cold_evaluation is None else verify_frozen_origin0_source(
        fixture, base, args.cold_evaluation, ROOT, sha(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json')))
    with np.load(args.public_data) as arrays:
        public, actions, valid = [arrays[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    inputs = json.loads(args.command_metadata.read_text())
    assert inputs['protocol_sha256'] == base['protocol_sha256']
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    configuration = json.loads((ROOT/'configs/g07_full_racket_origin0_residual_leg15_v4.json').read_text())
    config = copy.deepcopy(configuration['runner']['policy']); config.pop('class_name')
    observations = TensorDict({'policy':torch.from_numpy(public[:8].reshape(-1, 794))}, batch_size=[16])
    torch.manual_seed(751001)
    policy = ActorCritic(observations, configuration['runner']['obs_groups'], 18, **config).eval()
    initialization = initialize_fresh_body_actor(policy, configuration)
    assert not policy.act_inference(observations).any()
    metadata = dict(native_execution=False, base_controller_mode='frozen_public_policy',
        base_policy_sha256=sha(fixture), actual_control_env_steps=0,
        initialization=initialization, scope='CPU export/command compatibility only; no native training.')
    exported = export_residual_policy(base, policy, configuration['joint_scale_rad'], observations['policy'], metadata)
    assert not exported['native_execution'] and not exported['performance_inherited']
    assert same_tree(exported['base_controller'], base) and same_tree(base, before)
    assert exported['protocol_sha256'] == base['protocol_sha256']
    torch.save(exported, HERE/'CPU_UNEXECUTED_ZERO_POLICY.pt')
    new, reference = make_rally_controller(exported, 2, 'cpu'), frozen_origin0_controller(base, 2, 'cpu')
    maximum_reference = maximum_native = 0.; previous = None
    with torch.inference_mode():
        for obs, actual, keep in zip(public, actions, valid):
            observation = torch.from_numpy(obs)
            if args.reset_controller_after_done and previous is not None:
                reset = torch.where(observation[:, 775] < previous-1e-5)[0]
                if len(reset):
                    new.reset(reset); reference.reset(reset)
            command = new.action({'policy':observation}, default, limits)
            unchanged = reference.action({'policy':observation}, default, limits)
            maximum_reference = max(maximum_reference, float(abs(command-unchanged).max()))
            if keep.any():
                maximum_native = max(maximum_native, float(abs(command.numpy()[keep]-actual[keep]).max()))
            previous = observation[:, 775].clone()
    assert maximum_reference == 0. and maximum_native <= 5e-4
    rejected = []
    for name, altered in [('non_native', dict(base, native_execution=False)),
                          ('old_layout', dict(base, numerical_execution=dict(variant='r3-origin16', environment_spacing_m=16.)))]:
        try:
            validate_frozen_baseline(altered)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Unmeasured/old-layout baseline accepted')
    try:
        verify_frozen_origin0_source(fixture, base,
            MAIN/'reports/G07_full_racket_v4/contact_aim7_best_fixed_train128_01', ROOT,
            sha(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'))
    except ValueError as error:
        rejected.append('old_layout_cold128_receipt:'+str(error))
    else:
        raise AssertionError('Old16m cold128 result accepted as origin0 source')
    new.reset(torch.tensor([0])); assert not new.mixer.arm_rate.initialized[0]
    assert same_tree(base, before) and sha(fixture) == metadata['base_policy_sha256']
    paths = [ROOT/name for name in ['quadsmash/control/full_racket_origin0_residual_v4.py',
        'quadsmash/rl/full_racket_origin0_residual_ppo_v4.py', 'quadsmash/rl/frozen_origin0_source_v4.py',
        'scripts/train_full_racket_origin0_residual_ppo_v4.py', 'scripts/evaluate_full_racket_origin0_residual_train_v4.py',
        'configs/g07_full_racket_origin0_residual_leg15_v4.json', 'scripts/origin0_launcher_parser_v4.py']]+[Path(__file__)]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_native_source_fixture=str(fixture),
        source_fixture_sha256=sha(fixture), actual_valid_public_rows=int(valid.sum()),
        initial_exported_actor_mean_zero=True, export_max_actor_error=exported['residual_training_metadata']['export_actor_max_error'],
        fresh_learner_normalizer_counts=[int(policy.actor_obs_normalizer.count), int(policy.critic_obs_normalizer.count)],
        requested_physical_exploration_std_rad=initialization['requested_physical_exploration_std_rad'],
        new_zero_residual_vs_unchanged_source_max_action_error=maximum_reference,
        CPU_vs_recorded_origin0_GPU_max_action_error=maximum_native, original_action_tolerance=5e-4,
        immutable_source_tree_and_original_actors_preserved=True, partial_reset_checked=True,
        rejected_invalid_sources=rejected, positive_completed_origin0_cold_source_verification_pending=source_admission is None,
        actual_cold_source_admission=source_admission, actual_public_data_sha256=sha(args.public_data),
        actual_command_metadata_sha256=sha(args.command_metadata),
        reset_controller_after_done=args.reset_controller_after_done,
        CPU_unexecuted_policy_sha256=sha(HERE/'CPU_UNEXECUTED_ZERO_POLICY.pt'),
        source_sha256={str(path):sha(path) for path in paths},
        scope='Actual public-input CPU inference/export compatibility. No new simulator execution, PPO rollout/update, performance, or positive cold admission claimed.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE/'CPU_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--public-data', type=Path, required=True)
    parser.add_argument('--command-metadata', type=Path, required=True)
    parser.add_argument('--cold-evaluation', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reset-controller-after-done', action='store_true')
    main(parser.parse_args())
