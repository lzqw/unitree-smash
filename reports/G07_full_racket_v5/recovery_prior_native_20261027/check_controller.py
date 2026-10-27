"""Check all9 native-ready variants against actual published CPU action digests."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
SEED = HERE.parent / 'recovery_motion_prior_20261027'
sys.path.insert(0, str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('native_ready_recovery_prior', HERE / 'prior_controller.py')
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    metadata, arrays = factory.load_seed()
    policy = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source = torch.load(policy, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    measured = HERE.parent / 'measured_swing9_n128_01'
    data_path = measured / 'public_and_actions_000.npz'
    inputs = json.loads((measured / 'INPUTS.json').read_text())
    defaults, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                        ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(data_path) as data:
        public, native, valid = [data[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
        assert data['training_indices'].tolist() == [0, 29, 69, 100, 112, 123]
    assert public.shape == (800, 6, 794) and valid.sum() == 4800
    previous_check = json.loads((SEED / 'FRESH_ALL9_PUBLIC_CPU_CHECK.json').read_text())
    assert previous_check['public_dataset_sha256'] == sha(data_path)
    original = factory.make_base(source, 6, 'cpu')
    expected = []
    for value in public:
        expected.append(original.action({'policy': torch.from_numpy(value)}, defaults, limits).clone())
    results = []
    for candidate in factory.candidates():
        controller = factory.RecoveryPriorController(source, metadata, arrays, 6, 'cpu',
            **{k: candidate[k] for k in ['group', 'table', 'alignment', 'gain']})
        digest = hashlib.sha256()
        maximum_native = maximum_difference = maximum_rate = 0.
        previous = None
        landed_rows = 0
        seen = torch.zeros(6, dtype=torch.bool)
        for value, original_native, reference in zip(public, native, expected):
            obs = torch.from_numpy(value)
            snapshot = obs.clone()
            actual = controller.action({'policy': obs}, defaults, limits)
            assert torch.equal(obs, snapshot)
            assert actual.shape == (6, 18) and torch.isfinite(actual).all() and (abs(actual) <= 1).all()
            digest.update(actual.numpy().tobytes())
            maximum_native = max(maximum_native, float(abs(actual.numpy()-original_native).max()))
            maximum_difference = max(maximum_difference, float(abs(actual-reference).max()))
            if candidate['group'] == 'legs':
                assert torch.equal(actual[:, 12:], reference[:, 12:])
            elif candidate['group'] == 'arms':
                assert torch.equal(actual[:, :12], reference[:, :12])
            seen |= obs[:, 773] > .5
            if candidate['alignment'] == 'observed_hit':
                assert torch.equal(actual[~seen], reference[~seen])
                landed_rows += int((seen & (obs[:, 774] > .5) & (controller.last_request.abs().amax(1) > 0)).sum())
            target = limits[:, 0] + .5*(actual+1)*(limits[:, 1]-limits[:, 0])
            if previous is not None:
                maximum_rate = max(maximum_rate, float(abs(target[:, 12:]-previous[:, 12:]).max())/.01)
            previous = target.clone()
        old = previous_check['results'][candidate['candidate_id']]
        assert digest.hexdigest() == old['replayed_action_stream_sha256']
        assert maximum_rate <= 5.0005 and same_tree(source, before)
        if candidate['candidate_id'] == 0:
            assert maximum_difference == 0. and maximum_native <= 5e-4
        if candidate['alignment'] == 'observed_hit':
            assert landed_rows > 0
        saved_seen, saved_clock = controller.air_hit_seen[1:].clone(), controller.first_public_hit_clock[1:].clone()
        controller.reset(torch.tensor([0]))
        assert torch.equal(saved_seen, controller.air_hit_seen[1:])
        assert torch.equal(saved_clock, controller.first_public_hit_clock[1:])
        result = dict(candidate_id=candidate['candidate_id'], actual_public_rows=int(valid.sum()),
            maximum_original_GPU_action_difference=maximum_native,
            maximum_source_action_difference=maximum_difference, maximum_arm_target_rate_rad_s=maximum_rate,
            replayed_action_stream_sha256=digest.hexdigest(), entire_stream_matches_previously_downloaded_family=True,
            partial_reset_preserves_other_lanes=True, unchanged_public_inputs=True,
            nonzero_prior_grounded_rows_after_observed_hit=landed_rows)
        results.append(result)
        print(json.dumps(result), flush=True)
    files = [policy, data_path, measured / 'INPUTS.json', HERE / 'prior_controller.py', Path(__file__),
        HERE / 'prepare_factory.py', HERE / 'FACTORY_GENERATION_CHECK.json',
        *[SEED / name for name in ['prior_controller.py', 'recorded_recovery_prior.npz', 'SEED_RECEIPT.json',
            'CPU_PRIOR_CHECK.json', 'FRESH_ALL9_PUBLIC_CPU_CHECK.json']],
        HERE.parent / 'windowed_swing_20261027/window_controller.py',
        HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py',
        HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidates=factory.candidates(), results=results,
        factory_sha256=sha(HERE / 'prior_controller.py'), source_sha256={str(p): sha(p) for p in files},
        source_actor_and_normalizer_and_checkpoint_tree_unchanged=True,
        all9_action_streams_match_published_portable_proposals_bitwise=True,
        actual_windowed_source_admission_and_all9_replays_pending=True,
        native_execution=False, formal_acceptance=False, goal_complete=False)
    with (HERE / 'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
