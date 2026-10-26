"""Check the completed native identity candidate against the preceding cold source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_gain_body_adaptation_v4 import GainBodyAdaptationControllerV4
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'RESULT.json').exists()
    search = REPORTS / 'gain_body8_sequential_n128_01'
    cold = REPORTS / 'prepose_origin0_8_best_fixed_train128_01'
    source_path = REPORTS / 'prepose_origin0_8_sequential_n128_01/best_candidate.pt'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs = read(search / 'INPUTS.json')
    entry = read(search / 'metrics.json')['history'][0]
    assert entry['candidate_id'] == 0 and entry['candidate']['name'] == 'unchanged_source'
    assert entry['candidate']['gains'] == [[1., 1.], [1., 1.]] and not entry['candidate']['body_deltas']
    assert inputs['source_policy_sha256'] == sha(source_path) == read(cold / 'INPUTS.json')['policy_sha256']
    assert inputs['protocol_sha256'] == sha(protocol_path) == read(cold / 'INPUTS.json')['protocol_sha256']
    assert inputs['training_indices'] == indices == list(range(128))
    initial = read(search / 'initial_ground_states_000.json')
    rows = read(search / 'episodes_000.json')
    assert len(rows) == 128 and initial['all_clocks_zero']
    assert not np.asarray(initial['actual_environment_origins_m']).any()
    for lane, row in enumerate(rows):
        assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == 0
        assert row['episode_index'] == initial['episode_ids'][lane] == 0
        assert row['source_training_index'] == lane and row['case_id'] == cases[lane]['case_id'] and row['ground_init']
    summary = summarize_candidate(rows, protocol)
    assert summary == entry['summary'] and summary['numerical_failures'] == 0
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    controller = GainBodyAdaptationControllerV4(source, entry['candidate']['gains'], 2, 'cpu', {})
    default, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(search / 'public_and_actions_000.npz') as stream:
        public, actions, valid = [stream[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
    maximum, previous = 0., None
    with torch.inference_mode():
        for obs, actual, keep in zip(public, actions, valid):
            observation = torch.from_numpy(obs)
            if previous is not None:
                reset = torch.where(observation[:, 775] < previous - 1e-5)[0]
                if len(reset):
                    controller.reset(reset)
            command = controller.action({'policy': observation}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(command[keep] - actual[keep]).max()))
            previous = observation[:, 775].clone()
    assert maximum <= 5e-4
    cold_rows = read(cold / 'episodes.json')
    air = [i for i, row in enumerate(rows) if independent_episode(row, protocol)['airborne_contact']]
    assert air == [i for i, row in enumerate(cold_rows) if independent_episode(row, protocol)['airborne_contact']]
    with np.load(search / 'all_first_episodes_000.npz') as a, np.load(cold / 'all_first_episodes.npz') as b:
        common = a['valid_first_episode'] & b['valid_first_episode']
        assert common.shape == (400, 128) and common.any(axis=0).all()
        assert np.array_equal(a['episode_clock_s'][common], b['episode_clock_s'][common])
        position_error = np.linalg.norm(a['link_poses'][common][..., :3] - b['link_poses'][common][..., :3], axis=-1)
        ball_error = np.linalg.norm(a['ball_position_m'][common] - b['ball_position_m'][common], axis=-1)
        shared = dict(common_valid_native50Hz_rows=int(common.sum()),
            maximum_link_position_difference_m=float(position_error.max()),
            maximum_ball_position_difference_m=float(ball_error.max()),
            poses_bitwise_identical=np.array_equal(a['link_poses'][common], b['link_poses'][common]),
            balls_bitwise_identical=np.array_equal(a['ball_position_m'][common], b['ball_position_m'][common]),
            scope='Shared valid first-episode samples only. Search ends body-ground episodes earlier than cold evaluation, so later cold recovery and whole-episode metrics are not compared as identical.')
    files = [source_path, protocol_path, ROOT / protocol['training_suite']['path'], search / 'INPUTS.json',
        search / 'episodes_000.json', search / 'initial_ground_states_000.json', search / 'contacts_000.json',
        search / 'public_and_actions_000.npz', search / 'all_first_episodes_000.npz',
        cold / 'INPUTS.json', cold / 'episodes.json', cold / 'all_first_episodes.npz', Path(__file__)]
    held = read(REPORTS / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(p)) == s for p, s in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_candidate_entry=entry,
        original128_rows_and_actual_episode_ids_checked=True, actual_airborne_TRAIN_indices=air,
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, actual_native_shared_prefix_comparison=shared,
        held482_unchanged=True, source_sha256={str(p): sha(p) for p in files},
        caveat='One completed reference candidate inside an ongoing8-way job. No whole-job completion or modified-candidate improvement inferred.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_airborne_TRAIN_indices', 'actual_valid_public_rows',
        'CPU_vs_actual_GPU_max_action_error', 'actual_native_shared_prefix_comparison']}), flush=True)


if __name__ == '__main__':
    main()
