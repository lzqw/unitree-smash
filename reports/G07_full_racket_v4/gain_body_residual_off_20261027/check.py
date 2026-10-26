"""Independently check the completed residual-off ablation; preserve native IDs."""
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


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'RESULT.json').exists()
    search = REPORTS / 'gain_body8_sequential_n128_01'
    source_path = REPORTS / 'prepose_origin0_8_sequential_n128_01/best_candidate.pt'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs = read(search / 'INPUTS.json')
    assert inputs['source_policy_sha256'] == sha(source_path)
    assert inputs['protocol_sha256'] == sha(protocol_path) and inputs['training_indices'] == indices
    history = read(search / 'metrics.json')['history']
    entries, counts, source_files = [], [], [source_path, protocol_path, Path(__file__), search / 'INPUTS.json']
    for candidate_id in [0, 1]:
        suffix = f'{candidate_id:03d}'
        entry = next(e for e in history if e['candidate_id'] == candidate_id)
        rows = read(search / f'episodes_{suffix}.json')
        initial = read(search / f'initial_ground_states_{suffix}.json')
        assert len(rows) == 128 and initial['all_clocks_zero']
        assert not np.asarray(initial['actual_environment_origins_m']).any()
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == candidate_id
            assert row['episode_index'] == initial['episode_ids'][lane]
            assert row['source_training_index'] == lane and row['case_id'] == cases[lane]['case_id'] and row['ground_init']
        assert summarize_candidate(rows, protocol) == entry['summary'] and entry['summary']['numerical_failures'] == 0
        entries.append(entry)
        counts.append(dict(candidate_id=candidate_id, actual_episode_indices=initial['episode_ids'],
            actual_racket_contacts=sum(row['had_contact'] for row in rows),
            frame_hits=sum(row['frame_hit'] for row in rows),
            body_landings=sum(row['jump']['body_landing'] for row in rows),
            stable_recovery_flags=sum(row['jump']['stable_recovery'] for row in rows),
            actual_target_first_landings=sum(row['landed_target'] for row in rows)))
        source_files += [search / f'{name}_{suffix}.json' for name in ['episodes', 'initial_ground_states']]
    candidate = entries[1]['candidate']
    assert candidate['gains'] == [[0., 0.], [0., 0.]] and not candidate['body_deltas']
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    controller = GainBodyAdaptationControllerV4(source, candidate['gains'], 2, 'cpu', {})
    default, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    stream = search / 'public_and_actions_001.npz'
    with np.load(stream) as arrays:
        public, actions, valid = [arrays[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
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
    held = read(REPORTS / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    source_files += [stream, search / 'contacts_001.json', search / 'all_first_episodes_001.npz']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_candidate_entries=entries,
        original256_rows_and_native_episode_ids_checked=True, additional_observed_counts=counts,
        actual_valid_residual_off_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, held482_unchanged=True,
        source_sha256={str(path): sha(path) for path in source_files},
        interpretation='Disabling both existing residual layers gives0 airborne contacts versus4 for the source reference, without improving complete success or qualified recovery. Stable flags2 versus1 do not count as contact-conditioned recovery without an airborne string contact.',
        scope='Two completed sequential128-case search candidates. The other six candidates and the independent cold128 remain pending. Native contact-cache and reset schedules differ; no causal full-trajectory identity claim.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_valid_residual_off_public_rows', 'CPU_vs_actual_GPU_max_action_error']}), flush=True)


if __name__ == '__main__':
    main()
