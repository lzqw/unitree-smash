"""Check completed native reference generation against independent source128."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN / 'reports/G07_full_racket_v5'
RUN = REPORTS / 'measured_swing9_n128_01'
SOURCE = REPORTS / 'teacher_contact_recovery_cold128_02'
POLICY = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    assert Path('/proc/1951201').exists()
    row_list = read(RUN / 'episodes_000.json')
    old_rows = read(SOURCE / 'episodes.json')
    stripped = [{key: value for key, value in row.items() if key not in
                 ['candidate_id', 'closest_string_gap_m', 'logger_ordinal']} for row in row_list]
    differing = [i for i, (a, b) in enumerate(zip(stripped, old_rows))
                 if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)]
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    summary = summarize_training(stripped, cases, indices, protocol)
    assert all(summary[key] == read(SOURCE / 'metrics.json')[key] for key in summary)
    with np.load(SOURCE / 'all_first_episodes.npz') as old, np.load(RUN / 'all_first_episodes_000.npz') as new:
        masks_equal = np.array_equal(old['valid_first_episode'], new['valid_first_episode'])
        keep = old['valid_first_episode'] & new['valid_first_episode']
        comparisons = {key: dict(bitwise_equal=bool(np.array_equal(old[key][keep], new[key][keep])),
            maximum_absolute_difference=float(abs(old[key][keep] - new[key][keep]).max())) for key in
            ['link_poses', 'ball_position_m', 'episode_clock_s']}
    inputs = read(RUN / 'INPUTS.json')
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(RUN / 'public_and_actions_000.npz') as data:
        public, actions, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
        recorded = data['training_indices'].tolist()
    assert recorded == [0, 29, 69, 100, 112, 123]
    controller = make_rally_controller(torch.load(POLICY, map_location='cpu', weights_only=False), 6, 'cpu')
    maximum, previous = 0., None
    with torch.inference_mode():
        for value, action, active in zip(public, actions, valid):
            obs = torch.from_numpy(value)
            if previous is not None:
                ids = torch.where(obs[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    controller.reset(ids)
            actual = controller.action({'policy': obs}, default, limits).numpy()
            if active.any():
                maximum = max(maximum, float(abs(actual[active] - action[active]).max()))
            previous = obs[:, 775].clone()
    assert maximum <= 5e-4
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), source_original128_summary=summary,
        original128_rows_differing_after_removing_only_new_search_metadata=differing,
        valid_masks_bitwise_equal=masks_equal, common_valid_case_frames=int(keep.sum()),
        actual_recorded_motion_comparison=comparisons, actual_public_training_indices=recorded,
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, source_policy_sha256=sha(POLICY),
        previous_goal_turn_classification='progress: completed15 artifacts delivered and native9 comparison launched',
        current_native_process_confirmed_live=1951201, live_search_not_restarted=True,
        source_sha256={str(path): sha(path) for path in [Path(__file__), POLICY, RUN / 'episodes_000.json',
            RUN / 'all_first_episodes_000.npz', RUN / 'public_and_actions_000.npz',
            SOURCE / 'episodes.json', SOURCE / 'all_first_episodes.npz', protocol_path]},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    serialized = json.dumps(result, indent=2) + '\n'
    with (HERE / 'ACTUAL_REFERENCE_RESULT.json').open('x') as stream:
        stream.write(serialized)
    print(json.dumps({key: value for key, value in result.items() if key not in
                     ['source_sha256', 'source_original128_summary']}))


if __name__ == '__main__':
    main()
