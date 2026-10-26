"""Locate the observed native divergence without attributing an unproven cause."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'NATIVE_PREFIX_DIAGNOSIS.json').exists()
    search = REPORTS / 'gain_body8_sequential_n128_01'
    cold = REPORTS / 'prepose_origin0_8_best_fixed_train128_01'
    rows = read(search / 'episodes_000.json')
    earliest_terminal = min((row['terminal_native_state']['time_s'], row['env_id']) for row in rows)
    with np.load(search / 'all_first_episodes_000.npz') as a, np.load(cold / 'all_first_episodes.npz') as b:
        common = a['valid_first_episode'] & b['valid_first_episode']
        clock = a['episode_clock_s']
        pose_equal = (a['link_poses'] == b['link_poses']).all(axis=(-1, -2))
        different = common & ~pose_equal
        first_frame, first_lane = np.argwhere(different)[0]
        first_time = float(clock[first_frame, first_lane])
        before_reset = common & (clock < earliest_terminal[0])
        assert pose_equal[before_reset].all()
        assert first_time > earliest_terminal[0]
        assert np.array_equal(a['ball_position_m'][common], b['ball_position_m'][common])
        maximum = np.max(np.linalg.norm(a['link_poses'][..., :3] - b['link_poses'][..., :3], axis=-1), axis=-1)
        worst_frame, worst_lane = np.unravel_index(np.where(common, maximum, -1).argmax(), maximum.shape)
        identity = dict(shared_valid_pose_rows=int(common.sum()), differing_pose_rows=int(different.sum()),
            differing_TRAIN_indices=np.unique(np.where(different)[1]).tolist(),
            earliest_search_terminal_control_step_time_s=earliest_terminal[0],
            earliest_search_terminal_TRAIN_index=earliest_terminal[1],
            all_shared_poses_before_first_search_terminal_bitwise_identical=True,
            earliest_differing_pose_time_s=first_time, earliest_differing_TRAIN_index=int(first_lane),
            maximum_link_position_difference_m=float(maximum[worst_frame, worst_lane]),
            maximum_difference_time_s=float(clock[worst_frame, worst_lane]),
            maximum_difference_TRAIN_index=int(worst_lane), all_shared_balls_bitwise_identical=True)
        a_events, b_events = read(search / 'contacts_000.json'), read(cold / 'contacts.json')
        contacts = []
        for lane in [69, 100, 112, 123]:
            a_event = next(e for e in a_events if e['env_id'] == lane and e['episode_index'] == 0)
            b_event = next(e for e in b_events if e['env_id'] == lane and e['episode_index'] == 0)
            fields = ['time_s', 'point_m', 'body_link_poses_local', 'point_velocity_mps', 'ball_velocity_mps']
            matches = {key: bool(np.array_equal(a_event[key], b_event[key])) for key in fields}
            shared_pose_match = bool(pose_equal[:, lane][common[:, lane]].all())
            assert all(matches.values()) and shared_pose_match
            contacts.append(dict(TRAIN_index=lane, original_episode_index=0, actual_contact_fields=matches,
                all_shared_first_episode_body_poses_bitwise_identical=shared_pose_match))
    files = [search / n for n in ['episodes_000.json', 'all_first_episodes_000.npz', 'contacts_000.json']]
    files += [cold / n for n in ['episodes.json', 'all_first_episodes.npz', 'contacts.json']]
    files += [Path(__file__), HERE / 'RESULT.json']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), comparison=identity,
        actual_four_contact_matches=contacts, all_four_contacts_match_before_first_early_reset=True,
        interpretation='The identity controller reproduces every actual contact and all shared poses before the first search termination. Later shared poses diverge in24 non-airborne-contact lanes. The search and cold environments have different termination/reset schedules; this observation alone does not identify the causal mechanism.',
        unresolved='No controlled same-environment asynchronous-reset experiment has been run. Do not claim full-episode native identity or identify a logger/controller/PhysX bug from these two traces alone.',
        scoring_impact='No current success count changed. All four contacts still fail speed and recovery. Modified candidates require separate frozen cold128 and original formal gates.',
        source_sha256={str(path): sha(path) for path in files}, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False)
    (HERE / 'NATIVE_PREFIX_DIAGNOSIS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
