"""Measure actual pre-ball preparation changes in completed swing generations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'measured_swing9_n128_01'
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path('/proc/1951201').exists()
    history = json.loads((RUN / 'metrics.json').read_text())['history']
    completed = [row for row in history if row['candidate_id'] <= 4]
    assert len(completed) == 5
    names = json.loads((RUN / 'geometry.json').read_text())['body_names']
    tool = names.index('Racket')
    with np.load(RUN / 'all_first_episodes_000.npz') as data:
        reference, clock = data['link_poses'].copy(), data['episode_clock_s'].copy()
    entries, dependencies = [], [Path(__file__), RUN / 'all_first_episodes_000.npz', RUN / 'geometry.json']
    for candidate in completed:
        cid = candidate['candidate_id']
        data_path = RUN / f'all_first_episodes_{cid:03d}.npz'
        with np.load(data_path) as data:
            pose, times = data['link_poses'].copy(), data['episode_clock_s'].copy()
        measurements = []
        for target in [1., 1.5, 2.]:
            index = int(np.argmin(abs(clock[:, 0] - target)))
            assert abs(float(clock[index, 0]) - target) < 1e-5
            assert np.array_equal(times[index], clock[index])
            arm = pose[index, :, tool, :3] - reference[index, :, tool, :3]
            root = pose[index, :, 0, :3] - reference[index, :, 0, :3]
            measurements.append(dict(clock_s=target, original_cases=128,
                actual_racket_center_displacement_median_m=float(np.median(np.linalg.norm(arm, axis=1))),
                actual_racket_center_signed_mean_delta_m=arm.mean(0).tolist(),
                actual_root_position_displacement_median_m=float(np.median(np.linalg.norm(root, axis=1))),
                actual_root_signed_mean_delta_m=root.mean(0).tolist()))
        entries.append(dict(candidate_id=cid, definitions=candidate['candidate'],
            actual_original128_summary={key: candidate['summary'][key] for key in
                ['complete', 'airborne_contacts', 'conditional_recovery_count', 'attack_outgoing', 'numerical_failures']},
            actual_preparation_measurements=measurements))
        dependencies.append(data_path)
    teacher = ROOT / 'quadsmash/control/full_racket_teacher_v4.py'
    code = teacher.read_text()
    assert 'arm_pre=arm_hit-half_window[:,None]*velocity' in code
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_native_generations_checked=5,
        active_native_still_running=1951201, entries=entries,
        source_code_fact='The same velocity seed enters arm_pre=arm_hit-half_window*velocity, before the stroke begins.',
        inference='Changing the constant swing direction also changes the prepared arm pose and coupled root motion; an instantaneous contact-state rate optimum does not isolate impact speed.',
        prospective_method='Consider a time-windowed speed perturbation that is zero during preparation and at the nominal contact pose; requires separate actual trajectory and rule evaluation.',
        prospective_method_executed=False, original_physics_and_scores_unchanged=True,
        source_sha256={str(path): sha(path) for path in [*dependencies, teacher]},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with (HERE / 'ACTUAL_PREPARATION_SHIFT_CHECK.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed_generations=5, time1_5s=[dict(candidate_id=row['candidate_id'],
        racket_shift_m=row['actual_preparation_measurements'][1]['actual_racket_center_displacement_median_m'],
        root_shift_m=row['actual_preparation_measurements'][1]['actual_root_position_displacement_median_m']) for row in entries])))


if __name__ == '__main__':
    main()
