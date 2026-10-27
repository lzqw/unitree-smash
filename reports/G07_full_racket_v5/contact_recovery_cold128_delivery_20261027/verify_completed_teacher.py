"""Independently inspect completed v5 cold128 without changing its scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT.parent / 'unitree_smash_contact_recovery'
sys.path.insert(0, str(WORK))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02'
    old = ROOT / 'reports/G07_full_racket_v4/gain_body8_best_fixed_train128_01'
    queue = ROOT / 'reports/G07_full_racket_v5/contact_recovery_queue02'
    read = lambda path: json.loads(path.read_text())
    protocol = read(WORK / PROTOCOL)
    protocol['_root'] = str(WORK)
    cases, indices = training_cases(WORK, protocol)
    rows = read(run / 'episodes.json')
    summary = summarize_training(rows, cases, indices, protocol)
    metrics = read(run / 'metrics.json')
    assert all(value == metrics[key] for key, value in summary.items())
    assert (summary['complete'], summary['airborne_contacts'], summary['conditional_recovery_count'],
            summary['numerical_failures']) == (0, 4, 0, 0)
    job = read(run / 'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
    assert job['charged_env_steps'] == read(run / 'completed.json')['env_steps'] == 102400
    assert not any(Path('/proc', str(job[key])).exists() for key in ['pid', 'child_pid'])
    old_rows = read(old / 'episodes.json')
    differing = []
    for index, (row, previous) in enumerate(zip(rows, old_rows)):
        stripped = {key: value for key, value in row.items() if key != 'contact_flight_recovery'}
        if json.dumps(stripped, sort_keys=True) != json.dumps(previous, sort_keys=True):
            differing.append(index)
    with np.load(run / 'all_first_episodes.npz') as source:
        new_arrays = {key: source[key].copy() for key in source.files}
    with np.load(old / 'all_first_episodes.npz') as source:
        old_arrays = {key: source[key].copy() for key in source.files}
    common = new_arrays['valid_first_episode'] & old_arrays['valid_first_episode']
    differences = {}
    for key in ['link_poses', 'ball_position_m', 'episode_clock_s']:
        a, b = new_arrays[key][common], old_arrays[key][common]
        differences[key] = dict(exactly_identical=bool(np.array_equal(a, b)),
                               maximum_absolute_difference=float(np.max(abs(a.astype(float) - b.astype(float)))))
    expected = read(queue / 'PLAN.json')['source_sha256']
    assert all(sha(Path(name)) == digest for name, digest in expected.items())
    commands = read(queue / 'TEACHER_REVISED_COLD128_CPU_COMMANDS.json')
    assert commands['actual_valid_public_rows'] == 1600
    assert commands['CPU_vs_actual_GPU_max_action_error'] <= 5e-4
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_completed_new_native=True,
        independently_recomputed_summary=summary, actual_gpu_hours=job['charged_gpu_hours'],
        actual_control_environment_steps=102400, original128_rows_differing_after_removing_only_new_association=differing,
        actual_50Hz_frames=len(new_arrays['link_poses']), common_valid_case_frames=int(common.sum()),
        validity_masks_identical=bool(np.array_equal(new_arrays['valid_first_episode'], old_arrays['valid_first_episode'])),
        recorded_motion_comparison=differences,
        bound_contact_flight_cases=[row['source_training_index'] for row in rows if row['contact_flight_recovery']['bound']],
        qualified_contact_flight_landings=[row['source_training_index'] for row in rows if row['contact_flight_recovery']['qualified_contact_landing']],
        recovered_cases=[row['source_training_index'] for row in rows if row['contact_flight_recovery']['success']],
        actual_source_CPU_command_check=commands, held_source_count=len(expected), all_held_bytes_unchanged=True,
        source_policy_sha256=read(run / 'INPUTS.json')['policy_sha256'],
        protocol_sha256=sha(WORK / PROTOCOL), source_script_sha256=sha(Path(__file__)),
        scoring_revision_does_not_certify_new_policy_improvement=True, formal_cases_read=False, formal_acceptance=False)
    with (HERE / 'ACTUAL_COMPLETED_COLD128_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['actual_gpu_hours', 'common_valid_case_frames',
        'original128_rows_differing_after_removing_only_new_association', 'recorded_motion_comparison',
        'bound_contact_flight_cases', 'qualified_contact_flight_landings', 'recovered_cases']}))


if __name__ == '__main__':
    main()
