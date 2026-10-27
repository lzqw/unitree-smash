"""Analyze recorded native metric samples without changing scoring or execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'reports/G07_full_racket_v4'
REPLAY = BASE / 'recovery_sequence_replay7_n128_01'
ORIGINAL = BASE / 'origin0_residual_fresh_n128_updates15_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    comparison = json.loads((BASE / 'recovery_sequence_observer_20261027/ACTUAL_REPLAY_COMPARISON.json').read_text())
    assert comparison['initial_model_tensors_identical']
    assert comparison['first128_all_raw_rows_identical']
    observation = json.loads((REPLAY / 'recovery_sequence/OBSERVATION.json').read_text())
    columns = observation['columns']
    results = []
    for episode in observation['episodes']:
        path = REPLAY / 'recovery_sequence' / episode['data_file']
        assert sha(path) == episode['data_sha256']
        with np.load(path) as data:
            values = data['values']
            frames = data['actual_physics_frames']
            assert data['columns'].tolist() == columns
        assert len(values) == episode['samples']
        assert np.all(np.diff(frames) > 0)
        assert np.all(np.diff(values[:, columns.index('jump_clock_s')]) > 0)

        def col(name):
            return values[:, columns.index(name)]

        def row(index):
            result = {'sample_index': int(index), 'actual_physics_frame': int(frames[index])}
            for name in columns:
                value = float(values[index, columns.index(name)])
                result[name] = value if np.isfinite(value) else None
            return result

        hit = np.flatnonzero(col('airborne_string_hit') > 0)
        assert len(hit)
        hit_index = int(hit[0])
        assert col('flight_candidate')[hit_index] == 1
        assert col('flight_qualified')[hit_index] == 1
        takeoff = float(col('takeoff_time_s')[hit_index])
        contact = float(col('actual_contact_time_s')[hit_index])
        landed = np.flatnonzero((np.arange(len(values)) > hit_index) &
                               (col('valid_jump_landed') > 0) &
                               (col('takeoff_time_s') == takeoff))
        assert len(landed)
        landed_index = int(landed[0])
        assert takeoff < contact < float(col('landing_time_s')[landed_index])
        cleared = np.flatnonzero((np.arange(len(values)) > landed_index) &
                                (col('valid_jump_landed') == 0))
        assert len(cleared)
        cleared_index = int(cleared[0])
        assert col('flight_candidate')[cleared_index] == 1
        assert col('supported_feet')[cleared_index] == 0
        assert col('flight_qualified')[cleared_index] == 0
        subsequent_landing = np.flatnonzero((np.arange(len(values)) > cleared_index) &
                                           (col('flight_candidate') == 0))
        stable = np.flatnonzero((np.arange(len(values)) > landed_index) & (col('recovery_seen') > 0))
        results.append({
            'env_id': episode['env_id'],
            'recorded_file': str(path.relative_to(ROOT)),
            'sha256': sha(path),
            'actual_samples': len(values),
            'contact_and_qualified_landing_belong_to_same_recorded_flight': True,
            'first_recorded_contact_state': row(hit_index),
            'same_flight_qualified_landing': row(landed_index),
            'first_later_takeoff_clears_landing_qualification': row(cleared_index),
            'first_later_landing': row(int(subsequent_landing[0])) if len(subsequent_landing) else None,
            'first_later_stable_window': row(int(stable[0])) if len(stable) else None,
            'terminal_record': episode['closure']['actual_jump_record'],
            'last_recorded_state': row(len(values) - 1),
        })

    original_traces = {p.name: p for p in (ORIGINAL / 'trajectory_traces').glob('*.npz')}
    replay_traces = {p.name: p for p in (REPLAY / 'trajectory_traces').glob('*.npz')}
    checked = []
    different = []
    for name in sorted(original_traces.keys() & replay_traces.keys()):
        with np.load(original_traces[name]) as old, np.load(replay_traces[name]) as new:
            same = set(old.files) == set(new.files)
            for key in old.files if same else ():
                a, b = old[key], new[key]
                equal_nan = a.dtype.kind in 'fc'
                same = same and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=equal_nan)
            checked.append(name)
            if not same:
                different.append(name)
    job = json.loads((REPLAY / 'job.json').read_text())
    result = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'source_script_sha256': sha(Path(__file__)),
        'actual_replay_job_succeeded': job['status'] == 'SUCCEEDED' and job['exit_code'] == 0,
        'actual_charged_gpu_hours': job['charged_gpu_hours'],
        'actual_charged_env_steps': job['charged_env_steps'],
        'original_first128_all_raw_rows_identical': True,
        'original_initial_model_tensors_identical': True,
        'common_original_and_replay_trace_files_checked': checked,
        'common_trace_files_with_differences': different,
        'original_only_trace_files': sorted(original_traces.keys() - replay_traces.keys()),
        'replay_only_trace_files': sorted(replay_traces.keys() - original_traces.keys()),
        'current_jump_metrics_source_sha256': sha(ROOT / 'quadsmash/tasks/stand/jump_metrics.py'),
        'episodes': results,
        'finding': 'Both actual contact flights qualify and land; a later unsupported sample clears valid_jump_landed. TRAIN69 subsequently attains the unchanged stable window without body contact, but final qualified_jump_landed remains false. TRAIN112 still has body contact and no recovery.',
        'scoring_changed': False,
        'historical_results_rewritten': False,
        'full_motion_recorded_for_selected_69_and_112': False,
        'causal_effect_on_policy_learning_proven': False,
        'formal_cases_read': False,
        'formal_acceptance': False,
        'goal_complete': False,
    }
    output = HERE / 'ACTUAL_SEQUENCE_ANALYSIS.json'
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ['actual_replay_job_succeeded', 'original_first128_all_raw_rows_identical', 'common_original_and_replay_trace_files_checked', 'common_trace_files_with_differences', 'finding']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
