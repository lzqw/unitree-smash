"""Seal the completed eight-way native search and verify the actual cold handoff."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from scripts.initialize_centered_policy_v4 import same_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'RESULT.json').exists()
    base = ROOT / 'reports/G07_full_racket_v4'
    search = base / 'gain_body8_sequential_n128_01'
    cold = base / 'gain_body8_best_fixed_train128_01'
    owner = base / 'gain_body_after_prepose_queue02'
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json')
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    metrics, receipt = read(search / 'metrics.json'), read(owner / 'SEARCH_RECEIPT.json')
    assert len(metrics['history']) == 8 and receipt['all1024_native_rows_independently_checked']
    summaries, steps = [], 0
    for i, entry in enumerate(metrics['history']):
        assert entry['candidate_id'] == i
        rows = read(search / f'episodes_{i:03d}.json')
        initial = read(search / f'initial_ground_states_{i:03d}.json')
        assert len(rows) == 128 and initial['all_clocks_zero']
        assert not np.asarray(initial['actual_environment_origins_m']).any()
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == i
            assert row['episode_index'] == initial['episode_ids'][lane]
            assert row['case_id'] == cases[lane]['case_id'] and row['source_training_index'] == indices[lane]
            assert row['ground_init']
        summary = summarize_candidate(rows, protocol)
        assert summary == entry['summary'] and summary['numerical_failures'] == 0
        replay = receipt['actual_public_replays'][i]
        assert replay['candidate_id'] == i and replay['summary'] == summary
        assert replay['CPU_vs_actual_GPU_max_action_error'] <= 5e-4
        summaries.append(summary)
        steps += entry['env_steps']
    assert steps == read(search / 'completed.json')['env_steps'] == receipt['actual_control_env_steps']
    job = read(search / 'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
    assert job['charged_env_steps'] == 819200 and steps <= job['charged_env_steps']
    hashes = read(search / 'source_hashes.json')
    with zipfile.ZipFile(search / 'source_snapshot.zip') as zipped:
        assert zipped.testzip() is None and set(zipped.namelist()) == set(hashes)
        assert all(hashlib.sha256(zipped.read(name)).hexdigest() == digest for name, digest in hashes.items())
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    policy_path = search / 'best_candidate.pt'
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    source_path = base / 'prepose_origin0_8_sequential_n128_01/best_candidate.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert same_tree(policy['source_public_policy'], source) and torch.equal(policy['parameters'], source['parameters'])
    selected = max(range(8), key=lambda i: tuple(summaries[i]['rank']))
    assert selected == receipt['best_candidate_id'] == policy['actual_native_search']['candidate_id'] == 0
    assert policy['residual_gains_outer_to_inner'] == [[1., 1.], [1., 1.]] and policy['runtime_body_deltas'] == {}
    assert policy['native_execution'] and not policy['performance_inherited']
    cold_inputs = read(cold / 'INPUTS.json')
    assert cold_inputs['policy_sha256'] == sha(policy_path)
    assert cold_inputs['origin_spacing_m'] == 0 and cold_inputs['training_indices'] == list(range(128))
    assert cold_inputs['reset_noise_seed'] == 724201 and not any(cold_inputs['reset_noise_amplitudes'].values())
    assert not np.asarray(read(cold / 'origin_layout.json')['actual_environment_origins_m']).any()
    initial = read(cold / 'initial_states.json')
    assert len(initial) == 128
    for i, row in enumerate(initial):
        assert row['env_id'] == i
        state = np.asarray(row['root_state_local'])
        assert np.isfinite(state).all() and np.max(abs(state[:3] - protocol['initialization']['root_position_m'])) < 1e-5
        assert np.max(abs(state[3:7] - [1, 0, 0, 0])) < 1e-5 and np.max(abs(state[7:])) < 1e-5
        assert np.max(np.abs(row['joint_velocity_rad_s'])) == 0
    assert not Path('/proc/1903046').exists() and not Path('/proc/1903032').exists()
    processes = {}
    for pid, needle in [(1920441, 'run_gpu_job.py'), (1920455, 'evaluate_full_racket_gain_body_origin0_train_v4.py')]:
        process = Path('/proc') / str(pid)
        command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert needle in command and str(cold) in command
        environment = dict(x.split(b'=', 1) for x in (process / 'environ').read_bytes().split(b'\0') if b'=' in x)
        assert environment[b'CUDA_VISIBLE_DEVICES'] == b'0'
        processes[str(pid)] = dict(command=command, actual_GPU_visible='0', stat=(process / 'stat').read_text())
    held = read(base / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    for target, source_file in [('SEARCH_JOB.json', search / 'job.json'),
        ('SEARCH_COMPLETION.json', search / 'completed.json'), ('SEARCH_METRICS.json', search / 'metrics.json'),
        ('SEARCH_RECEIPT.json', owner / 'SEARCH_RECEIPT.json')]:
        shutil.copy2(source_file, HERE / target)
    sources = [Path(__file__), policy_path, source_path, search / 'INPUTS.json', search / 'source_hashes.json',
               search / 'source_snapshot.zip', cold / 'INPUTS.json', cold / 'origin_layout.json', cold / 'initial_states.json']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), original_first_episode_rows_checked=1024,
        original_case_and_actual_episode_ids_retained=True, candidate_summaries=summaries,
        actual_control_env_steps=steps, conservatively_charged_env_steps=job['charged_env_steps'],
        charged_gpu_hours=job['charged_gpu_hours'], source_snapshot_CRC_and_SHA_members_checked=len(hashes),
        actual_valid_public_rows=sum(x['actual_valid_public_rows'] for x in receipt['actual_public_replays']),
        maximum_CPU_vs_actual_GPU_action_error=max(x['CPU_vs_actual_GPU_max_action_error'] for x in receipt['actual_public_replays']),
        selected_candidate_id=selected, selected_policy_sha256=sha(policy_path),
        embedded_source_checkpoint_tree_unchanged=True, actual_cold_initial_states_checked=128,
        actual_cold_processes=processes, old_search_guard_and_native_actually_exited=True,
        held482_unchanged=True, cold_performance_pending=True,
        source_sha256={str(path): sha(path) for path in sources},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['original_first_episode_rows_checked', 'actual_control_env_steps',
        'charged_gpu_hours', 'actual_valid_public_rows', 'maximum_CPU_vs_actual_GPU_action_error',
        'selected_candidate_id', 'selected_policy_sha256', 'actual_cold_initial_states_checked']}), flush=True)


if __name__ == '__main__':
    main()
