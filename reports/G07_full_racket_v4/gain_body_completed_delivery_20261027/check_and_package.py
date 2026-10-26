"""Verify completed gain/body search and its separate cold128, then archive actual evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
SEARCH = REPORTS / 'gain_body8_sequential_n128_01'
COLD = REPORTS / 'gain_body8_best_fixed_train128_01'
OWNER = REPORTS / 'gain_body_after_prepose_queue02'
OLD_COLD = REPORTS / 'prepose_origin0_8_best_fixed_train128_01'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_gain_body_adaptation_v4 import make_rally_controller
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from scripts.initialize_centered_policy_v4 import same_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert Path.cwd() == ROOT
    assert not (HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json').exists()
    assert not Path('/proc/1895642').exists(), 'Wait for the actual body owner to exit'
    assert read(OWNER / 'state.json')['state'] == 'COMPLETED_GAIN_BODY_ORIGIN0_SEARCH_AND_COLD128'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs = read(COLD / 'INPUTS.json')
    rows, metrics = read(COLD / 'episodes.json'), read(COLD / 'metrics.json')
    summary = summarize_training(rows, cases, indices, protocol)
    assert summary == read(OWNER / 'COLD_TRAIN128_RESULT.json')
    assert all(metrics[k] == v for k, v in summary.items())
    assert summary['episodes'] == 128 and summary['numerical_failures'] == 0
    assert inputs['training_indices'] == indices == list(range(128))
    assert inputs['protocol_sha256'] == metrics['protocol_sha256'] == sha(protocol_path)
    assert not any(inputs['reset_noise_amplitudes'].values())
    assert inputs['reset_noise_seed'] == metrics['noise_seed'] == 724201
    assert metrics['noise_level'] == 'none' and inputs['origin_spacing_m'] == 0.
    origins = np.asarray(read(COLD / 'origin_layout.json')['actual_environment_origins_m'])
    assert origins.shape == (128, 3) and not origins.any()
    initial = read(COLD / 'initial_states.json')
    assert len(initial) == len(rows) == 128
    for lane, (state, row) in enumerate(zip(initial, rows)):
        root = np.asarray(state['root_state_local'])
        assert state['env_id'] == row['env_id'] == row['jump']['env_id'] == lane
        assert row['source_training_index'] == lane and row['case_id'] == cases[lane]['case_id']
        assert row['ground_init'] and row['episode_index'] == 0
        assert np.max(abs(root[:3] - protocol['initialization']['root_position_m'])) < 1e-5
        assert np.max(abs(root[3:7] - [1., 0., 0., 0.])) < 1e-5 and not root[7:].any()
        assert not np.asarray(state['joint_velocity_rad_s']).any()
    actual_steps, search_summaries = 0, []
    for group, entry in enumerate(read(SEARCH / 'metrics.json')['history']):
        group_rows = read(SEARCH / f'episodes_{group:03d}.json')
        ground = read(SEARCH / f'initial_ground_states_{group:03d}.json')
        assert len(group_rows) == 128 and ground['all_clocks_zero']
        assert not np.asarray(ground['actual_environment_origins_m']).any()
        for lane, row in enumerate(group_rows):
            assert row['env_id'] == row['jump']['env_id'] == row['source_training_index'] == lane
            assert row['candidate_id'] == group and row['case_id'] == cases[lane]['case_id']
            assert row['episode_index'] == ground['episode_ids'][lane] and row['ground_init']
        result = summarize_candidate(group_rows, protocol)
        assert result == entry['summary'] and result['numerical_failures'] == 0
        assert entry['env_steps'] == entry['control_ticks'] * 128
        search_summaries.append(result)
        actual_steps += entry['env_steps']
    assert len(search_summaries) == 8 and actual_steps == 715136
    for run, cap in [(SEARCH, 819200), (COLD, 102400)]:
        job, completed = read(run / 'job.json'), read(run / 'completed.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
        assert not Path('/proc/' + str(job['pid'])).exists()
        assert not Path('/proc/' + str(job['child_pid'])).exists()
        assert job['charged_env_steps'] == cap and 0 < completed['env_steps'] <= cap
        expected_steps = actual_steps if run == SEARCH else metrics['actual_control_env_steps']
        assert completed['env_steps'] == expected_steps
        source_hashes = read(run / 'source_hashes.json')
        with zipfile.ZipFile(run / 'source_snapshot.zip') as archive:
            assert archive.testzip() is None and set(archive.namelist()) == set(source_hashes)
            for name, digest in source_hashes.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
                assert sha(ROOT / name) == digest, name
    policy_path = SEARCH / 'best_candidate.pt'
    assert sha(policy_path) == inputs['policy_sha256'] == metrics['policy_sha256']
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    prior_path = REPORTS / 'prepose_origin0_8_sequential_n128_01/best_candidate.pt'
    prior = torch.load(prior_path, map_location='cpu', weights_only=False)
    assert same_tree(policy['source_public_policy'], prior)
    search_inputs = read(SEARCH / 'INPUTS.json')
    default, limits = [torch.tensor(search_inputs[k], dtype=torch.float32)
                       for k in ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(COLD / 'public_and_actions_first2.npz') as z:
        public, actions, valid = [z[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
    controller = make_rally_controller(policy, 2, 'cpu')
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
    assert maximum <= 5e-4 and valid.sum() > 0
    comparison = {}
    with np.load(COLD / 'all_first_episodes.npz') as current, np.load(OLD_COLD / 'all_first_episodes.npz') as prior_arrays:
        n = min(len(current['valid_first_episode']), len(prior_arrays['valid_first_episode']))
        common = current['valid_first_episode'][:n] & prior_arrays['valid_first_episode'][:n]
        comparison['shared_valid_samples'] = int(common.sum())
        comparison['all_validity_masks_identical'] = np.array_equal(current['valid_first_episode'], prior_arrays['valid_first_episode'])
        for key in ['link_poses', 'ball_position_m', 'episode_clock_s']:
            actual, old = current[key][:n][common], prior_arrays[key][:n][common]
            assert np.isfinite(current[key]).all()
            comparison[key] = dict(bitwise_identical=bool(np.array_equal(actual, old)),
                                   maximum_absolute_difference=float(np.max(abs(actual - old))))
        comparison['all_validity_masks_identical'] = bool(comparison['all_validity_masks_identical'])
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_search_rows=1024,
        actual_cold_rows=128, search_candidate_summaries=search_summaries,
        original128_cold_summary=summary, actual_search_control_env_steps=actual_steps,
        actual_cold_control_env_steps=metrics['actual_control_env_steps'],
        charged_search_env_steps_cap=819200, charged_cold_env_steps_cap=102400,
        source_policy_sha256=sha(policy_path), embedded_original_source_checkpoint_unchanged=True,
        actual_valid_cold_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, prior_cold_shared_valid_trajectory_comparison=comparison,
        prior_cold_reference_sha256={str(p): sha(p) for p in
            [OLD_COLD / 'all_first_episodes.npz', OLD_COLD / 'metrics.json', prior_path]},
        source_snapshot_member_counts={run.name: len(read(run / 'source_hashes.json')) for run in [SEARCH, COLD]},
        all_source_snapshots_CRC_and_SHA_verified=True, formal_cases_read=False, formal_acceptance=False)
    save(HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json', receipt)
    payload, omitted = {}, []
    for label, run in [('search8', SEARCH), ('cold128', COLD)]:
        for path in sorted(run.iterdir()):
            if not path.is_file() or path.suffix in ['.lock', '.tmp']:
                continue
            if '.partial.' in path.name:
                candidates = [p for p in run.glob(path.name.split('.partial.')[0] + '*.' + path.suffix[1:])
                              if p != path and '.partial.' not in p.name]
                matching = [p.name for p in candidates if sha(p) == sha(path)]
                if matching:
                    omitted.append(dict(path=str(path), byte_identical_to=matching))
                    continue
            payload[label + '/' + path.name] = path
    for name in ['run.py', 'PLAN.json', 'ADMISSION_DECISION.json', 'ACTUAL_PROPOSALS.json',
                 'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json', 'SEARCH_RECEIPT.json',
                 'COLD_TRAIN128_RESULT.json', 'COLD128_EPISODES.csv', 'COLD128_EPISODES.json',
                 'BEST_MEASURED_SOURCE.json', 'state.json']:
        payload['coordinator/' + name] = OWNER / name
    for folder in ['gain_body_search_completed_20261027', 'gain_body_timing_results_20261027',
                   'first_ground_geometry_diagnostic_20261027']:
        for path in sorted((REPORTS / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                payload['diagnostics/' + str(path.relative_to(REPORTS))] = path
    for name in ['g07_full_racket_eval_v4_r3_origin0.json', 'court_net_v4.json', 'shuttle_physics_v2.json']:
        payload['program/configs/' + name] = ROOT / 'configs' / name
    payload['program/' + protocol['training_suite']['path']] = ROOT / protocol['training_suite']['path']
    for path in [OLD_COLD / 'all_first_episodes.npz', OLD_COLD / 'metrics.json']:
        payload['prior_cold_reference/' + path.name] = path
    payload['prior_cold_reference/source_prepose_policy.pt'] = prior_path
    precision_path = REPORTS / 'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
    precision = read(precision_path)
    assert precision['admitted']
    payload['precision_admission/PRECISION_ADMISSION.json'] = precision_path
    for source, digest in precision['source_sha256'].items():
        path = Path(source)
        assert sha(path) == digest
        payload['precision_probe/' + str(path.relative_to(REPORTS))] = path
    for path in [Path(__file__), HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json', HERE / 'REPRODUCTION.md']:
        payload['delivery/' + path.name] = path
    records = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in payload.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=records,
        source_scope='Actual eight-way body/gain search1024 and separate frozen cold128; full original case coverage.',
        original128_cold_summary=summary, best_policy_file='search8/best_candidate.pt',
        best_policy_sha256=sha(policy_path), omitted_byte_identical_partial_files=omitted,
        runtime_dependencies='Actual source/config/USD snapshots included. The previously published full runtime companion and pinned external IsaacLab/IsaacSim/Torch remain required; restore original paths or audit relocation.',
        formal_cases_read=False, formal_acceptance=False)
    archive_path = HERE / 'G07_origin0_gain_body8_completed_search_and_cold128_20261027.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in payload.items():
            assert sha(path) == records[name]['sha256'], 'Input changed before archive: ' + str(path)
            archive.write(path, name)
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(records) + 1
        for name, record in records.items():
            content = archive.read(name)
            assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive_path),
        bytes=archive_path.stat().st_size, sha256=sha(archive_path), members=len(records) + 1,
        payloads_SHA_verified=len(records), all_CRC_passed=True, formal_acceptance=False)
    save(HERE / 'ARCHIVE_CHECK.json', result)
    print(json.dumps(dict(archive=result, cold_summary=summary, trajectory_comparison=comparison)), flush=True)


if __name__ == '__main__':
    main()
