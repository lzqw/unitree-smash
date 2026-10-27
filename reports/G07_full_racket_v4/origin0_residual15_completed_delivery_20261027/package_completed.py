"""Archive actual completed residual15 learning, separate frozen cold128 and its verified video."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
TRAIN = REPORTS / 'origin0_residual_fresh_n128_updates15_01'
COLD = REPORTS / 'origin0_residual_fresh15_fixed_train128_01'
OWNER = REPORTS / 'origin0_residual_after_gain_body_queue01'
VIDEO_ROOT = ROOT.parent / 'unitree_smash_video_framing'
VIDEO_REPORT = VIDEO_ROOT / 'reports/G07_full_racket_v4/origin0_residual15_cold_video_20261027'
VIDEO = VIDEO_ROOT / 'reports/G07_full_racket_v4/video/Current_effect_origin0_residual15_fixed_cold128_20261027.mp4'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode
from scripts.initialize_centered_policy_v4 import same_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    assert Path.cwd() == ROOT and not (HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json').exists()
    assert not Path('/proc/1896271').exists()
    assert read(OWNER / 'state.json')['state'] == 'COMPLETED_ORIGIN0_RESIDUAL15_AND_COLD128'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    cold_rows, metrics = read(COLD / 'episodes.json'), read(COLD / 'metrics.json')
    summary = summarize_training(cold_rows, cases, indices, protocol)
    assert summary == read(OWNER / 'COLD_TRAIN128_RESULT.json')
    assert all(metrics[k] == v for k, v in summary.items())
    assert summary['episodes'] == 128 and summary['numerical_failures'] == 0
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    source_path = REPORTS / 'gain_body8_sequential_n128_01/best_candidate.pt'
    assert same_tree(policy['base_controller'], torch.load(source_path, map_location='cpu', weights_only=False))
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 15 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 245760
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert int(native['model_state_dict']['actor_obs_normalizer.count']) == 245760
    assert int(native['model_state_dict']['critic_obs_normalizer.count']) == 245760
    rows = read(TRAIN / 'episodes.json')
    seen, checked = [0] * 128, []
    for row in rows:
        lane = row['env_id']
        assert row['ground_init'] and row['source_training_index'] == lane and row['episode_index'] == seen[lane]
        seen[lane] += 1
        checked.append(independent_episode(row, protocol))
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist()
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == read(TRAIN / 'metrics.json')['counts'] == read(OWNER / 'TRAINING_RECEIPT.json')['independent_changing_policy_counts']
    assert counts['numerical_failures'] == 0
    inputs = read(COLD / 'INPUTS.json')
    assert inputs['policy_sha256'] == metrics['policy_sha256'] == sha(TRAIN / 'policy.pt')
    assert inputs['training_indices'] == list(range(128)) and not any(inputs['reset_noise_amplitudes'].values())
    assert inputs['reset_noise_seed'] == 724201 and inputs['protocol_sha256'] == sha(protocol_path)
    assert not np.asarray(read(COLD / 'origin_layout.json')['actual_environment_origins_m']).any()
    gpu_hours = 0.
    for run, cap in [(TRAIN, 245760), (COLD, 102400)]:
        job = read(run / 'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
        assert job['charged_env_steps'] == read(run / 'completed.json')['env_steps'] == cap
        assert not Path('/proc/' + str(job['pid'])).exists() and not Path('/proc/' + str(job['child_pid'])).exists()
        gpu_hours += job['charged_gpu_hours']
        hashes = read(run / 'source_hashes.json')
        with zipfile.ZipFile(run / 'source_snapshot.zip') as zipped:
            assert set(zipped.namelist()) == set(hashes) and zipped.testzip() is None
            for name, digest in hashes.items():
                assert hashlib.sha256(zipped.read(name)).hexdigest() == digest and sha(ROOT / name) == digest
    commands = read(OWNER / 'COLD_CPU_COMMAND_CHECK.json')
    assert commands['CPU_vs_actual_GPU_max_action_error'] <= 5e-4
    video_check = read(VIDEO_REPORT / 'DECODE_AND_NATIVE_INPUT_CHECK.json')
    visual = read(VIDEO_REPORT / 'VISUAL_REVIEW.json')
    assert video_check['all_frames_decoded'] and video_check['all_displayed_poses_balls_clocks_bitwise_match_native']
    assert visual['review_completed'] and not visual['material_visual_defects_found']
    assert video_check['sha256'] == sha(VIDEO)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_completed_updates=15,
        actual_training_control_env_steps=245760, independently_checked_changing_policy_counts=counts,
        actual_cold_control_env_steps=102400, original128_cold_summary=summary,
        actual_combined_GPU_hours=gpu_hours, all128_training_episode_chains_checked=True,
        canonical_checkpoint_sha256=sha(TRAIN / 'checkpoint.pt'), actual_policy_sha256=sha(TRAIN / 'policy.pt'),
        embedded_original_source_tree_unchanged=True,
        actual_valid_cold_public_rows=commands['actual_valid_public_rows'],
        CPU_vs_actual_cold_GPU_max_action_error=commands['CPU_vs_actual_GPU_max_action_error'],
        original_action_tolerance=5e-4, source_snapshot_member_counts={p.name: len(read(p / 'source_hashes.json')) for p in [TRAIN, COLD]},
        video_sha256=sha(VIDEO), video_duration_s=video_check['duration_s'],
        new_policy_not_promoted_over_earlier4_air_teacher=True,
        formal_cases_read=False, formal_acceptance=False)
    save(HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json', receipt)
    payload, omitted = {}, []
    for label, directory in [('training15', TRAIN), ('cold128', COLD)]:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix in ['.lock', '.tmp', '.pyc'] or '__pycache__' in path.parts:
                continue
            if '.partial.' in path.name:
                final = path.with_name(path.name.replace('.partial.', '.'))
                if final.is_file() and sha(final) == sha(path):
                    omitted.append(str(path))
                    continue
            payload[label + '/' + str(path.relative_to(directory))] = path
    for name in ['run.py', 'PLAN.json', 'ADMISSION_DECISION.json', 'ACTUAL_SOURCE_CPU/CPU_CHECK.json',
                 'ACTUAL_UPDATE_01.json', 'ACTUAL_UPDATE_07.json', 'ACTUAL_UPDATE_15.json',
                 'TRAINING_RECEIPT.json', 'COLD_TRAIN128_RESULT.json', 'COLD_CPU_COMMAND_CHECK.json',
                 'COLD128_EPISODES.csv', 'COLD128_EPISODES.json', 'MEASURED_RESULT.json', 'state.json']:
        payload['coordinator/' + name] = OWNER / name
    for path in sorted(VIDEO_REPORT.rglob('*')):
        if path.is_file() and 'snapshot' not in path.relative_to(VIDEO_REPORT).parts and '__pycache__' not in path.parts:
            payload['video_evidence/' + str(path.relative_to(VIDEO_REPORT))] = path
    for path in [VIDEO, VIDEO.with_suffix('.json')]:
        payload['video/' + path.name] = path
    for path in VIDEO.parent.glob(VIDEO.stem + '_case*.jpg'):
        payload['video/' + path.name] = path
    for name in ['render_full_racket_latest_effect_v4.py', 'render_full_racket_r3_effect.py',
                 'native_video_framing_v4.py', 'render_current_v4_progress.py']:
        payload['video_program/scripts/' + name] = VIDEO_ROOT / 'scripts' / name
    for path in [source_path, protocol_path, ROOT / 'configs/court_net_v4.json',
                 ROOT / 'configs/evaluation_protocol_v2.json', ROOT / protocol['training_suite']['path']]:
        payload['reference/' + str(path.relative_to(ROOT))] = path
    for name in ['package_completed.py', 'ACTUAL_COMPLETED_DATA_CHECK.json', 'REPRODUCTION.md']:
        payload['delivery/' + name] = HERE / name
    records = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in payload.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=records,
        scope='Completed15 changing-policy learning plus independent fixed cold128 and actual current video; original cases retained.',
        original128_cold_summary=summary, actual_training_counts=counts,
        current_policy_file='training15/policy.pt', canonical_checkpoint_file='training15/checkpoint.pt',
        current_policy_sha256=sha(TRAIN / 'policy.pt'), original_source_reference='reference/' + str(source_path.relative_to(ROOT)),
        omitted_byte_identical_partial_files=omitted,
        runtime_dependencies='Actual source/config/USD snapshots included; previously published full runtime/mesh/walking companion and pinned external IsaacLab/IsaacSim/Torch remain required. Restore paths or audit relocation.',
        formal_cases_read=False, formal_acceptance=False)
    archive_path = HERE / 'G07_origin0_residual15_completed_training_cold128_and_current_video_20261027.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name, path in payload.items():
            assert sha(path) == records[name]['sha256']
            zipped.write(path, name)
        zipped.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive_path) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(records) + 1
        for name, value in records.items():
            content = zipped.read(name)
            assert len(content) == value['bytes'] and hashlib.sha256(content).hexdigest() == value['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive_path),
        bytes=archive_path.stat().st_size, sha256=sha(archive_path), members=len(records) + 1,
        payloads_SHA_verified=len(records), all_CRC_passed=True, formal_acceptance=False)
    save(HERE / 'ARCHIVE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
