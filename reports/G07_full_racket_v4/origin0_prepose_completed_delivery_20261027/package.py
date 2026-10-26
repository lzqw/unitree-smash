"""Archive completed origin0 eight-way search, independent cold128 and current actual video."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
SEARCH = REPORTS / 'prepose_origin0_8_sequential_n128_01'
COLD = REPORTS / 'prepose_origin0_8_best_fixed_train128_01'
OWNER = REPORTS / 'prepose_origin0_after_precision_queue02'
VIDEO_ROOT = ROOT.parent / 'unitree_smash_video_framing'
VIDEO_REPORT = VIDEO_ROOT / 'reports/G07_full_racket_v4/origin0_cold_four_contacts_video_20261027'
VIDEO = VIDEO_ROOT / 'reports/G07_full_racket_v4/video/Current_effect_origin0_cold128_four_air_contacts_and_miss_20261027.mp4'
sys.path.insert(0, str(ROOT))
import numpy as np
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = HERE / 'G07_origin0_prepose8_completed_search_cold128_and_four_contacts_video_20261027.zip'
    assert not archive.exists() and not Path('/proc/1884350').exists()
    assert read(OWNER / 'state.json')['state'] == 'COMPLETED_PREPOSE_ORIGIN0_SEARCH_AND_COLD128'
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json')
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    cold_summary = summarize_training(read(COLD / 'episodes.json'), cases, indices, protocol)
    assert cold_summary == read(OWNER / 'COLD_TRAIN128_RESULT.json')
    assert cold_summary == read(VIDEO_REPORT / 'NATIVE_EPISODE_CHECK.json')['cold_summary']
    history = read(SEARCH / 'metrics.json')['history']
    assert len(history) == 8
    summaries, actual_steps = [], 0
    for candidate, entry in enumerate(history):
        rows = read(SEARCH / f'episodes_{candidate:03d}.json')
        initial = read(SEARCH / f'initial_ground_states_{candidate:03d}.json')
        assert len(rows) == 128 and not np.asarray(initial['actual_environment_origins_m']).any()
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == candidate
            assert row['episode_index'] == initial['episode_ids'][lane] and row['ground_init']
            assert row['source_training_index'] == lane and row['case_id'] == cases[lane]['case_id']
        summary = summarize_candidate(rows, protocol)
        assert summary == entry['summary'] and summary['numerical_failures'] == 0
        summaries.append(summary)
        actual_steps += entry['env_steps']
        with np.load(SEARCH / f'all_first_episodes_{candidate:03d}.npz') as arrays:
            assert all(np.isfinite(arrays[key]).all() for key in ['link_poses', 'ball_position_m', 'episode_clock_s'])
    assert actual_steps == 566784
    for run, cap in [(SEARCH, 819200), (COLD, 102400)]:
        job = read(run / 'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
        assert job['charged_env_steps'] == cap and read(run / 'completed.json')['env_steps'] <= cap
    native_check = read(VIDEO_REPORT / 'NATIVE_EPISODE_CHECK.json')
    decode = read(VIDEO_REPORT / 'DECODE_AND_NATIVE_INPUT_CHECK.json')
    visual = read(VIDEO_REPORT / 'VISUAL_REVIEW.json')
    assert decode['all_frames_decoded'] and decode['all_displayed_poses_balls_clocks_bitwise_match_native']
    assert visual['review_completed'] and not visual['material_visual_defects_found']
    assert decode['sha256'] == sha(VIDEO)
    search_receipt = read(OWNER / 'SEARCH_RECEIPT.json')
    audit = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_search_rows=1024,
        actual_cold_rows=128, search_candidate_summaries=summaries, cold_original128_summary=cold_summary,
        actual_search_control_env_steps=actual_steps, conservatively_charged_search_env_steps=819200,
        actual_cold_control_env_steps=102400, actual_best_policy_sha256=sha(SEARCH / 'best_candidate.pt'),
        actual_valid_cold_public_rows=native_check['actual_valid_public_rows'],
        CPU_vs_actual_cold_GPU_max_action_error=native_check['CPU_vs_actual_GPU_max_action_error'],
        original_action_tolerance=5e-4, completed_search_command_receipt_sha256=sha(OWNER / 'SEARCH_RECEIPT.json'),
        search_public_replay_receipt=search_receipt, current_video_sha256=sha(VIDEO),
        current_video_duration_s=decode['duration_s'], all_native_pose_streams_finite=True,
        cold_diagnoses=native_check['selected_episode_diagnoses'], original_origin_spacing_m=0.,
        no_old_origin16_or_search_performance_inherited=True,
        formal_cases_read=False, formal_acceptance=False)
    (HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json').write_text(json.dumps(audit, indent=2) + '\n')
    payload, omitted = {}, []
    for label, run in [('search8', SEARCH), ('cold128', COLD)]:
        for path in sorted(run.iterdir()):
            if not path.is_file() or path.suffix in ['.lock', '.tmp']:
                continue
            if '.partial.' in path.name:
                final = path.with_name(path.name.replace('.partial.', '.'))
                if final.exists() and sha(final) == sha(path):
                    omitted.append(str(path))
                    continue
            payload[label + '/' + path.name] = path
    for name in ['run.py', 'PLAN.json', 'ADMISSION_DECISION.json', 'ACTUAL_PROPOSALS.json',
                 'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json', 'SEARCH_RECEIPT.json',
                 'COLD_TRAIN128_RESULT.json', 'COLD128_EPISODES.csv', 'COLD128_EPISODES.json',
                 'BEST_MEASURED_SOURCE.json', 'state.json']:
        path = OWNER / name
        if path.exists():
            payload['coordinator/' + name] = path
    for name in ['ACTUAL_PROPOSALS.json', 'ACTUAL_SELECTED_SOURCE_COMMAND_CHECK.json', 'ADMISSION_DECISION.json']:
        path = REPORTS / 'gain_body_after_prepose_queue02' / name
        payload['next_method_admission/' + name] = path
    for path in sorted(VIDEO_REPORT.rglob('*')):
        if not path.is_file() or 'snapshot' in path.relative_to(VIDEO_REPORT).parts:
            continue
        payload['video_evidence/' + str(path.relative_to(VIDEO_REPORT))] = path
    for path in [VIDEO, VIDEO.with_suffix('.json')]:
        payload['video/' + path.name] = path
    for path in VIDEO.parent.glob(VIDEO.stem + '_case*.jpg'):
        payload['video/' + path.name] = path
    for name in ['render_full_racket_latest_effect_v4.py', 'render_full_racket_r3_effect.py',
                 'native_video_framing_v4.py', 'render_current_v4_progress.py']:
        payload['video_program/scripts/' + name] = VIDEO_ROOT / 'scripts' / name
    for name in ['g07_full_racket_eval_v4_r3_origin0.json', 'g07_full_racket_eval_v4_r3.json',
                 'court_net_v4.json', 'shuttle_physics_v2.json']:
        payload['program/configs/' + name] = ROOT / 'configs' / name
    for path in sorted((REPORTS / 'origin_precision_after_aim_queue01').iterdir()):
        if path.is_file() and path.suffix in ['.py', '.json', '.md']:
            payload['precision_admission/' + path.name] = path
    precision = read(REPORTS / 'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json')
    assert precision['admitted']
    for source, digest in precision['source_sha256'].items():
        path = Path(source)
        assert sha(path) == digest
        payload['precision_probe/' + str(path.relative_to(REPORTS))] = path
    for path in [Path(__file__), HERE / 'ACTUAL_COMPLETED_DATA_CHECK.json', HERE / 'REPRODUCTION.md']:
        payload['delivery/' + path.name] = path
    records = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in payload.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), payloads=records,
        source_scope='Completed origin0 eight-way search1024 plus independent cold128, all original cases retained; actual current cold video.',
        best_policy_file='search8/best_candidate.pt', best_policy_sha256=sha(SEARCH / 'best_candidate.pt'),
        original128_cold_summary=cold_summary, omitted_byte_identical_partial_files=omitted,
        runtime_dependencies='Native source snapshots included. Full racket/robot USD and mesh runtime companion plus pinned external IsaacLab/IsaacSim/Torch are required. Restore original absolute paths or audit relocation; this ZIP is not a standalone simulator.',
        current_video_file='video/' + VIDEO.name, formal_cases_read=False, formal_acceptance=False)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for name, path in payload.items():
            zipped.write(path, name)
        zipped.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(payload) + 1
        for name, record in records.items():
            content = zipped.read(name)
            assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive),
        bytes=archive.stat().st_size, sha256=sha(archive), members=len(payload) + 1,
        payloads_SHA_verified=len(payload), all_CRC_passed=True, formal_acceptance=False)
    (HERE / 'ARCHIVE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
