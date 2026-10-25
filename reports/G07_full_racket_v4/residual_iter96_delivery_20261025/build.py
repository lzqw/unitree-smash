"""Package terminal iter96 evidence only after both actual videos are reviewed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
QUEUE = REPORTS / 'residual_resize2048_queue01'
TRAIN = REPORTS / 'residual_ppo_n2048_resume64_01'
EVALUATION = REPORTS / 'residual_ppo_fixed_train128_iter96_n2048_01'
MILESTONE = REPORTS / 'residual_n2048_first_complete_20261025'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_new(path, value):
    assert not path.exists(), 'Never overwrite preserved evidence: ' + str(path)
    temporary = path.with_name(path.name + '.tmp')
    assert not temporary.exists(), 'Inspect an existing partial write first'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def native_checks():
    spec = importlib.util.spec_from_file_location('completed_iter96_checks', QUEUE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confirm_contacts(training):
    from contact_checks import match_complete_contacts
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    required = read(MILESTONE / 'complete_episodes_update48.json')
    result = match_complete_contacts(read(TRAIN / 'episodes.json'), read(TRAIN / 'contacts.json'), protocol, required)
    assert result['independently_recomputed_episode_counts'] == training['changing_policy_counts']
    result.update(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Finalized changing-policy TRAIN completions matched to native contacts; no frozen performance.',
        original_contacts_sha256=sha(TRAIN / 'contacts.json'),
        original_episodes_sha256=sha(TRAIN / 'episodes.json'),
        preserved_complete_rows_sha256=sha(MILESTONE / 'complete_episodes_update48.json'),
        original_snapshot_pending_flags_retained_as_history=True)
    output = MILESTONE / 'FINAL_CONTACT_CONFIRMATION.json'
    if output.exists():
        previous = read(output)
        assert previous['exact_contact_record_matches']
        for key in ['original_contacts_sha256', 'original_episodes_sha256', 'preserved_complete_rows_sha256',
                    'independently_recomputed_episode_counts', 'matches']:
            assert previous[key] == result[key]
        return previous
    atomic_new(output, result)
    return result


def verify_video(video, snapshot, review_path, expected_cases, policy_sha, rows, plan_path):
    review, metadata, data, plan = map(read,
        [review_path, video.with_suffix('.json'), snapshot / 'cases.json', plan_path])
    assert review['status'] == 'APPROVED_FOR_UPLOAD'
    verified = review['verification']
    assert verified['sha256'] == sha(video) == metadata['sha256']
    assert verified['all_frames_decoded'] and verified['frames_decoded'] == metadata['frames']
    assert verified['fps'] == 50 and verified['resolution'] == [1280, 720]
    assert metadata['native_recorded_motion'] and not metadata['interpolated_or_generated_motion']
    assert not metadata['renderer_uses_gpu'] and 'llvmpipe' in metadata['renderer']
    assert data['policy_sha256'] == policy_sha and data['preselected_before_fixed_evaluation']
    assert plan['preselected_TRAIN_indices'] == expected_cases
    assert plan['evaluation_output_did_not_exist_at_registration']
    assert plan['training_progress_at_registration']['update'] < 96
    assert [c['env_id'] for c in data['cases']] == expected_cases
    assert [c['env_id'] for c in metadata['cases']] == expected_cases
    assert all(c['run'] == EVALUATION.name for c in metadata['cases'])
    assert metadata['sources'][str((snapshot / 'cases.json').relative_to(ROOT))] == sha(snapshot / 'cases.json')
    for case in metadata['cases']:
        assert json.dumps(case['row'], sort_keys=True) == json.dumps(rows[case['env_id']], sort_keys=True)
    for path, expected in metadata['sources'].items():
        assert sha(ROOT / path) == expected, 'Video source changed: ' + path
    return dict(video=str(video.relative_to(ROOT)), sha256=sha(video), frames=metadata['frames'],
        duration_s=metadata['duration_s'], preselected_TRAIN_indices=expected_cases,
        review=str(review_path.relative_to(ROOT)), predeclared_plan=str(plan_path.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=REPORTS / 'video/Current_effect_residual_PPO_iter96_20261025.mp4')
    parser.add_argument('--snapshot', type=Path, default=REPORTS / 'video_snapshot_residual_ppo_iter96_20261025')
    parser.add_argument('--review', type=Path, default=REPORTS / 'residual_iter96_video_queue01/VISUAL_REVIEW.json')
    parser.add_argument('--targeted-video', type=Path, default=REPORTS / 'video/Current_effect_residual_PPO_iter96_TRAIN58_20261025.mp4')
    parser.add_argument('--targeted-snapshot', type=Path, default=REPORTS / 'video_snapshot_residual_ppo_iter96_TRAIN58_20261025')
    parser.add_argument('--targeted-review', type=Path, default=REPORTS / 'residual_iter96_train58_video_queue01/VISUAL_REVIEW.json')
    parser.add_argument('--output', type=Path, default=REPORTS / 'G07_PPO_iter96_fixed_TRAIN128_training_videos_and_raw_20261025.zip')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-prerequisites', action='store_true')
    mode.add_argument('--confirm-training-contacts-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    videos = [(args.video.resolve(), args.snapshot.resolve(), args.review.resolve(), [19, 23, 107, 0, 8],
               REPORTS / 'residual_iter96_video_queue01/PREDECLARED_PLAN.json'),
              (args.targeted_video.resolve(), args.targeted_snapshot.resolve(), args.targeted_review.resolve(), [58],
               REPORTS / 'residual_iter96_train58_video_queue01/PREDECLARED_PLAN.json')]
    prerequisites = [TRAIN / name for name in ['job.json', 'completed.json', 'metrics.json', 'checkpoint.pt',
        'policy.pt', 'episodes.json', 'contacts.json', 'initial_ground_states.json', 'source_snapshot.zip', 'INPUTS.json']]
    prerequisites.append(MILESTONE / 'complete_episodes_update48.json')
    training_prerequisites = list(prerequisites)
    prerequisites += [EVALUATION / name for name in ['job.json', 'completed.json', 'metrics.json', 'episodes.json',
        'contacts.json', 'all_first_episodes.npz', 'geometry.json', 'geometry.npz', 'initial_states.json',
        'source_snapshot.zip', 'asset_dependencies.json', 'INPUTS.json']]
    prerequisites += [QUEUE / (stem + suffix) for stem in ['iter96_training_episodes', 'iter96_fixed_episodes']
                      for suffix in ['.json', '.csv']]
    for video, snapshot, review, _, plan in videos:
        prerequisites += [video, video.with_suffix('.json'), snapshot / 'cases.json', review, plan]
    missing = [str(p) for p in (training_prerequisites if args.confirm_training_contacts_only else prerequisites)
               if not p.is_file()]
    if args.check_prerequisites:
        print(json.dumps(dict(prerequisite_files_present=not missing, missing=missing,
            completed_run_content_validated=False, archive_created=False, archive_exists=args.output.exists(),
            current_training_contacts_confirmed=False, formal_acceptance=False)), flush=True)
        return
    if missing:
        raise RuntimeError('Actual native/video/CSV prerequisites still pending: ' + repr(missing))
    checks = native_checks()
    core = checks.core
    core.verify_job(TRAIN, checks.STEPS)
    training = checks.verify_training()
    if args.confirm_training_contacts_only:
        confirmation = confirm_contacts(training)
        print(json.dumps(dict(matched_complete_episodes=confirmation['matched_complete_episodes'],
            exact_contact_record_matches=True, archive_created=False, formal_acceptance=False)))
        return
    assert not any(p.exists() for p in [args.output, HERE / 'MANIFEST.json', HERE / 'ARCHIVE.json']), 'Preserve previous deliveries'
    metrics = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    assert metrics['episodes'] == 128 and metrics['numerical_failures'] == 0
    confirmation = confirm_contacts(training)
    policy_sha = sha(TRAIN / 'policy.pt')
    rows = read(EVALUATION / 'episodes.json')
    video_results = [verify_video(video, snapshot, review, cases, policy_sha, rows, plan)
        for video, snapshot, review, cases, plan in videos]
    csv_manifests = []
    for source, stem in [(TRAIN, 'iter96_training_episodes'), (EVALUATION, 'iter96_fixed_episodes')]:
        manifest = read(QUEUE / (stem + '.json'))
        assert manifest['original_runtime_metrics_match'] and not manifest['formal_acceptance']
        assert manifest['source_episode_sha256'] == sha(source / 'episodes.json')
        assert manifest['csv_sha256'] == sha(QUEUE / (stem + '.csv'))
        key = 'final_exported_policy_sha256' if source == TRAIN else 'assessed_policy_sha256'
        assert manifest[key] == policy_sha
        assert manifest['rows'] == (training['changing_policy_counts']['completed_episodes'] if source == TRAIN else 128)
        csv_manifests.append(manifest)
    paths = [p for run in [TRAIN, EVALUATION] for p in run.iterdir()
             if p.is_file() and 'partial' not in p.name and p.suffix != '.tmp']
    for video, snapshot, review, _, plan in videos:
        paths += [p for p in snapshot.rglob('*') if p.is_file()]
        paths += list(video.parent.glob(video.stem + '*'))
        paths += [review, plan]
    paths += [QUEUE / (stem + suffix) for stem in ['iter96_training_episodes', 'iter96_fixed_episodes']
              for suffix in ['.csv', '.json']]
    # The actual canonical source is included so the learner fork is explicit.
    paths += [checks.PRIOR_TRAIN / name for name in ['checkpoint.pt', 'policy.pt', 'INPUTS.json', 'metrics.json',
        'source_snapshot.zip', 'source_hashes.json', 'initial_ground_states.json', 'job.json']]
    for previous in ['residual_ppo_fixed_train128_iter32_n512_01', 'train128_lateral_c12_1025']:
        paths += [REPORTS / previous / name for name in ['episodes.json', 'metrics.json', 'INPUTS.json', 'job.json',
            'completed.json', 'source_snapshot.zip', 'source_hashes.json', 'asset_dependencies.json']]
    diagnostic_lineage = {}
    for item in read(MILESTONE / 'weight_lineage.json'):
        diagnostic_lineage[item['path']] = item['sha256']
    for supplement in sorted(MILESTONE.glob('SUPPLEMENT_UPDATE*.json')):
        for item in read(supplement)['diagnostic_model_checks']:
            assert item['path'] not in diagnostic_lineage or diagnostic_lineage[item['path']] == item['sha256']
            diagnostic_lineage[item['path']] = item['sha256']
    for name, expected in diagnostic_lineage.items():
        assert sha(ROOT / name) == expected
        paths.append(ROOT / name)
    for directory in ['residual_n2048_first_ground_cohort_20261025', 'residual_n2048_first_complete_20261025',
                      'residual_iter32_fixed_diagnostic_20261025']:
        paths += [p for p in (REPORTS / directory).iterdir() if p.is_file() and p.suffix != '.tmp']
    paths += [ROOT / 'scripts/export_full_racket_train_csv_v4.py', ROOT / 'scripts/prepare_full_racket_latest_video_v4.py',
        ROOT / 'scripts/render_full_racket_latest_effect_v4.py', ROOT / 'scripts/render_full_racket_r3_effect.py',
        ROOT / 'scripts/render_current_v4_progress.py', QUEUE / 'run.py', core.BASE,
        REPORTS / 'MODEL_RULES.json', REPORTS / 'model_task_comparison_20261025/REPORT.json',
        Path(__file__), HERE / 'contact_checks.py']
    assert all(p.is_file() for p in paths)
    entries = {str(p.relative_to(ROOT)): p for p in paths}
    from quadsmash.usd_dependencies_v4 import verify_dependency_manifest, MANIFEST
    verify_dependency_manifest(ROOT)
    dependencies = read(EVALUATION / 'asset_dependencies.json')
    assert dependencies['all_layers_pinned'] and dependencies['manifest_sha256'] == sha(ROOT / MANIFEST)
    entries[str(MANIFEST)] = ROOT / MANIFEST
    walk = read(ROOT / 'configs/g07_go2_walk_transfer.json')
    walk_path = (ROOT / walk['policy_path']).resolve()
    assert sha(walk_path) == walk['policy_sha256']
    for path in [*[Path(name) for name in dependencies['resolved_layers']], walk_path]:
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'external_dependencies/' + str(path.relative_to('/'))
        entries[name] = path
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual completed iter96 native learner fork, frozen TRAIN128, two reviewed prospective videos and raw evidence. Not formal A/B acceptance.',
        changing_policy_training=training, independently_recomputed_fixed_TRAIN128=metrics,
        native_training_contact_matches=confirmation['matched_complete_episodes'],
        csv_summaries=[v['canonical_summary'] for v in csv_manifests], videos=video_results,
        policy_sha256=policy_sha, formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        diagnostic_checkpoint_scope='Selected per-update model files referenced by the preserved training-completion lineages are included with their original hashes. They are not canonical native resumes or frozen-policy performance evidence.',
        selection_note='Prospective five-case19/23/107/0/8 comparison plus separate prospective TRAIN58 targeted retest. Neither is random sampling; actual fixed results are retained.',
        training_note='Training completions span changing actors, normalizers and exploration. No full training motion was recorded; these videos are the new frozen assessment.',
        dependency_note='All10 pinned USD layers and actual walking actor are included; restore external_dependencies to the recorded layout or remap references explicitly.',
        runtime_note='Each native source_snapshot.zip identifies its exact runtime. Later renderer/exporter sources are separately included. The unused joint-reward candidate is excluded.',
        comparison_note='Earlier iter32 and lateral-baseline raw rows are descriptive comparisons from separate native runs, not causal evidence.',
        files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(entries.items())})
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    temporary = args.output.with_name(args.output.name + '.tmp')
    assert not temporary.exists(), 'Inspect an existing partial archive first'
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr('DELIVERY_MANIFEST.json', payload)
        for name, path in sorted(entries.items()):
            archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.read('DELIVERY_MANIFEST.json') == payload
        for name, info in manifest['files'].items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(part)
            assert archive.getinfo(name).file_size == info['bytes'] and digest.hexdigest() == info['sha256']
    os.replace(temporary, args.output)
    atomic_new(HERE / 'MANIFEST.json', manifest)
    result = dict(archive=str(args.output.relative_to(ROOT)), bytes=args.output.stat().st_size,
        sha256=sha(args.output), members=len(entries) + 1, all_member_CRC_and_SHA_checked=True,
        formal_acceptance=False, goal_complete=False)
    atomic_new(HERE / 'ARCHIVE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
