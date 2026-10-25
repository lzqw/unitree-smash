"""Package completed iter32 native evidence after the actual video is reviewed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/G07_full_racket_v4'
HERE = Path(__file__).resolve().parent
TRAIN = REPORTS / 'residual_ppo_n512_resume24_01'
EVALUATION = REPORTS / 'residual_ppo_fixed_train128_iter32_n512_01'
SNAPSHOT = REPORTS / 'video_snapshot_residual_ppo_iter32_balanced_20261025'
CSV = REPORTS / 'residual_csv_queue01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path,
        default=REPORTS / 'video/Current_effect_residual_PPO_iter32_20261025_v3.mp4')
    parser.add_argument('--review', type=Path,
        default=REPORTS / 'residual_iter32_video_queue01/VISUAL_REVIEW.json')
    parser.add_argument('--output', type=Path,
        default=REPORTS / 'G07_PPO_iter32_fixed_TRAIN128_video_training_and_raw_20261025.zip')
    parser.add_argument('--check-prerequisites', action='store_true')
    args = parser.parse_args()
    video, review_path, output = args.video.resolve(), args.review.resolve(), args.output.resolve()
    prerequisites = [TRAIN / name for name in ['job.json', 'completed.json', 'metrics.json',
        'checkpoint.pt', 'policy.pt', 'episodes.json', 'contacts.json']]
    prerequisites += [EVALUATION / name for name in ['job.json', 'completed.json', 'metrics.json',
        'episodes.json', 'contacts.json', 'all_first_episodes.npz', 'geometry.json', 'geometry.npz']]
    prerequisites += [CSV / (stem + suffix) for stem in ['iter32_training_episodes', 'iter32_fixed_episodes']
                      for suffix in ['.json', '.csv']]
    prerequisites += [video, video.with_suffix('.json'), review_path, SNAPSHOT / 'cases.json']
    missing = [str(p) for p in prerequisites if not p.is_file()]
    if args.check_prerequisites:
        print(json.dumps(dict(prerequisite_files_present=not missing, missing=missing,
            completed_run_content_validated=False, archive_created=False, formal_acceptance=False)), flush=True)
        return
    if missing:
        raise RuntimeError('Native completion/video/CSV prerequisites are pending: ' + repr(missing))
    assert not output.exists(), 'Never overwrite an evidence archive'
    spec = importlib.util.spec_from_file_location('native_iter32_checks', REPORTS / 'residual_resize512_queue01/run.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    core = checks.core
    core.verify_job(TRAIN, checks.STEPS)
    training = checks.verify_training()
    metrics = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    assert metrics['episodes'] == 128 and metrics['numerical_failures'] == 0
    policy_sha = sha(TRAIN / 'policy.pt')
    review = read(review_path)
    assert review['status'] == 'APPROVED_FOR_UPLOAD'
    verified = review['verification']
    assert verified['sha256'] == sha(video) and verified['all_frames_decoded']
    assert verified['fps'] == 50 and verified['resolution'] == [1280, 720]
    metadata, snapshot = read(video.with_suffix('.json')), read(SNAPSHOT / 'cases.json')
    assert metadata['sha256'] == sha(video) and metadata['frames'] == verified['frames_decoded']
    assert metadata['native_recorded_motion'] and not metadata['interpolated_or_generated_motion']
    assert not metadata['renderer_uses_gpu'] and 'llvmpipe' in metadata['renderer']
    assert snapshot['policy_sha256'] == policy_sha
    expected_cases = [19, 23, 107, 0, 8]
    assert snapshot['preselected_TRAIN_indices_before_fixed_evaluation'] == expected_cases[:4]
    assert snapshot['posthoc_demonstration_TRAIN_indices'] == [8]
    assert snapshot['all_cases_preselected_before_fixed_evaluation'] is False
    assert [c['env_id'] for c in snapshot['cases']] == expected_cases
    assert [c['env_id'] for c in metadata['cases']] == expected_cases
    assert all(c['run'] == EVALUATION.name for c in metadata['cases'])
    assert metadata['sources'][str((SNAPSHOT / 'cases.json').relative_to(ROOT))] == sha(SNAPSHOT / 'cases.json')
    rows = read(EVALUATION / 'episodes.json')
    for case in metadata['cases']:
        # String comparison preserves equality for unobserved NaN event fields.
        assert json.dumps(case['row'], sort_keys=True) == json.dumps(rows[case['env_id']], sort_keys=True)
    for name, expected in metadata['sources'].items():
        assert sha(ROOT / name) == expected, 'Recorded video source changed: ' + name
    csv_manifests = []
    for source, stem in [(TRAIN, 'iter32_training_episodes'), (EVALUATION, 'iter32_fixed_episodes')]:
        manifest = read(CSV / (stem + '.json'))
        assert manifest['original_runtime_metrics_match'] and not manifest['formal_acceptance']
        assert manifest['source_episode_sha256'] == sha(source / 'episodes.json')
        assert manifest['csv_sha256'] == sha(CSV / (stem + '.csv'))
        key = 'final_exported_policy_sha256' if source == TRAIN else 'assessed_policy_sha256'
        assert manifest[key] == policy_sha
        assert manifest['rows'] == (training['changing_policy_counts']['completed_episodes'] if source == TRAIN else 128)
        csv_manifests.append(manifest)
    paths = [p for run in [TRAIN, EVALUATION] for p in run.iterdir()
             if p.is_file() and 'partial' not in p.name and p.suffix != '.tmp']
    paths += [p for p in SNAPSHOT.rglob('*') if p.is_file()]
    paths += list(video.parent.glob(video.stem + '*'))
    paths += [REPORTS / 'video' / ('Current_effect_residual_PPO_iter32_20261025' + suffix)
              for suffix in ['.mp4', '.json', '_verification.json']]
    paths += [REPORTS / 'video' / ('Current_effect_residual_PPO_iter32_20261025_v2' + suffix)
              for suffix in ['.mp4', '.json']]
    paths += [REPORTS / 'residual_iter32_video_review_20261025' / name
              for name in ['FIRST_EXPORT_REVIEW.json', 'renderer_first_export.py',
                           'SECOND_EXPORT_REVIEW.json', 'renderer_second_export.py']]
    for directory in ['residual_iter32_training_completion_20261025', 'residual_iter32_fixed_diagnostic_20261025']:
        paths += [p for p in (REPORTS / directory).iterdir() if p.is_file()]
    paths += [CSV / (stem + suffix) for stem in ['iter32_training_episodes', 'iter32_fixed_episodes']
              for suffix in ['.csv', '.json']]
    paths += [REPORTS / 'residual_ppo_fixed_train128_1250' / name
              for name in ['episodes.json', 'metrics.json', 'INPUTS.json', 'job.json']]
    paths += [REPORTS / 'train128_lateral_c12_1025' / name
              for name in ['episodes.json', 'metrics.json', 'INPUTS.json', 'job.json']]
    paths += list((REPORTS / 'residual_n512_first_complete_20261025').glob('*.json'))
    paths += list((REPORTS / 'residual_n512_first_complete_20261025').glob('*.gz'))
    confirmation = REPORTS / 'residual_resize2048_queue01/PREVIOUS_TRAINING_COMPLETE_CONTACT_CONFIRMATION.json'
    if confirmation.exists():
        assert read(confirmation)['exact_contact_record_matches']
        paths.append(confirmation)
    paths += [review_path, core.BASE, REPORTS / 'MODEL_RULES.json',
        REPORTS / 'model_task_comparison_20261025/REPORT.json',
        REPORTS / 'video_spec_residual_ppo_iter32_20261025.json',
        REPORTS / 'video_spec_residual_ppo_iter32_balanced_20261025.json',
        REPORTS / 'residual_resize512_queue01/run.py',
        ROOT / 'scripts/export_full_racket_train_csv_v4.py',
        ROOT / 'scripts/render_full_racket_latest_effect_v4.py',
        ROOT / 'scripts/render_full_racket_r3_effect.py', ROOT / 'scripts/render_current_v4_progress.py',
        ROOT / 'scripts/prepare_full_racket_latest_video_v4.py', Path(__file__)]
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
        scope='Actual iter32 fixed TRAIN128 assessment, reviewed five-case video, and separately scoped changing-policy512-env training. Not formal A/B.',
        canonical_fixed_TRAIN128_metrics=metrics, changing_policy_training=training,
        csv_summaries=[m['canonical_summary'] for m in csv_manifests],
        policy_sha256=policy_sha, video_sha256=sha(video),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        excluded='Subsequent2048-env raw rollouts/checkpoints and policy performance are excluded. A separately labelled diagnostic may mention its startup/ground checks; no pending result is inherited.',
        selection_note='The four preselected cases19/23/107/0 all failed and are retained. TRAIN8 was selected posthoc by target miss distance among actual no-frame air-attack-and-recovery episodes. The five-case video is not random population sampling.',
        retained_initial_exports='The original four-case clip and v2 are retained with rejected framing reviews and their renderer sources. V3 is the reviewed primary video: court-end landing visible, robot recovery kept close.',
        training_note='The TRAIN19_LANE275_EP1 success spans changing actors/normalizers. Event and contact/terminal poses do not supply its unsaved full motion trajectory or one frozen successful policy.',
        runtime_note='Use each native run source_snapshot.zip for its exact runtime. Later renderer/exporter source is separately included. Isaac runtime and pinned libraries are separately required.',
        dependency_note='All10 pinned USD layers and the actual walking actor are included. Restore external_dependencies to their recorded absolute layout or explicitly remap references.',
        comparison_note='Prior iter8 and lateral baseline raw TRAIN128 rows allow descriptive comparison. Separate native runs and policies do not establish causal effects.',
        files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(entries.items())})
    payload = json.dumps(manifest, indent=2) + '\n'
    (HERE / 'MANIFEST.json').write_text(payload)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr('DELIVERY_MANIFEST.json', payload)
        for name, path in sorted(entries.items()):
            archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, info in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info['sha256']
    result = dict(archive=str(output.relative_to(ROOT)), bytes=output.stat().st_size,
        sha256=sha(output), members=len(entries) + 1, all_members_rehashed=True, formal_acceptance=False)
    (HERE / 'ARCHIVE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
