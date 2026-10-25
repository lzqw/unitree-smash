"""Archive completed residual iter8 TRAIN128 trajectories and the reviewed video."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports/G07_full_racket_v4'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('checks', REPORTS / 'residual_resume24_queue01/run.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    run = REPORTS / 'residual_ppo_fixed_train128_1250'
    pilot = REPORTS / 'residual_ppo_pilot_train128_1230'
    metrics = checks.verify_evaluation(run, pilot / 'policy.pt')
    assert metrics['airborne_contacts'] == 18 and metrics['conditional_recovery_count'] == 9
    assert metrics['complete'] == metrics['numerical_failures'] == 0
    output = REPORTS / 'G07_PPO_iter8_fixed_TRAIN128_video_and_raw_20261025.zip'
    assert not output.exists()
    video = REPORTS / 'video/Current_effect_residual_PPO_iter8_20261025_v2.mp4'
    review = json.loads((REPORTS / 'residual_video_queue01/V2_VISUAL_REVIEW.json').read_text())
    assert review['status'] == 'APPROVED_FOR_UPLOAD' and review['verification']['sha256'] == sha(video)
    paths = [p for p in run.iterdir() if p.is_file() and 'partial' not in p.name]
    paths += list((REPORTS / 'video_snapshot_residual_ppo_iter8_20261025').rglob('*'))
    paths += list(video.parent.glob(video.stem + '*'))
    paths += [pilot / name for name in ['policy.pt', 'checkpoint.pt', 'INPUTS.json', 'metrics.json', 'source_snapshot.zip', 'job.json', 'completed.json']]
    paths += [REPORTS / 'train128_lateral_c12_1025' / name for name in ['episodes.json', 'metrics.json', 'INPUTS.json', 'job.json']]
    paths += [checks.BASE, REPORTS / 'RESIDUAL_PPO_ITER8_FIXED_TRAIN128_20261025.json',
        REPORTS / 'video_spec_residual_ppo_iter8_20261025.json', REPORTS / 'residual_video_queue01/V2_VISUAL_REVIEW.json',
        ROOT / 'scripts/render_full_racket_latest_effect_v4.py', ROOT / 'scripts/render_full_racket_r3_effect.py',
        ROOT / 'scripts/render_current_v4_progress.py', ROOT / 'scripts/prepare_full_racket_latest_video_v4.py', Path(__file__)]
    entries = {str(p.relative_to(ROOT)): p for p in paths if p.is_file()}
    dependencies = json.loads((run / 'asset_dependencies.json').read_text())['resolved_layers']
    walk = json.loads((ROOT / 'configs/g07_go2_walk_transfer.json').read_text())
    walk_path = (ROOT / walk['policy_path']).resolve()
    assert sha(walk_path) == walk['policy_sha256']
    for p in [*[Path(name) for name in dependencies], walk_path]:
        name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'external_dependencies/' + str(p.relative_to('/'))
        entries[name] = p
    manifest = dict(scope='Actual fixed residual PPO iter8 TRAIN128 and three preselected independent video cases23,107,0. Not formal A/B.',
        canonical_metrics=metrics, policy_sha256=sha(pilot / 'policy.pt'), video_sha256=sha(video),
        formal_acceptance=False, formal_cases_read=False,
        active_run_excluded='Ongoing512-env PPO continuation is excluded; no performance is inherited.',
        runtime_note='Use each completed run source_snapshot.zip for its exact runtime. Later renderer files export the same recorded native geometry, poses and outcomes. Isaac runtime is separately required.',
        dependency_note='Restore external_dependencies to the recorded absolute layout or explicitly remap USD references. The actual pinned walking actor and10 USD layers are included.',
        comparison_note='Earlier baseline raw128 rows are included for descriptive paired recomputation. Separate native executions and different policies do not establish causal or significance claims.',
        files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(entries.items())})
    text = json.dumps(manifest, indent=2) + '\n'
    (HERE / 'MANIFEST.json').write_text(text)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr('DELIVERY_MANIFEST.json', text)
        for name, p in sorted(entries.items()):
            archive.write(p, name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, item in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item['sha256']
    result = dict(archive=str(output.relative_to(ROOT)), bytes=output.stat().st_size,
        sha256=sha(output), members=len(entries) + 1, all_members_rehashed=True, formal_acceptance=False)
    (HERE / 'ARCHIVE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
