"""Package completed joint8 evidence and explicitly unexecuted next proposals."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
TRAIN = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01'
FIXED = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
TRIAL = REPORTS/'live_tangent_trial_20261025'
QUEUE = REPORTS/'joint_placement_probe8_queue01'
VIDEO_QUEUE = REPORTS/'joint_placement_probe8_video_queue01'
VIDEO = REPORTS/'video/Current_effect_residual_PPO_joint_placement_probe8_20261025.mp4'
OUTPUT = REPORTS/'G07_joint_reward_probe8_native_evidence_and_next_trial_20261025.zip'
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save_once(path, value):
    assert not path.exists(), 'Preserve delivery evidence: '+str(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    assert not temporary.exists()
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    assert not any(path.exists() for path in [OUTPUT, HERE/'MANIFEST.json', HERE/'ARCHIVE.json'])
    spec = importlib.util.spec_from_file_location('completed_joint8_original_checks', QUEUE/'after_training.py')
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    training = probe.verify_training()
    fixed = probe.core.verify_evaluation(FIXED, TRAIN/'policy.pt')
    assert training['independent_changing_policy_counts']['complete'] == 0
    assert fixed['complete'] == 0 and fixed['airborne_contacts'] == 20 and fixed['conditional_recovery_count'] == 12
    review, receipt = read(VIDEO_QUEUE/'VISUAL_REVIEW.json'), read(VIDEO_QUEUE/'VIDEO_UPLOAD_RECEIPT.json')
    assert review['status'] == 'APPROVED_FOR_UPLOAD' and receipt['download_verified']
    assert sha(VIDEO) == review['video_sha256'] == receipt['sha256']
    assert review['root_decode_check']['all_frames_decoded']
    video_meta = read(VIDEO.with_suffix('.json'))
    for name, expected in video_meta['sources'].items():
        assert sha(ROOT/name) == expected, 'Video input changed: '+name
    preparation = read(TRIAL/'PREPARATION.json')
    assert not preparation['native_execution']
    plan = read(TRIAL/'PREDECLARED_PLAN.json')
    for name, expected in plan['source_hashes'].items():
        assert sha(ROOT/name) == expected, 'Current pinned source changed: '+name
    health = read(REPORTS/'joint_resume_health_20261025/REPORT.json')
    assert not health['frozen_policy_performance']
    paths = [path for directory in [TRAIN, FIXED] for path in directory.iterdir()
        if path.is_file() and 'partial' not in path.name and path.suffix != '.tmp']
    # Existing exact native source owners, including every new proposal.
    paths += [ROOT/name for name in plan['source_hashes']]
    paths += [TRIAL/'PREDECLARED_PLAN.json', TRIAL/'QUEUE_PREPARATION_CHECKS.json']
    for directory in [QUEUE, VIDEO_QUEUE,
            REPORTS/'joint_placement_first_ground_cohort_20261025',
            REPORTS/'joint_placement_export_check_20261025',
            REPORTS/'joint_placement_probe8_signal_replay_20261025',
            REPORTS/'joint8_contact_centering_20261025',
            REPORTS/'joint_resume_health_20261025']:
        paths += [path for path in directory.iterdir() if path.is_file()
            and path.suffix in ['.json', '.py', '.csv', '.gz', '.jpg', '.png']]
    paths += list(VIDEO.parent.glob(VIDEO.stem+'*'))
    paths += [path for path in (REPORTS/'video_snapshot_residual_ppo_joint_placement_probe8_20261025').rglob('*') if path.is_file()]
    paths += [ROOT/health['actual_diagnostic_checkpoint']]
    # Actual transferred actor32 source remains separate from this new learner.
    prior = REPORTS/'residual_ppo_n512_resume24_01'
    paths += [prior/name for name in ['checkpoint.pt', 'policy.pt', 'INPUTS.json', 'metrics.json',
        'job.json', 'source_hashes.json', 'source_snapshot.zip']]
    paths += [probe.core.BASE, REPORTS/'MODEL_RULES.json',
        REPORTS/'model_task_comparison_20261025/REPORT.json', Path(__file__).resolve()]
    entries = {str(path.relative_to(ROOT)):path for path in paths}
    from quadsmash.usd_dependencies_v4 import verify_dependency_manifest, MANIFEST
    verify_dependency_manifest(ROOT)
    dependencies = read(FIXED/'asset_dependencies.json')
    assert dependencies['all_layers_pinned'] and dependencies['manifest_sha256'] == sha(ROOT/MANIFEST)
    walk = read(ROOT/'configs/g07_go2_walk_transfer.json')
    walk_path = ROOT/walk['policy_path']
    assert sha(walk_path) == walk['policy_sha256']
    for path in [*[Path(name) for name in dependencies['resolved_layers']], walk_path]:
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'external_dependencies/'+str(path.relative_to('/'))
        entries[name] = path
    assert all(path.is_file() for path in entries.values())
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed fresh joint-placement learner8, all frozen TRAIN128 outcomes and reviewed native video. Separate ongoing learner21 changing-policy snapshot and explicitly unexecuted causal-feedback proposals are clearly labeled. No new formal A/B result.',
        actual_completed_joint8_training=training, independently_recomputed_frozen_TRAIN128=fixed,
        video_url=receipt['url'], video_sha256=sha(VIDEO), all4308_frames_decoded_and_reviewed=True,
        live_training_snapshot_iteration=health['actual_iteration'],
        live_training_snapshot_counts=health['independent_all_closed_changing_policy_counts'],
        snapshot_note='The included learner21 diagnostic checkpoint and7001 closed episodes are a changing-policy prefix of the still-running joint40 job. They are not a final checkpoint, frozen performance, or a claim that joint40 completed.',
        next_proposals_note='Both prepared gain035/gain070 policies retain native_execution=false. Preparation proves CPU replay and bounds only. No future native proposal result is included in this archive.',
        native_completed_outcome_note='Joint8 frozen TRAIN8 is a legal airborne attack/recovery near miss: speed8.006m/s and first landing x4.218m exceed target x4 by0.218m. All8 target hits are below8m/s. Complete success remains0/128.',
        first_ground_cohort_note='All209 air contacts in the first2048 ground episodes occurred before the first actor gradient update. These first hits are not evidence of reward-caused improvement.',
        source_note='Each completed native run retains its exact source_snapshot.zip; new trial and reporting sources are separately included with all hashes. Restore external_dependencies to the recorded paths or explicitly remap USD references.',
        renderer_note='Actual saved whole-body/arm/racket motion, side/top; no interpolated or generated motion. All six prospectively selected examples and failures remain.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        files={name:dict(bytes=path.stat().st_size, sha256=sha(path)) for name,path in sorted(entries.items())})
    payload = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    temporary = OUTPUT.with_suffix('.zip.tmp')
    assert not temporary.exists()
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr('DELIVERY_MANIFEST.json', payload)
        for name, path in sorted(entries.items()):
            archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.read('DELIVERY_MANIFEST.json') == payload
        assert len(archive.namelist()) == len(entries)+1
        for name, expected in manifest['files'].items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(block)
            assert archive.getinfo(name).file_size == expected['bytes'] and digest.hexdigest() == expected['sha256']
    os.replace(temporary, OUTPUT)
    save_once(HERE/'MANIFEST.json', manifest)
    result = dict(archive=str(OUTPUT.relative_to(ROOT)), bytes=OUTPUT.stat().st_size,
        sha256=sha(OUTPUT), members=len(entries)+1, all_member_CRC_and_SHA_checked=True,
        formal_acceptance=False, goal_complete=False)
    save_once(HERE/'ARCHIVE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
