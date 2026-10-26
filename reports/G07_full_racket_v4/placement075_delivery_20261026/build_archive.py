"""Package only completed075 stage8 learning, frozen128 and reviewed video.

The ongoing stage8-to40 continuation contributes no outcomes. Original producer
files remain untouched; a partial duplicate is omitted only after byte equality.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import numpy as np
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from quadsmash.evaluation_v4 import independent_episode

    reports = ROOT / 'reports/G07_full_racket_v4'
    train = reports / 'residual_ppo_centered_placement075_n2048_probe8_01'
    frozen = reports / 'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
    queue = reports / 'centered_placement075_pilot8_queue01'
    health = reports / 'centered_placement075_health_20261026'
    source = reports / 'residual_ppo_centered_base8_n2048_resume32_01'
    source_eval = reports / 'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
    base = reports / 'residual_ppo_joint_placement_actor32_n2048_probe8_01'
    snapshot = reports / 'video_snapshot_centered_placement075_pilot8_20261026'
    video = reports / 'video/Current_effect_centered_placement075_pilot8_20261026.mp4'
    output = reports / 'G07_placement075_stage8_native_training_frozen_TRAIN128_and_video_20261026.zip'
    assert not output.exists() and not (HERE / 'ARCHIVE_CHECK.json').exists()
    for folder, steps in [(train, 2097152), (frozen, 102400)]:
        job = read(folder / 'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
        assert job['charged_env_steps'] == read(folder / 'completed.json')['env_steps'] == steps
    review = read(HERE / 'VISUAL_REVIEW.json')
    uploaded = read(HERE / 'VIDEO_UPLOAD_RECEIPT.json')
    assert review['approved_for_authorized_GitHub_video_upload'] and uploaded['download_verified']
    assert sha(video) == review['video_sha256'] == uploaded['sha256']
    audit = read(queue / 'TRAINING_TRACE_AUDIT.json')
    assert read(queue / 'TRACE_AUDIT_ATTEMPT.json')['passed']
    assert audit['all_selected_trajectories_checked'] and audit['selected_episodes_checked'] == 12
    for name, digest in audit['source_sha256'].items():
        assert sha(Path(name)) == digest, name
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    rows = read(frozen / 'episodes.json')
    summary = summarize_training(rows, cases, indices, protocol)
    assert summary == read(queue / 'FIXED_TRAIN128_RESULT.json')
    assert len(rows) == 128 and [v['source_training_index'] for v in rows] == list(range(128))
    assert summary['complete'] == 0 and summary['airborne_contacts'] == 26
    assert summary['conditional_recovery_count'] == 19 and summary['numerical_failures'] == 0
    changing_rows = read(train / 'episodes.json')
    changing = [independent_episode(v, protocol) for v in changing_rows]
    counts = dict(completed_episodes=len(changing_rows), air=sum(v['airborne_contact'] for v in changing),
                  conditional_recovery_count=sum(v['conditional_recovery'] for v in changing),
                  complete=sum(v['complete'] for v in changing),
                  numerical_failures=sum(v['numerical_failures'] for v in changing_rows))
    assert counts == read(train / 'metrics.json')['counts']
    assert len(changing_rows) == 3696 and counts['numerical_failures'] == 0
    with np.load(frozen / 'all_first_episodes.npz') as poses:
        assert poses['link_poses'].shape == (400, 128, 28, 7)
        assert poses['ball_position_m'].shape == (400, 128, 3)
        valid = poses['valid_first_episode']
        assert valid.shape == (400, 128) and valid[0].all()
        assert np.isfinite(poses['link_poses'][valid]).all()
        assert np.isfinite(poses['ball_position_m'][valid]).all()
    with np.load(frozen / 'public_and_actions_first2.npz') as trace:
        assert trace['observations'].shape == (800, 2, 794)
        assert trace['actions'].shape == (800, 2, 18)
        assert trace['valid_first_episode'].shape == (800, 2)
    assert len(list((train / 'trajectory_traces').glob('lane*.npz'))) == 12
    assert len(list((train / 'ppo').glob('model_*.pt'))) == 8

    duplicate_omissions = []
    omitted = set()
    for partial, final in [(train / 'episodes.partial.json', train / 'episodes.json'),
                           (frozen / 'episodes.partial.json', frozen / 'episodes.json'),
                           (frozen / 'all_first_episodes.partial.npz', frozen / 'all_first_episodes.npz')]:
        if sha(partial) == sha(final):
            omitted.add(partial)
            duplicate_omissions.append(dict(omitted=str(partial.relative_to(ROOT)),
                                           identical_to=str(final.relative_to(ROOT)), sha256=sha(final)))
    paths = [p for folder in [train, frozen, snapshot, health] for p in folder.rglob('*')
             if p.is_file() and p not in omitted]
    paths += [p for p in queue.iterdir() if p.is_file() and p.name != 'observer.lock']
    paths += [p for p in HERE.rglob('*') if p.is_file() and not p.name.startswith('ARCHIVE_')
              and '__pycache__' not in p.parts]
    paths += [video, video.with_suffix('.json'), video.with_name(video.stem + '_verification.json')]
    paths += list(video.parent.glob(video.stem + '_case*.jpg'))
    metadata_names = ['job.json', 'INPUTS.json', 'completed.json', 'metrics.json', 'asset_dependencies.json']
    paths += [base / name for name in ['policy.pt', *metadata_names]]
    paths += [source / name for name in ['policy.pt', 'checkpoint.pt', 'source_snapshot.zip', *metadata_names]]
    paths += [source_eval / name for name in ['INPUTS.json', 'metrics.json', 'job.json', 'completed.json',
                                             'episodes.json', 'source_hashes.json']]
    initialization = read(train / 'INPUTS.json')['initialization']
    assert sha(source / 'policy.pt') == initialization['source_policy_sha256']
    for name, digest in initialization['source_evidence_sha256'].items():
        assert sha(source_eval / name) == digest, name
    paths += [ROOT / 'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json',
              ROOT / 'configs/g07_full_racket_eval_v4_r3.json', ROOT / protocol['training_suite']['path'],
              reports / 'MODEL_RULES.json', ROOT / 'docs/ENVIRONMENT.md', ROOT / 'docs/SOURCES.md',
              ROOT / 'reports/G00/environment.lock.json']
    for name, digest in protocol['physical_configs'].items():
        assert sha(ROOT / name) == digest, name
        paths.append(ROOT / name)
    walking_manifest = ROOT / 'configs/g07_go2_walk_transfer.json'
    walking = read(walking_manifest)
    assert sha(ROOT / walking['policy_path']) == walking['policy_sha256']
    paths += [walking_manifest, ROOT / walking['policy_path']]
    paths += list((ROOT / 'assets/vendor/piper_isaac_sim/piper_description').rglob('*.urdf'))
    paths += list((ROOT / 'assets/vendor/piper_isaac_sim/piper_description/meshes').glob('*.STL'))
    paths += [Path(name) for name in read(train / 'asset_dependencies.json')['resolved_layers']]
    for folder in ['centered40_same_motion_tool_geometry_20261026', 'current_legacy_comparison_20261026']:
        paths += [p for p in (reports / folder).iterdir() if p.is_file() and p.suffix in ['.py', '.json', '.csv', '.md']]
    mandatory = [train / name for name in ['checkpoint.pt', 'policy.pt', 'episodes.json', 'contacts.json',
                                          'source_snapshot.zip', 'trajectory_traces/manifest.json']]
    mandatory += [frozen / name for name in ['episodes.json', 'all_first_episodes.npz', 'public_and_actions_first2.npz']]
    mandatory += [health / name for name in ['UPDATE3_RAW_EPISODES_SNAPSHOT.json', 'FIRST2048_EPISODES_SNAPSHOT.json']]
    mandatory += [base / name for name in ['policy.pt', *metadata_names]]
    mandatory += [source_eval / name for name in initialization['source_evidence_sha256']]
    mandatory += [video, HERE / 'VISUAL_REVIEW.json', HERE / 'VIDEO_UPLOAD_RECEIPT.json']
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    assert set(mandatory).issubset(paths)
    members = {}
    external = []
    for path in sorted(set(paths)):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'runtime_external/' + str(path).lstrip('/')
        assert name not in members
        members[name] = path
        if not path.is_relative_to(ROOT):
            external.append(dict(original_absolute_path=str(path), archive_member=name, sha256=sha(path)))
    assert all('_resume32_02/' not in name and 'centered_placement075_resume32_queue02/' not in name for name in members)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed075 stage8 native learning plus all original frozenTRAIN128 outcomes and reviewed87.66s video. Ongoing stage8-to40 continuation excluded. No formal A/B score.',
        actual_training_control_env_steps=2097152, actual_training_GPU_hours=read(train / 'job.json')['charged_gpu_hours'],
        actual_changing_policy_counts=counts, independent_frozen_TRAIN128=summary,
        canonical_checkpoint_sha256=sha(train / 'checkpoint.pt'), policy_sha256=sha(train / 'policy.pt'),
        source_initialization=initialization, duplicate_omissions_after_byte_equality=duplicate_omissions,
        stored_trajectory_scope=dict(training='12 selected complete100Hz episodes; full public794 rows checked online but not stored for every rollout.',
                                     frozen='All128 first-episode body/ball poses at50Hz until termination; public794 observations and18 actions at100Hz only for first2 cases.'),
        actual_video=dict(duration_s=87.66, frames=4383, resolution=[1280, 720], fps=50,
                          source_motion='Actual native recorded poses; no generated or interpolated motion.'),
        historical_comparison_scope='Same-motion old/new tool kinematics are a CPU counterfactual, not an old-mass native test. Historical short-board protocol scores are separate and not inherited.',
        external_runtime_layers=external,
        external_runtime_restore_note='Restore external layers to their recorded original absolute paths or equivalent symlinks. Native Isaac/Kit and locked RSL-RL environment are not bundled. OmniPBR.mdl is a built-in visual shader.',
        required_companion_archives=[], mandatory_members=[str(p.relative_to(ROOT)) for p in mandatory],
        source_files={name:dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in members.items()},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    notes = '''This package contains completed075 stage8 learning and its frozen originalTRAIN128 assessment.
Training3696 closed episodes:331 air,118 contact-then-recoveries,0 complete,NF0. Changing-policy counts are not frozen performance.
Frozen128:26 air,19 contact-then-recoveries,5 qualifying attacks,0 complete,NF0. Formal A/B each200 and50/70/85 are still unproven.
All128 first-episode body/ball trajectories are50Hz. Full794 public observations/18 actions at100Hz are stored only for first2 frozen cases.
Training stores12 selected100Hz episodes. unpack_public in quadsmash.rl.episode_trace_v4 reconstructs their packed public rows. Online full-row checks do not imply all trajectories were retained.
For launch-time code, extract the training source_snapshot.zip at a fresh repository root, then overlay this archive payload. The original job command and locked environment metadata are included.
Source40 public policy and its six matching fixed-evaluation provenance files, fixed base846 policy with native receipts, walking actor, standardPiPER URDF/STLs and all10 USD layers are included. No companion archive is required for these data dependencies.
runtime_external/ contains Go2 layers with MANIFEST original paths. Restore those paths or equivalent symlinks before native use; Isaac/Kit and locked RSL-RL remain external software dependencies.
Native replay: use scripts/evaluate_full_racket_train_v4.py with the included stage8 policy, --noise none --seed724201 --headless and a NEW output directory. Use the original shared GPU0 guard and resource accounting; never overwrite included native evidence.
Strict native resume accepts canonical checkpoint.pt, not ppo/model_i.pt diagnostics. The completed stage8 actor RMS12582912 includes10485760 earlier statistics; critic RMS2097152 does not.
CPU video replay uses the included video_snapshot_centered_placement075_pilot8_20261026 and scripts/render_full_racket_latest_effect_v4.py with Mesa/llvmpipe.
The87.66s video retains six independent cases and failures. TRAIN107 late side view is partly cropped; overhead complements it. TRAIN0 rounded vz=-1.00 is actually-0.9951156974, failing the attack threshold.
All archive member hashes and CRCs are checked. This does not claim a fresh native replay or formal acceptance has been run on a new host.
The current32-update continuation and its outcomes are excluded.
'''
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in members.items():
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in ['.zip', '.npz', '.mp4', '.jpg'] else zipfile.ZIP_DEFLATED)
        archive.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        archive.writestr('REPRODUCE.txt', notes)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(members) + 2
        for name, entry in manifest['source_files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256'], name
            assert sha(members[name]) == entry['sha256'], name
        assert set(manifest['mandatory_members']).issubset(archive.namelist())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(output.relative_to(ROOT)),
                  bytes=output.stat().st_size, sha256=sha(output), source_payload_members=len(members),
                  total_members=len(members) + 2, mandatory_members=len(mandatory),
                  all_member_SHA_and_CRC_verified=True, all_sources_unchanged_after_build=True,
                  required_companion_archives=[], ongoing_continuation_excluded=True,
                  formal_acceptance=False, goal_complete=False)
    (HERE / 'ARCHIVE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
