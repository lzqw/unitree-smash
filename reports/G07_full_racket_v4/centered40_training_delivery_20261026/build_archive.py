"""Archive the completed native centered8-to40 TRAIN and its actual videos."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    archive = REPORTS/'G07_centered40_completed_native_training_and_two_complete_videos_20261026.zip'
    assert not archive.exists() and not (HERE/'ARCHIVE_CHECK.json').exists()
    run = REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
    source = REPORTS/'residual_ppo_centered_base8_n2048_probe8_02'
    queue = REPORTS/'centered_residual_resume32_queue01'
    job = read(run/'job.json')
    receipt = read(queue/'TRAINING_RECEIPT.json')
    audit = read(queue/'TRAINING_TRACE_AUDIT.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
    assert job['charged_env_steps'] == read(run/'completed.json')['env_steps'] == 8388608
    assert receipt['final_iteration'] == 40 and receipt['cumulative_new_learner_steps'] == 10485760
    assert receipt['all17_Adam_states_advanced640_steps'] and receipt['actual_episode_chains']['every_source_inclusive_to_final_exclusive_chain_exact']
    assert receipt['checkpoint_sha256'] == sha(run/'checkpoint.pt') and receipt['policy_sha256'] == sha(run/'policy.pt')
    assert audit['all_selected_trajectories_checked'] and audit['selected_episodes_checked'] == 14
    assert read(queue/'TRACE_AUDIT_ATTEMPT.json')['passed']
    for pid in [job['pid'], job['child_pid']]:
        path = Path('/proc', str(pid), 'cmdline')
        assert not path.exists() or str(run).encode() not in path.read_bytes()
    for owner in [queue, REPORTS/'centered_intermediate_selection_queue01']:
        for name, digest in read(owner/'PREDECLARED_PLAN.json')['source_hashes'].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == digest, name
    diagnostics = []
    for iteration in range(9, 41):
        path = run/'ppo'/f'model_{iteration-1}.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['iter'] == iteration and not state['native']['canonical_restart']
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values() if torch.is_tensor(value))
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values() for value in item.values() if torch.is_tensor(value))
        for key in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
            assert int(state['model_state_dict'][key]) == iteration*262144
        diagnostics.append(dict(iteration=iteration, path=str(path.relative_to(ROOT)), sha256=sha(path), canonical_restart=False))
    assert sha(run/'episodes.partial.json') == sha(run/'episodes.json')
    # Preserve final raw episodes once; the atomic partial publication is byte-identical.
    paths = [path for path in run.rglob('*') if path.is_file() and path.name != 'episodes.partial.json']
    paths += [source/name for name in ['checkpoint.pt', 'policy.pt', 'INPUTS.json', 'job.json', 'completed.json', 'metrics.json']]
    paths += [REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt']
    paths += [queue/name for name in ['run.py', 'PREDECLARED_PLAN.json', 'PARENT_VERIFIED_DECISION.json', 'TRAINING_RECEIPT.json',
                                     'TRAINING_TRACE_AUDIT.json', 'TRACE_AUDIT_ATTEMPT.json', 'ROOT_STRICT_RESUME_FIRST_UPDATE_CHECK.json',
                                     'ROOT_ACTUAL_UPDATE15_FIRST_COHORT_CHECK.json']]
    selected_videos = []
    for stem, snapshot_name, review_name in [
        ('Current_training_centered_first_complete_and_failures_20261026', 'video_snapshot_centered_resume_first_complete_20261026', 'centered_resume_training_capture_20261026'),
        ('Current_training_centered_second_complete_and_failure_20261026', 'video_snapshot_centered_second_complete_20261026', 'centered_second_complete_video_20261026'),
    ]:
        video = REPORTS/'video'/f'{stem}.mp4'
        review_dir = REPORTS/review_name
        review = read(review_dir/'VISUAL_REVIEW.json')
        upload = read(review_dir/'VIDEO_UPLOAD_RECEIPT.json')
        assert review['approved_for_authorized_GitHub_video_upload'] and review['video_sha256'] == sha(video)
        assert upload['download_verified'] and upload['sha256'] == sha(video)
        selected_videos.append(dict(path=str(video.relative_to(ROOT)), sha256=sha(video), scope=upload['scope']))
        paths += [path for path in (REPORTS/snapshot_name).rglob('*') if path.is_file()]
        paths += [video, video.with_suffix('.json'), *video.parent.glob(stem+'_case*.jpg')]
        verification = video.with_name(stem+'_verification.json')
        if verification.exists():
            paths.append(verification)
        paths += [path for path in review_dir.glob('*') if path.is_file() and path.suffix in {'.json', '.py', '.md'}]
        paths += list((review_dir/'decoded_samples').glob('*.jpg'))
    paths += [ROOT/name for name in ['configs/full_racket_v4_r3.json', 'configs/court_net_v4.json', 'configs/shuttle_physics_v2.json',
                                    'configs/g07_full_racket_eval_v4_r3.json', 'configs/g07_full_racket_residual_ppo_centered_v4.json',
                                    'configs/full_racket_usd_dependencies_v4.json', 'scripts/render_full_racket_latest_effect_v4.py',
                                    'scripts/render_full_racket_r3_effect.py', 'scripts/render_current_v4_progress.py',
                                    'reports/G07_full_racket_v4/MODEL_RULES.json']]
    paths.append(Path(__file__))
    paths = sorted(set(paths))
    assert all(path.is_file() for path in paths)
    payload = {str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in paths}
    companion = read(REPORTS/'centered_training_capture_video_20261026/ARCHIVE_UPLOAD_RECEIPT.json')
    assert companion['download_verified']
    reproduction = ("Completed native centered8-to40 TRAIN, not frozen-policy performance or formal acceptance.\n"
                    "Restore the recorded native Isaac/RSL environment and the required runtime companion from MANIFEST.json.\n"
                    "The companion includes the original Go2 walking policy, standard PiPER URDF and all10 USD layers; restore its external paths or equivalent links.\n"
                    "Original training source is in the included source_snapshot.zip; source_hashes.json and INPUTS.json identify the exact runtime/configuration.\n"
                    "Re-execution uses the original job.json command/seed and included canonical8 source in a NEW output directory through the original GPU0 guard/ledger.\n"
                    "checkpoint.pt at40 is a canonical restart. ppo/model_8.pt through model_39.pt are diagnostic boundaries, not restart checkpoints.\n"
                    "Training episode IDs continue from the source; actions, normalizers, weights and exploration vary within a TRAIN episode.\n"
                    "For motion review use the saved native100Hz poses/actions/public inputs and video snapshots; no in-contact PhysX state restoration is claimed.\n"
                    "All17156 final raw episodes, native contacts and all14 selected whole-episode traces are included. episodes.partial.json is omitted because it is byte-identical to episodes.json.\n"
                    "The separately running final40 frozen TRAIN128 and intermediate16/24/32 assessments are excluded; do not infer their scores from this archive.\n").encode()
    payload['REPRODUCTION.txt'] = dict(bytes=len(reproduction), sha256=hashlib.sha256(reproduction).hexdigest())
    manifest = dict(schema='completed-centered40-native-training-v1', created_at=datetime.now(timezone.utc).isoformat(), payload=payload,
                    source_job_complete=True, source_job=job, training_verification=receipt, selected_training_traces_checked=14,
                    source_canonical8_sha256=sha(source/'checkpoint.pt'), final_canonical40_sha256=sha(run/'checkpoint.pt'),
                    final_public40_sha256=sha(run/'policy.pt'), actual_diagnostic_boundaries=diagnostics, actual_training_videos=selected_videos,
                    required_runtime_companion={key: companion[key] for key in ['url', 'bytes', 'sha256']},
                    excluded_live_assessments=['final40 original TRAIN128', 'intermediate16/24/32 original TRAIN128'],
                    frozen_policy_performance=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    print(json.dumps(dict(phase='WRITING_IMMUTABLE_TRAIN_ARCHIVE', files=len(paths), uncompressed_bytes=sum(record['bytes'] for record in payload.values()))), flush=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in paths:
            bundle.write(path, str(path.relative_to(ROOT)))
        bundle.writestr('REPRODUCTION.txt', reproduction)
        bundle.writestr('MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None and len(bundle.namelist()) == len(payload)+1
        for name, record in payload.items():
            data = bundle.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                  sha256=sha(archive), payload_members=len(payload), total_members=len(payload)+1,
                  all_member_CRC_verified=True, all_source_payload_SHA_verified=True, source_job_complete=True,
                  frozen_policy_performance=False, formal_acceptance=False, goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
