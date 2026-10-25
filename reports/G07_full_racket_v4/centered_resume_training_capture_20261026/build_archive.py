"""Archive the actual partial native TRAIN milestone, never a completed job."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = REPORTS/'G07_centered_resume_first_complete_training_video_20261026.zip'
    assert not archive.exists() and not (HERE/'ARCHIVE_CHECK.json').exists()
    snapshot = REPORTS/'video_snapshot_centered_resume_first_complete_20261026'
    video = REPORTS/'video/Current_training_centered_first_complete_and_failures_20261026.mp4'
    run = REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
    source = REPORTS/'residual_ppo_centered_base8_n2048_probe8_02'
    base = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    review = read(HERE/'VISUAL_REVIEW.json')
    assert review['approved_for_authorized_GitHub_video_upload'] and review['video_sha256'] == sha(video)
    assert read(HERE/'VIDEO_UPLOAD_RECEIPT.json')['download_verified']
    active = [row for row in read(ROOT/'status/GPU_LEDGER.json')['runs'] if row['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == run
    for key in ['pid','child_pid']:
        os.kill(active[0][key],0)
        assert str(run).encode() in Path('/proc',str(active[0][key]),'cmdline').read_bytes()
    for folder in ['centered_residual_pilot8_queue02','centered_residual_resume32_queue01']:
        for path,digest in read(REPORTS/folder/'PREDECLARED_PLAN.json')['source_hashes'].items():
            assert sha(ROOT/path) == digest, 'Active pinned source changed: '+path
    observation = dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_guard_and_native_verified_live=True,
        original_main_shared_ledger_entry=active[0],progress_at_observation=read(run/'progress.json'),
        source_job_is_complete=False,scope='Read-only publication while the actual native TRAIN source job remains running. No terminal receipt or full-job resource total is synthesized.',
        formal_acceptance=False,goal_complete=False)
    (HERE/'RUNNING_OBSERVATION.json').write_text(json.dumps(observation,indent=2)+'\n')
    actual_iterations = read(snapshot/'cases.json')['cases'][0]['actor_iterations_encountered']
    assert actual_iterations == list(range(10,17))
    diagnostics = []
    for iteration in actual_iterations:
        path = run/'ppo'/f'model_{iteration-1}.pt'
        state = torch.load(path,map_location='cpu',weights_only=False)
        assert state['iter'] == iteration and not state['native']['canonical_restart']
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values() if torch.is_tensor(value))
        diagnostics.append(path)
    paths = [path for path in snapshot.rglob('*') if path.is_file()]
    paths += [video,video.with_suffix('.json'),video.with_name(video.stem+'_verification.json')]
    paths += sorted(video.parent.glob(video.stem+'_case*.jpg'))
    paths += [path for path in HERE.rglob('*') if path.is_file() and path.suffix in {'.py','.json','.jpg'}]
    paths += [source/'checkpoint.pt',source/'policy.pt',base,*diagnostics]
    paths += [ROOT/name for name in ['configs/full_racket_v4_r3.json','configs/court_net_v4.json',
        'configs/shuttle_physics_v2.json','configs/g07_full_racket_eval_v4_r3.json',
        'configs/g07_full_racket_residual_ppo_centered_v4.json','configs/full_racket_usd_dependencies_v4.json',
        'scripts/render_full_racket_latest_effect_v4.py','scripts/render_full_racket_r3_effect.py',
        'scripts/render_current_v4_progress.py','reports/G07_full_racket_v4/MODEL_RULES.json',
        'reports/G07_full_racket_v4/centered_residual_resume32_queue01/PREDECLARED_PLAN.json',
        'reports/G07_full_racket_v4/centered_residual_resume32_queue01/PARENT_VERIFIED_DECISION.json',
        'reports/G07_full_racket_v4/centered_residual_resume32_queue01/ROOT_STRICT_RESUME_FIRST_UPDATE_CHECK.json']]
    paths = sorted(set(paths)); assert all(path.is_file() for path in paths)
    payload = {str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in paths}
    companion = read(REPORTS/'centered_training_capture_video_20261026/ARCHIVE_UPLOAD_RECEIPT.json')
    manifest = dict(schema='centered-resume-first-complete-native-v1',created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual partial native centered8-to40 training: one complete TRAIN111 lane1519 episode3, two retained failures, all observed4516 raw rows,100Hz motion/actions/public inputs, real video, code and weight custody. No completed job or frozen-policy score.',
        payload=payload,source_canonical8_sha256=sha(source/'checkpoint.pt'),source_public8_sha256=sha(source/'policy.pt'),
        fixed_base_policy_sha256=sha(base),actual_actor_iteration_boundary_diagnostics=actual_iterations,
        weight_limitations='Diagnostic snapshots are noncanonical, not restart checkpoints. Weights, public normalizers and exploration change during collection; recorded native actions are authoritative. Strict re-execution begins at the included original canonical8 and runs the original recorded command/seed with the exact runtime, rather than restoring an in-contact episode.',
        required_runtime_companion={key:companion[key] for key in ['url','bytes','sha256']},
        runtime_note='Original native USD layers, Go2 walking actor and standard PiPER URDF are in the verified215MB companion. Its external USD original-path mapping must be restored or equivalently linked. Native Isaac/RSL environment is required for new robot execution; CPU video replay uses the included actual poses/geometry.',
        source_job_complete=False,frozen_policy_performance=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in paths:
            bundle.write(path,str(path.relative_to(ROOT)))
        bundle.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(payload)+1
        for name,record in payload.items():
            data = bundle.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(archive.relative_to(ROOT)),
        bytes=archive.stat().st_size,sha256=sha(archive),payload_members=len(payload),total_members=len(payload)+1,
        all_member_CRC_verified=True,all_source_payload_SHA_verified=True,source_job_complete=False,
        frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
