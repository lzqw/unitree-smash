"""Package the completed centered TRAIN pilot and actual exploratory video.

The currently running frozen TRAIN128 assessment is intentionally excluded.
No native or historical record is modified by this read-only packaging step.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    reports=ROOT/'reports/G07_full_racket_v4'
    run=reports/'residual_ppo_centered_base8_n2048_probe8_02';queue=reports/'centered_residual_pilot8_queue02'
    baseline=reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01'
    snapshot=reports/'video_snapshot_centered_training_capture_20261026_v2'
    video=reports/'video/Current_training_centered_initial_exploration_20261026.mp4'
    output=reports/'G07_centered_residual_probe8_native_training_and_initial_video_20261026_v2.zip'
    assert not output.exists()and not(HERE/'ARCHIVE_CHECK_V2.json').exists()
    job=read(run/'job.json');assert job['status']=='SUCCEEDED'and job['receipt_present']and job['exit_code']==0
    assert read(run/'completed.json')['env_steps']==2097152
    assert read(queue/'TRACE_AUDIT_ATTEMPT.json')['passed']and read(queue/'TRAINING_TRACE_AUDIT.json')['selected_episodes_checked']==12
    assert read(HERE/'VISUAL_REVIEW.json')['ready_to_upload']and read(HERE/'VIDEO_UPLOAD_RECEIPT.json')['download_verified']
    assert read(HERE/'native_train8_episodes.json')['original_runtime_metrics_match']
    paths=[p for p in run.iterdir()if p.is_file()and p.name!='episodes.partial.json']
    paths+=list((run/'ppo').glob('model_*.pt'))+list((run/'trajectory_traces').glob('*'))
    paths+=list((run/'ppo/git').rglob('*'))
    paths+=[p for p in queue.iterdir()if p.is_file()and p.suffix in ['.json','.xml','.py']and p.name!='state.json']
    paths+=list(snapshot.rglob('*'))+[video,video.with_suffix('.json')]
    paths+=list(video.parent.glob(video.stem+'_case*.jpg'))
    paths+=[p for p in HERE.iterdir()if p.is_file()and p.suffix in ['.py','.json','.csv','.jpg']and p.name!='ARCHIVE_CHECK.json']
    paths+=[baseline/name for name in ['policy.pt','job.json','INPUTS.json','completed.json','metrics.json','asset_dependencies.json']]
    paths += [ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json',ROOT/'configs/g07_full_racket_eval_v4_r3.json',
        ROOT/'configs/court_net_v4.json',reports/'MODEL_RULES.json']
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    paths+=[ROOT/protocol['training_suite']['path']]
    paths+=[ROOT/name for name in protocol['physical_configs']]
    paths+=list(ROOT.glob('assets/**/*.urdf'))
    walking_manifest=ROOT/'configs/g07_go2_walk_transfer.json';walking=read(walking_manifest)
    walking_actor=ROOT/walking['policy_path'];assert sha(walking_actor)==walking['policy_sha256']
    paths+=[walking_manifest,walking_actor]
    layers=[Path(name)for name in read(run/'asset_dependencies.json')['resolved_layers']];paths+=layers
    members={};external=[]
    for path in sorted(set(paths)):
        if not path.is_file():continue
        name=str(path.relative_to(ROOT))if path.is_relative_to(ROOT)else'runtime_external/'+str(path).lstrip('/')
        assert name not in members;members[name]=path
        if not path.is_relative_to(ROOT):external.append(dict(original_absolute_path=str(path),archive_member=name,sha256=sha(path)))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed native centered learner8 TRAIN job and actual initial exploratory failure video. Frozen TRAIN128 assessment is still running and contributes no performance to this archive.',
        actual_control_env_steps=2097152,actual_GPU_hours=job['charged_gpu_hours'],
        actual_changing_policy_counts=read(run/'metrics.json')['counts'],
        canonical_restart_checkpoint=str((run/'checkpoint.pt').relative_to(ROOT)),
        new_policy_sha256=sha(run/'policy.pt'),new_checkpoint_sha256=sha(run/'checkpoint.pt'),
        fixed_public_base_policy_sha256=sha(baseline/'policy.pt'),
        video_scope='TRAIN107 and TRAIN8 first exploratory episodes across actor iterations0..6; both fail. Not the final frozen learner8 execution.',
        selected_training_traces_checked=12,video_all_frames_decoded=1109,
        source_files={name:dict(sha256=sha(path),bytes=path.stat().st_size)for name,path in members.items()},
        external_runtime_layers=external,
        original_walking_actor=dict(path=walking['policy_path'],sha256=walking['policy_sha256'],manifest=str(walking_manifest.relative_to(ROOT)),role='Unchanged source walking prior used under actual full payload; no new physical performance inherited.'),
        external_runtime_restore_note='The full-racket USD references the original Go2 layer paths. Preserve their byte-identical content and restore those paths or equivalent symlinks; do not silently edit physical parameters. Recorded-video replay only needs the included snapshot geometry/poses.',
        no_frozen_assessment_files_included=True,no_formal_A_or_B_cases_read=True,
        formal_acceptance=False,goal_complete=False)
    notes='This archive contains completed native TRAIN learning evidence, not formal acceptance.\n'
    notes+='Actual training:2097152 control-env steps,8 updates,0.6707108824566886 GPU hours;3703 closed episodes,311 air,112 conditional recoveries,0 complete,NF0.\n'
    notes+='The22.18-second video shows two initial exploratory failures; it is not a demonstration of the final frozen8 actor.\n'
    notes+='All12 selected100Hz traces passed the original final trace/raw-contact audit. Full public794 inputs can be reconstructed from packed_public with quadsmash.rl.episode_trace_v4.unpack_public.\n'
    notes+='For CPU video replay use the included video_snapshot_centered_training_capture_20261026_v2 with scripts/render_full_racket_latest_effect_v4.py and Mesa/llvmpipe. The original prepared meshes and all recorded poses are included.\n'
    notes+='checkpoint.pt is the actual canonical ground restart. ppo/model_i.pt files are diagnostic snapshots and cannot be used as strict native resumes.\n'
    notes+='source_snapshot.zip retains the launch-time runtime code. MANIFEST.json lists exact file hashes and external Go2 USD layer locations. Isaac Lab,Isaac Sim and the locked RSL-RL environment remain required for native execution.\n'
    notes+='The original8 policy is only the fixed public base. Its performance is not inherited. New frozen TRAIN128 and formal A/B200 results are absent from this archive.\n'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as archive:
        for name,path in members.items():
            compression=zipfile.ZIP_STORED if path.suffix in ['.zip','.npz','.mp4','.jpg']else zipfile.ZIP_DEFLATED
            archive.write(path,name,compress_type=compression)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        archive.writestr('REPRODUCE.txt',notes)
    with zipfile.ZipFile(output)as archive:
        assert archive.testzip()is None
        for name,entry in manifest['source_files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
        sha256=sha(output),source_members=len(members),total_members=len(members)+2,all_member_SHA_and_CRC_verified=True,
        frozen_assessment_excluded=True,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE_CHECK_V2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
