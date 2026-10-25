"""Archive completed frozen TRAIN diagnostics and the reviewed six-case video.

Concurrent intermediate16/24/32 assessments contribute no outcomes to this package.
Native runtime USDs/walking actor/canonical8 are in the verified companion.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports=ROOT/'reports/G07_full_racket_v4'
    runs=[reports/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02',
        reports/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01']
    policies=[reports/'residual_ppo_centered_base8_n2048_probe8_02/policy.pt',
        reports/'residual_ppo_centered_base8_n2048_resume32_01/policy.pt']
    diagnostic=reports/'centered8_to40_fixed_diagnostic_20261026'
    snapshot=reports/'video_snapshot_centered_residual_base8_iter40_20261026'
    video=reports/'video/Current_effect_centered_residual_base8_iter40_20261026.mp4'
    output=reports/'G07_centered8_to40_frozen_TRAIN128_video_and_diagnostics_20261026.zip'
    assert not output.exists()and not(HERE/'ARCHIVE_CHECK.json').exists()
    assert read(HERE/'VISUAL_REVIEW.json')['approved_for_authorized_GitHub_video_upload']and read(HERE/'VIDEO_UPLOAD_RECEIPT.json')['download_verified']
    companion=read(reports/'centered_training_capture_video_20261026/ARCHIVE_UPLOAD_RECEIPT.json')
    assert companion['download_verified']and companion['sha256']=='d672edf51ec90649b1b0381fe9ff861e0c2433c98de944e0a07937e52954c4e4'
    paths=[]
    for run in runs:
        job=read(run/'job.json');assert job['status']=='SUCCEEDED'and job['exit_code']==0 and job['receipt_present']
        paths += [p for p in run.rglob('*')if p.is_file()and'.partial'not in p.name]
    paths+=policies+list(snapshot.rglob('*'))+[video,video.with_suffix('.json'),video.with_name(video.stem+'_verification.json')]
    paths+=list(video.parent.glob(video.stem+'_case*.jpg'))
    paths+=list((HERE/'decoded_samples').glob('*.jpg'))
    paths+=[reports/'residual_ppo_centered_base8_n2048_probe8_02/checkpoint.pt',reports/'residual_ppo_centered_base8_n2048_resume32_01/checkpoint.pt',reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt']
    paths+=[p for p in diagnostic.iterdir()if p.is_file()and p.suffix in ['.py','.json','.csv']]
    paths+=[p for p in HERE.iterdir()if p.is_file()and p.suffix in ['.py','.json','.jpg']]
    queue=reports/'centered_residual_resume32_queue01'
    paths+=[queue/name for name in ['PREDECLARED_PLAN.json','FIXED_TRAIN128_RESULT.json','centered_fixed_episodes.csv','centered_fixed_episodes.json','TRAINING_RECEIPT.json','TRAINING_TRACE_AUDIT.json']]
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    paths+=[ROOT/'configs/g07_full_racket_eval_v4_r3.json',ROOT/protocol['training_suite']['path'],reports/'MODEL_RULES.json']
    paths+=[ROOT/name for name in protocol['physical_configs']]
    paths+=[ROOT/'scripts'/name for name in ['render_full_racket_latest_effect_v4.py','render_full_racket_r3_effect.py','render_current_v4_progress.py','prepare_full_racket_latest_video_v4.py']]
    members={str(p.relative_to(ROOT)):p for p in sorted(set(paths))if p.is_file()}
    for name,digest in read(diagnostic/'REPORT.json')['source_sha256'].items():assert sha(ROOT/name)==digest,name
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed frozen centered learners8 and40 on all originalTRAIN128 cases with paired diagnostics and actual40 video. Concurrent intermediate16/24/32 assessments excluded; no formal A/B score.',
        new_policy_sha256=sha(policies[1]),previous_centered8_policy_sha256=sha(policies[0]),
        fixed_TRAIN128=read(queue/'FIXED_TRAIN128_RESULT.json'),
        actual_video_duration_s=84.66,actual_video_frames=4233,
        required_companion=dict(url=companion['url'],bytes=companion['bytes'],sha256=companion['sha256'],
            purpose='Original native runtime USD layers,walking actor/URDF,launch-time source and actual canonical8 restart. Extract it at the same repository root before native replay.'),
        source_files={name:dict(bytes=path.stat().st_size,sha256=sha(path))for name,path in members.items()},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as archive:
        for name,path in members.items():
            archive.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix in ['.zip','.npz','.jpg','.mp4']else zipfile.ZIP_DEFLATED)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output)as archive:
        assert archive.testzip()is None
        for name,entry in manifest['source_files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
        sha256=sha(output),source_payload_members=len(members),total_members=len(members)+1,
        all_member_CRC_and_source_payload_SHA_verified=True,active_intermediate_assessments_excluded=True,
        companion_archive_sha256=companion['sha256'],formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
