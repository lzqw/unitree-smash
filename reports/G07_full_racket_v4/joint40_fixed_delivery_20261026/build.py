"""Bundle both real frozen diagnostics and reviewed failed joint40 videos."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    output=HERE/'G07_joint40_frozen_TRAIN128_failed_videos_and_diagnostics_20261026.zip'
    assert not output.exists() and not (HERE/'MANIFEST.json').exists()
    selected={Path(__file__).resolve()}
    for name in ['residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01',
        'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01']:
        run=REPORTS/name;assert read(run/'job.json')['status']=='SUCCEEDED'
        for path in run.iterdir():
            if path.is_file() and '.partial.'not in path.name:selected.add(path)
    for name in ['joint40_video_review_20261026','joint40_export_diagnostic_20261026','joint40_fixed_diagnostic_20261026']:
        for path in (REPORTS/name).iterdir():
            if path.is_file():selected.add(path)
    review=read(REPORTS/'joint40_video_review_20261026/VIDEO_UPLOAD_RECEIPT.json')
    assert all(video['download_verified']for video in review['videos'])
    for video in review['videos']:
        path=ROOT/video['video'];selected.update([path,path.with_suffix('.json')])
    for name in ['residual_ppo_joint_placement_actor32_n2048_resume32_01/checkpoint.pt',
        'residual_ppo_joint_placement_actor32_n2048_resume32_01/policy.pt',
        'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',
        'cem32_lateral_feedback_1015/best_candidate.pt',
        'CPU_REPLAY_AUTHORED_JOINTS_0724.json',
        'joint_placement_iter40_recovery_20261026/FIXED_TRAIN128_RESULT.json',
        'joint_placement_iter40_recovery_20261026/joint_iter40_fixed_episodes.csv',
        'joint_placement_iter40_recovery_20261026/joint_iter40_fixed_episodes.json']:
        selected.add(REPORTS/name)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';selected.add(protocol_path)
    selected.update(ROOT/name for name in read(protocol_path)['physical_configs'])
    entries=[dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))for path in sorted(selected,key=str)]
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed real frozen TRAIN128 failure of joint40, both actual reviewed videos, all paired8/40 rows/motion and CPU diagnostic programs. No formal A/B or successful training-movie claim.',
        frozen40_policy_sha256=sha(REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/policy.pt'),
        frozen40_counts=dict(episodes=128,airborne=0,conditional_recovery=0,complete=0,numerical_failures=0),
        frozen8_counts=dict(episodes=128,airborne=20,conditional_recovery=12,complete=0,numerical_failures=0),
        external_runtime_assets='Actual composed USD layers, walking actor and full changing-policy training provenance are retained in the separately verified joint40 training bundle. Restore/map its absolute external layers before native replay.',
        runtime_asset_bundle=read(REPORTS/'joint40_training_delivery_20261026/UPLOAD_RECEIPT.json'),
        selected_source_files=entries,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    manifest=HERE/'MANIFEST.json';manifest.write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for entry in entries:archive.write(ROOT/entry['path'],entry['path'],compress_type=zipfile.ZIP_STORED if entry['path'].endswith(('.zip','.mp4','.jpg'))else None)
        archive.write(manifest,'MANIFEST.json')
    with zipfile.ZipFile(output)as archive:
        assert archive.testzip()is None
        for entry in entries:
            digest=hashlib.sha256()
            with archive.open(entry['path'])as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==entry['sha256']
    proof=dict(archive=str(output),bytes=output.stat().st_size,sha256=sha(output),members=len(entries)+1,
        all_members_CRC_and_SHA256_verified=True,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
