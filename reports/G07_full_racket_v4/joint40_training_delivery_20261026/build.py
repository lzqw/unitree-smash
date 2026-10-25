"""Bundle completed native joint40 TRAIN evidence; never include live evals."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
TRAIN=REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01'
MILESTONE=REPORTS/'joint_placement_first_training_complete_20261026'
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
ARCHIVE=HERE/'G07_joint40_training_raw_and_contact_evidence_20261026.zip'


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    assert not ARCHIVE.exists() and not (HERE/'MANIFEST.json').exists()
    job=read(TRAIN/'job.json');assert job['status']=='SUCCEEDED' and job['exit_code']==0
    assert not Path(f'/proc/{job["pid"]}').exists() and not Path(f'/proc/{job["child_pid"]}').exists()
    with (HERE/'csv_export.log').open('x')as log:
        subprocess.run([PYTHON,'scripts/export_full_racket_train_csv_v4.py','--run',str(TRAIN),
            '--output-prefix',str(HERE/'joint40_training_episodes')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=600)
    assert read(HERE/'joint40_training_episodes.json')['original_runtime_metrics_match']
    selected=set()
    for path in TRAIN.iterdir():
        if path.is_file() and (path.suffix in {'.json','.pt','.zip','.patch','.log'}):
            selected.add(path)
    for path in MILESTONE.iterdir():
        if path.is_file():selected.add(path)
    for item in read(MILESTONE/'weight_lineage.json'):
        path=ROOT/item['path'];assert sha(path)==item['sha256'];selected.add(path)
    for name in ['residual_ppo_joint_placement_actor32_n2048_probe8_01/checkpoint.pt',
        'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',
        'cem32_lateral_feedback_1015/best_candidate.pt']:
        selected.add(REPORTS/name)
    for name in ['joint_placement_resume32_queue01','joint_placement_contact_confirmation_queue01',
        'joint_placement_iter40_recovery_20261026','joint_placement_iter40_video_recovery_20261026',
        'live_tangent_trial_recovery_20261026']:
        for path in (REPORTS/name).iterdir():
            if path.is_file() and path.suffix in {'.json','.py','.txt'} and path.name!='state.json':selected.add(path)
    # The original failed historical state is terminal and preserved; live
    # recovery state files and incomplete frozen40 results are excluded.
    selected.add(REPORTS/'joint_placement_resume32_queue01/state.json')
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    selected.add(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    selected.add(ROOT/'configs/g07_full_racket_residual_ppo_joint_placement_v4.json')
    for name in protocol['physical_configs']:selected.add(ROOT/name)
    for name in read(TRAIN/'asset_dependencies.json')['resolved_layers']:selected.add(Path(name))
    for path in [ROOT/'quadsmash/assets/pretrained/go2_walk.pt',ROOT/'artifacts/pretrained/go2_walk.pt']:
        if path.is_file():selected.add(path)
    # Read the actual controller initialization's walking-actor reference.
    import sys
    sys.path.insert(0,str(ROOT))
    import torch
    base=torch.load(REPORTS/'cem32_lateral_feedback_1015/best_candidate.pt',map_location='cpu',weights_only=False)
    for value in base.values():
        if isinstance(value,str) and value.endswith('.pt'):
            path=Path(value);path=path if path.is_absolute()else ROOT/path
            if path.is_file():selected.add(path)
    selected.update([HERE/'joint40_training_episodes.csv',HERE/'joint40_training_episodes.json',Path(__file__).resolve()])
    entries=[]
    for path in sorted(selected,key=str):
        assert path.is_file()
        name=('unitree_smash/'+str(path.relative_to(ROOT)))if path.is_relative_to(ROOT)else('external_layers/'+str(path).lstrip('/'))
        entries.append(dict(source_path=str(path),member=name,bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed changing-policy joint-placement learner40 training, actual contact confirmations, weights, raw episodes and source. Frozen40 assessment and videos are still running and excluded.',
        original_training_job_git_sha=job['git_sha'],report_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        actual_control_env_steps=job['charged_env_steps'],actual_GPU_hours=job['charged_gpu_hours'],
        native_training_counts=read(TRAIN/'metrics.json')['counts'],
        original_training_motion_trajectory_available=False,
        complete_episodes_crossed_actor_updates=True,no_single_frozen_winning_checkpoint_claimed=True,
        asset_layer_path_mapping='Workspace assets are under unitree_smash; four original absolute Go2 dependency paths are preserved under external_layers. Restore or explicitly remap their original absolute USD references before native replay.',
        previous_complete_initialization_archive=dict(
            url='https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint_reward_probe8_native_evidence_and_next_trial_20261025.zip',
            sha256='3f90e2796dac59fd00d45a0d0f3acef0be9f36401b5a063d275296840347955c'),
        separate_unexecuted_v3_candidate='https://github.com/lzqw/unitree-smash/tree/e24e1c0',
        members=entries,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    manifest_path=HERE/'MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for entry in entries:
            archive.write(entry['source_path'],entry['member'],compress_type=zipfile.ZIP_STORED if entry['member'].endswith(('.gz','.zip'))else None)
        archive.write(manifest_path,'MANIFEST.json')
    with zipfile.ZipFile(ARCHIVE)as archive:
        assert archive.testzip()is None
        for entry in entries:
            h=hashlib.sha256()
            with archive.open(entry['member'])as stream:
                for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
            assert h.hexdigest()==entry['sha256']
        assert len(archive.namelist())==len(entries)+1
    result=dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),members=len(entries)+1,
        all_members_CRC_and_SHA256_verified=True,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
