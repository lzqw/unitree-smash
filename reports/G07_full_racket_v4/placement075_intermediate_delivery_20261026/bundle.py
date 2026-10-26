"""Package terminal075 intermediate evaluations and independently restore-check.

Read completed native snapshots instead of extending custody of live main
sources. This workflow never schedules a GPU job or uploads automatically.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
SELECTOR = REPORTS/'centered_placement075_intermediate_selection_queue01'
TRAIN_NAME = 'residual_ppo_centered_placement075_n2048_resume32_02'
SOURCE_NAME = 'residual_ppo_centered_placement075_n2048_probe8_01'
BASE_NAME = 'residual_ppo_joint_placement_actor32_n2048_probe8_01'
ARCHIVE = REPORTS/'G07_placement075_intermediate16_24_32_frozen_TRAIN128_and_source_diagnostics_20261026.zip'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, iterations, require_bundled=False):
    root = root.resolve(); reports = root/'reports/G07_full_racket_v4'
    sys.path[:0] = [str(root/'vendor/rsl_rl'), str(root)]
    import numpy as np
    import torch
    import rsl_rl
    from tensordict import TensorDict
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from scripts.initialize_centered_policy_v4 import same_tree
    bundled = Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    assert bundled or not require_bundled
    exporter = reports/'centered_placement075_intermediate_selection_queue01/export_diagnostic.py'
    spec = importlib.util.spec_from_file_location('archived075_actor_export', exporter)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before = torch.load(reports/SOURCE_NAME/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert before['iter'] == 8 and before['native']['canonical_restart']
    base = torch.load(reports/BASE_NAME/'policy.pt', map_location='cpu', weights_only=False)
    protocol = read(root/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(root)
    cases, indices = training_cases(root, protocol);assert indices == list(range(128))
    entries = []
    for iteration in iterations:
        folder = reports/f'residual_ppo_centered_placement075_diag_iter{iteration}_fixed_train128_01'
        policy_path = reports/f'centered_placement075_intermediate_selection_queue01/exports/iter{iteration:02d}/policy.pt'
        job, inputs, metrics = [read(folder/name) for name in ['job.json','INPUTS.json','metrics.json']]
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
        assert job['charged_env_steps'] == read(folder/'completed.json')['env_steps'] == 102400
        assert inputs['policy_sha256'] == metrics['policy_sha256'] == sha(policy_path)
        assert inputs['reset_noise_seed'] == metrics['noise_seed'] == 724201
        assert metrics['noise_level'] == 'none' and all(v == 0 for v in inputs['reset_noise_amplitudes'].values())
        summary = summarize_training(read(folder/'episodes.json'), cases, indices, protocol)
        assert all(metrics[key] == value for key,value in summary.items()) and summary['numerical_failures'] == 0
        with np.load(folder/'all_first_episodes.npz') as data:
            mask = data['valid_first_episode'].astype(bool)
            assert data['link_poses'].shape == (400,128,28,7) and mask.shape == (400,128) and mask[0].all()
            assert np.isfinite(data['link_poses'][mask]).all() and np.isfinite(data['ball_position_m'][mask]).all()
        with np.load(folder/'public_and_actions_first2.npz') as data:
            assert data['observations'].shape == (800,2,794) and data['actions'].shape == (800,2,18)
            samples = torch.from_numpy(data['observations'][data['valid_first_episode'].astype(bool)].copy())
        native_path = reports/TRAIN_NAME/f'ppo/model_{iteration-1}.pt'
        native = torch.load(native_path, map_location='cpu', weights_only=False)
        assert native['iter'] == iteration and not native['native']['canonical_restart']
        assert native['native']['training_signature'] == before['native']['training_signature']
        for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
            assert int(native['model_state_dict'][name])-int(before['model_state_dict'][name]) == (iteration-8)*262144
        a,b = before['optimizer_state_dict']['state'],native['optimizer_state_dict']['state']
        assert len(b) == 17 and a.keys() == b.keys()
        assert all(float(b[key]['step'])-float(a[key]['step']) == (iteration-8)*20 for key in a)
        assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
        public = torch.load(policy_path, map_location='cpu', weights_only=False)
        assert same_tree(public['base_controller'],base) and not public['performance_inherited']
        actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
        rebuilt = module.policy_from_native(native)
        with torch.no_grad():
            expected = rebuilt.act_inference(TensorDict({'policy':samples},batch_size=[len(samples)]))
            error = float((actor(samples)-expected).abs().max())
        assert error <= 2e-5
        entries.append(dict(iteration=iteration, summary=summary, public_policy_sha256=sha(policy_path),
            source_diagnostic_sha256=sha(native_path), actual_public_samples=len(samples), actor_export_max_error=error,
            all17_Adam_states_advanced=(iteration-8)*20, actor_RMS_count=int(native['model_state_dict']['actor_obs_normalizer.count']),
            actual_GPU_hours=job['charged_gpu_hours']))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), assessments=entries,
        bundled_RSL_used=bundled, original_all128_recomputed=True, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)


def build():
    assert not Path('/proc/1744903').exists(), 'Wait for actual comparison owner exit'
    assert read(SELECTOR/'state.json')['state'] == 'COMPLETED_INTERMEDIATE_SELECTION'
    selection = read(SELECTOR/'RESULT.json');entries = selection['measured_candidates']
    assert {entry['iteration'] for entry in entries} == {8,16,24,32,40}
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.zip.partial').exists()
    check = validate(ROOT,[16,24,32])
    members = {}
    def add(name, data):
        assert name not in members or members[name] == data, 'Conflicting exact source versions: '+name
        members[name] = data
    def file(path):add(str(path.relative_to(ROOT)),path.read_bytes())
    for iteration in [16,24,32]:
        folder = REPORTS/f'residual_ppo_centered_placement075_diag_iter{iteration}_fixed_train128_01'
        source_hashes = read(folder/'source_hashes.json')
        with zipfile.ZipFile(folder/'source_snapshot.zip') as archive:
            for name,digest in source_hashes.items():
                data = archive.read(name);assert hashlib.sha256(data).hexdigest() == digest;add(name,data)
        for path in folder.rglob('*'):
            if path.is_file() and '.partial' not in path.name:file(path)
        for path in (SELECTOR/f'exports/iter{iteration:02d}').rglob('*'):
            if path.is_file():file(path)
        file(REPORTS/TRAIN_NAME/f'ppo/model_{iteration-1}.pt')
    for path in SELECTOR.iterdir():
        if path.is_file() and path.suffix in ['.py','.json','.csv']:file(path)
    for name in ['checkpoint.pt','policy.pt','INPUTS.json','metrics.json','job.json','completed.json']:
        file(REPORTS/SOURCE_NAME/name)
    file(REPORTS/BASE_NAME/'policy.pt')
    middle = ROOT/'reports/G07_placement075_native_intermediate16_24_CPU_export_evidence_20261026.zip'
    assert sha(middle) == '18f53e0d36fb8ed8a9d9ea2f42a7952b8908e8631813079ffb07f4f65a3a9770'
    with zipfile.ZipFile(middle) as archive:
        manifest = json.loads(archive.read('CPU_EXPORT_MANIFEST.json'))
        for name in archive.namelist():
            if name.startswith('vendor/rsl_rl/'):
                data = archive.read(name);assert hashlib.sha256(data).hexdigest() == manifest['payloads'][name]['sha256'];add(name,data)
    file(Path(__file__))
    companion = read(REPORTS/'placement075_completed40_data_queue01/UPLOAD_RECEIPT.json')
    assert companion['download_verified']
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), assessments=check,
        original075_five_source_selection=selection,
        source_files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in members.items()},
        mandatory_native_runtime_companion={key:companion[key] for key in ['url','bytes','sha256']},
        companion_scope='Full assets/external USD mapping/walking actor and original training. CPU comparison data, three source diagnostics and RSL are included here.',
        exact_native_launch_sources_used=True, main_runtime_custody_extended=False,
        diagnostic_states_are_not_native_restart_points=True, formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    temporary = ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in members.items():archive.writestr(name,data,compress_type=zipfile.ZIP_STORED if name.endswith(('.npz','.zip')) else zipfile.ZIP_DEFLATED)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(members)+1
        for name,entry in manifest['source_files'].items():
            data=archive.read(name);assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    temporary.rename(ARCHIVE)
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,
        sha256=sha(ARCHIVE),all_CRC_and_payload_SHA_verified=True,source_payloads=len(members),
        native_launch_sources_preserved=True,new_native_execution=False,formal_acceptance=False,goal_complete=False)
    (HERE/'ARCHIVE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-root',type=Path)
    parser.add_argument('--iterations',nargs='+',type=int,default=[16,24,32])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-bundled',action='store_true')
    args=parser.parse_args()
    if args.check_root:
        assert args.output and not args.output.exists()
        result=validate(args.check_root,args.iterations,args.require_bundled)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    else:build()
