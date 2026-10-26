"""Check the completed075 delivery from a fresh extraction using restored code."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

CHECK = r'''
import hashlib,json,pathlib,sys,torch
from scripts.initialize_centered_policy_v4 import verify_measured_source
from quadsmash.rl.frozen_baseline_source_v4 import verify_frozen_baseline_source
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
import quadsmash
root=pathlib.Path(sys.argv[1]);assert pathlib.Path(quadsmash.__file__).resolve().is_relative_to(root)
r=root/'reports/G07_full_racket_v4';train=r/'residual_ppo_centered_placement075_n2048_probe8_01'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(root/'MANIFEST.json')
for name,entry in manifest['source_files'].items():assert sha(root/name)==entry['sha256'],name
base=r/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
base_policy=torch.load(base,map_location='cpu',weights_only=False)
inputs=read(train/'INPUTS.json');contract=inputs['contract']
baseline=verify_frozen_baseline_source(base,base_policy,contract['protocol_sha256'])
source=verify_measured_source(root,r/'residual_ppo_centered_base8_n2048_resume32_01/policy.pt',r/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01',contract)
assert source['source_policy_sha256']==inputs['initialization']['source_policy_sha256']
native=torch.load(train/'checkpoint.pt',map_location='cpu',weights_only=False)
policy=torch.load(train/'policy.pt',map_location='cpu',weights_only=False)
assert native['iter']==8 and native['native']['canonical_restart']
assert native['native']['total_timesteps']==2097152
assert int(native['model_state_dict']['actor_obs_normalizer.count'])==12582912
assert int(native['model_state_dict']['critic_obs_normalizer.count'])==2097152
assert all(torch.isfinite(v).all()for v in native['model_state_dict'].values()if torch.is_tensor(v))
assert len(native['optimizer_state_dict']['state'])==17
assert sha(train/'checkpoint.pt')==manifest['canonical_checkpoint_sha256']
assert sha(train/'policy.pt')==manifest['policy_sha256']
assert policy['base_controller']['residual_actor_jit']==base_policy['residual_actor_jit']
protocol=read(root/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(root)
cases,indices=training_cases(root,protocol)
summary=summarize_training(read(r/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01/episodes.json'),cases,indices,protocol)
assert summary==manifest['independent_frozen_TRAIN128']
for layer in manifest['external_runtime_layers']:assert sha(root/layer['archive_member'])==layer['sha256']
print(json.dumps(dict(restored_module_path=str(quadsmash.__file__),payloads_rehashed=len(manifest['source_files']),base_provenance=baseline,source40_provenance=source,canonical_stage8_finite=True,all128_independent_results_match=True,external_layers_available_and_hashed=len(manifest['external_runtime_layers']),native_replay_executed=False,formal_acceptance=False,goal_complete=False)))
'''


def main():
    record = json.loads((HERE / 'ARCHIVE_CHECK.json').read_text())
    archive_path = ROOT / record['archive']
    restored = Path(tempfile.mkdtemp(prefix='g07_placement075_restored_20261026_'))
    source_member = 'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_01/source_snapshot.zip'
    with zipfile.ZipFile(archive_path) as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read(source_member))) as sources:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in sources.namelist())
            sources.extractall(restored)
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
        archive.extractall(restored)
    process = subprocess.run(['/home/autodrive/Isaac/env_isaaclab/bin/python', '-c', CHECK, str(restored)],
        cwd=restored, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONPATH=str(restored)),
        check=True, text=True, capture_output=True)
    result = json.loads(process.stdout)
    result.update(checked_at=datetime.now(timezone.utc).isoformat(), archive_sha256=record['sha256'],
                  fresh_extraction=str(restored), restored_code_used=True,
                  scope='CPU provenance, data and canonical weight checks in a fresh extraction; no native physics replay or formal acceptance.')
    (HERE / 'RESTORE_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
