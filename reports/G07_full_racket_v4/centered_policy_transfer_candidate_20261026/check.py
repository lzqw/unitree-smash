"""Reproduce CPU transfer evidence from real completed native source data."""
import contextlib
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
MEASURED=ROOT.parent/'unitree_smash'
sys.path.insert(0,str(ROOT))
from scripts.initialize_centered_policy_v4 import apply_centered_public_actor,verify_measured_source
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    output=HERE/'ACTUAL_NATIVE_SOURCE_CPU_CHECK.json';assert not output.exists()
    reports=MEASURED/'reports/G07_full_racket_v4';source=reports/'residual_ppo_centered_base8_n2048_probe8_02'
    evaluation=reports/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
    base_path=reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    native=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    public=torch.load(source/'policy.pt',map_location='cpu',weights_only=False)
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    configuration=read(MEASURED/'configs/g07_full_racket_residual_ppo_centered_v4.json')
    contract=copy.deepcopy(native['infos']['residual_contract'])
    contract['joint_placement_reward']['distance_scale_m']=.75;contract['seed']+=100
    provenance=verify_measured_source(MEASURED,source/'policy.pt',evaluation,contract)
    options=dict(configuration['runner']['policy']);options.pop('class_name')
    obs=TensorDict({'policy':torch.zeros(2,794)},batch_size=[2])
    torch.manual_seed(748901)
    with contextlib.redirect_stdout(io.StringIO()):policy=ActorCritic(obs,configuration['runner']['obs_groups'],18,**options)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    original={name:value.clone()for name,value in policy.state_dict().items()}
    rng=torch.get_rng_state().clone()
    transfer=apply_centered_public_actor(policy,public,native['native']['interface'],contract,base)
    assert torch.equal(rng,torch.get_rng_state()) and not optimizer.state
    retained=[name for name in original if not name.startswith(('actor.','actor_obs_normalizer.'))]
    assert all(torch.equal(original[name],policy.state_dict()[name])for name in retained)
    snapshot=reports/'video_snapshot_centered_resume_first_complete_20261026'
    cases=read(snapshot/'cases.json')['cases'];inputs=[];dependencies=[]
    for case in cases:
        path=snapshot/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        meta=read(path.with_suffix('.json'));assert sha(path)==meta['data_sha256']
        with np.load(path)as data:sample=unpack_public(data['packed_public'])
        assert hashlib.sha256(sample.tobytes()).hexdigest()==meta['full_public794_sha256']
        inputs.append(torch.from_numpy(sample));dependencies.extend([path,path.with_suffix('.json')])
    inputs=torch.cat(inputs);assert inputs.shape==(1890,794)
    actor=torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval();policy.eval()
    with torch.inference_mode():
        expected=actor(inputs);actual=policy.act_inference(TensorDict({'policy':inputs},batch_size=[len(inputs)]))
    assert torch.equal(expected,actual)
    held={}
    for name in ['centered_residual_pilot8_queue02','centered_residual_resume32_queue01','centered_intermediate_selection_queue01']:
        source_map=read(reports/name/'PREDECLARED_PLAN.json')['source_hashes']
        for path,digest in source_map.items():assert sha(MEASURED/path)==digest,path
        held[name]=len(source_map)
    paths=[Path(__file__),ROOT/'scripts/initialize_centered_policy_v4.py',ROOT/'scripts/train_full_racket_residual_ppo_v4.py',
        ROOT/'tests/test_initialize_centered_policy_v4.py',source/'checkpoint.pt',source/'policy.pt',base_path,*dependencies]
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU compatibility preparation in an isolated candidate worktree. Actual source training/evaluation and recorded observations are reused only as provenance/model inputs; no new native execution or performance.',
        source_provenance=provenance,actual_public_inputs=len(inputs),source_actor_normalizer_count_before_new_learning=2097152,
        actual_source_public_mean_max_absolute_error=float((actual-expected).abs().max()),source_public_mean_bitwise_preserved=True,
        fresh_critic_and_exploration_tensors_preserved=retained,critic_normalizer_count=0,fresh_Adam_state_entries=len(optimizer.state),
        torch_CPU_RNG_unchanged_by_model_copy=True,transfer_metadata=transfer,
        in_memory_probe_contract_changes=dict(placement_distance_scale_m=.75,seed_change=True,
            role='CPU compatibility exercise only, not a selected or launched training revision; no config file was changed'),
        main_held_source_maps_unchanged=held,
        source_sha256={str(path):sha(path)for path in paths},
        integration_status='Candidate only; wait for actual parent and intermediate-selection owner exit before changing held main runtime sources.',
        new_native_execution=False,new_training=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key]for key in ['actual_public_inputs','source_public_mean_bitwise_preserved',
        'actual_source_public_mean_max_absolute_error','fresh_Adam_state_entries','main_held_source_maps_unchanged','new_native_execution']}))


if __name__=='__main__':main()
