"""CPU witness for explicit action-unit mapping; does not select a future source."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'
sys.path.insert(0,str(ROOT))
from scripts.initialize_centered_action_units_v4 import apply_action_unit_mapped_actor
from scripts.initialize_centered_policy_v4 import verify_measured_source
from quadsmash.rl.episode_trace_v4 import unpack_public


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'ACTUAL_FROZEN40_CPU_ACTION_UNIT_CHECK.json';assert not output.exists()
    reports = MAIN/'reports/G07_full_racket_v4'
    source = reports/'residual_ppo_centered_base8_n2048_resume32_01'
    evaluation = reports/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
    base_path = reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    native = torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    public = torch.load(source/'policy.pt',map_location='cpu',weights_only=False)
    base = torch.load(base_path,map_location='cpu',weights_only=False)
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json'
    config = read(config_path)
    contract = copy.deepcopy(native['infos']['residual_contract'])
    contract.update(joint_scale_rad=config['joint_scale_rad'],joint_placement_reward=config['joint_placement_reward'],seed=749301)
    provenance = verify_measured_source(MAIN,source/'policy.pt',evaluation,contract)
    options = dict(config['runner']['policy']);options.pop('class_name')
    torch.manual_seed(749301)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),config['runner']['obs_groups'],18,**options)
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
    before = {name:value.clone() for name,value in policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    snapshot = reports/'video_snapshot_centered_resume_first_complete_20261026'
    samples,dependencies = [],[]
    for case in read(snapshot/'cases.json')['cases']:
        path = snapshot/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        metadata = read(path.with_suffix('.json'));assert sha(path) == metadata['data_sha256']
        with np.load(path) as a:observations = unpack_public(a['packed_public'])
        assert hashlib.sha256(observations.tobytes()).hexdigest() == metadata['full_public794_sha256']
        samples.append(torch.from_numpy(observations));dependencies += [path,path.with_suffix('.json')]
    training_sample = torch.cat(samples);assert training_sample.shape == (1890,794)
    evaluated_path = evaluation/'public_and_actions_first2.npz'
    with np.load(evaluated_path) as a:
        evaluated_sample = torch.from_numpy(a['observations'][a['valid_first_episode'].astype(bool)].copy())
    witness = torch.cat([training_sample,evaluated_sample])
    result = apply_action_unit_mapped_actor(policy,public,native['native']['interface'],contract,base,witness)
    assert not optimizer.state and torch.equal(rng,torch.get_rng_state())
    preserved = [name for name in before if name.startswith(('critic.','critic_obs_normalizer.'))]
    assert all(torch.equal(before[name],policy.state_dict()[name]) for name in preserved)
    policy.eval();actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        old = actor(witness).clamp(-1,1)*torch.tensor(public['residual_joint_scale_rad'])
        new = policy.act_inference(TensorDict({'policy':witness},batch_size=[len(witness)])).clamp(-1,1)*torch.tensor(config['joint_scale_rad'])
    error = float((new-old).abs().max());assert error <= 2e-6
    old_std = torch.tensor(public['residual_joint_scale_rad'])*.1
    new_std = torch.tensor(config['joint_scale_rad'])*policy.log_std.detach().exp()
    assert torch.allclose(old_std,new_std,atol=1e-8,rtol=0)
    active_plan_path = reports/'centered_placement075_pilot8_queue01/PREDECLARED_PLAN.json'
    held = read(active_plan_path)['source_hashes']
    held[str(active_plan_path.relative_to(MAIN))] = sha(active_plan_path)
    for name,digest in held.items():
        p = Path(name) if Path(name).is_absolute() else MAIN/name
        assert sha(p) == digest,name
    for pid,needle in [(1722226,'centered_placement075_pilot8_queue01/run.py'),(1722244,'residual_ppo_centered_placement075_n2048_probe8_01')]:
        p = Path('/proc')/str(pid)/'cmdline';assert p.exists() and needle.encode() in p.read_bytes()
    paths = [Path(__file__),ROOT/'scripts/initialize_centered_action_units_v4.py',ROOT/'scripts/train_full_racket_residual_ppo_v4.py',
        ROOT/'tests/test_initialize_centered_action_units_v4.py',config_path,source/'checkpoint.pt',source/'policy.pt',base_path,evaluated_path,*dependencies]
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Conditional candidate only. Actual measured40 is a CPU witness, not a selected future source; ongoing075 has no completed frozen score.',
        source_provenance=provenance,actual_training_witness_rows=len(training_sample),actual_first2_evaluation_witness_rows=len(evaluated_sample),
        actual_total_witness_rows=len(witness),actual_physical_residual_mean_max_error_rad=error,
        actual_physical_residual_means_bitwise_equal=torch.equal(old,new),
        fresh_preclipping_physical_std_reference_error_rad=float((new_std-old_std).abs().max()),
        fresh_critic_and_critic_normalizer_preserved=preserved,fresh_Adam_entries=len(optimizer.state),torch_CPU_RNG_unchanged=True,
        transfer=result,main_held_source_count=len(held),all_main_held_sources_unchanged=True,
        source_sha256={str(p):sha(p) for p in paths},new_native_execution=False,new_training=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with output.open('x') as file:json.dump(record,file,indent=2);file.write('\n')
    print(json.dumps({key:record[key] for key in ['actual_total_witness_rows','actual_physical_residual_mean_max_error_rad',
        'actual_physical_residual_means_bitwise_equal','fresh_preclipping_physical_std_reference_error_rad','main_held_source_count','new_native_execution']}))


if __name__ == '__main__':main()
