"""Validate fresh leg15 exploration on real source observations, CPU only."""
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import py_compile
import sys
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.initialize_centered_action_units_leg15_explore_v4 import apply_action_unit_mapped_actor
from scripts.initialize_centered_policy_v4 import verify_measured_source


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result_path=HERE/'ACTUAL_SOURCE_MEAN_AND_STD_CHECK.json';assert not result_path.exists()
    directory=MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_01'
    evaluation=MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
    base_path=MAIN/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    cfg_path=ROOT/'configs/g07_full_racket_residual_ppo_leg15_explore_v4.json'
    config=read(cfg_path)
    canonical=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    contract=dict(canonical['infos']['residual_contract'],joint_scale_rad=config['joint_scale_rad'],
        fresh_exploration_std=config['fresh_exploration_std'],seed=750201,actor_warmup_updates=7)
    proof=verify_measured_source(MAIN,directory/'policy.pt',evaluation,contract)
    with np.load(evaluation/'public_and_actions_first2.npz') as arrays:
        witness=torch.from_numpy(arrays['observations'][arrays['valid_first_episode'].astype(bool)].copy())
    assert witness.shape == (1600,794)
    options=dict(config['runner']['policy']);options.pop('class_name')
    torch.manual_seed(750201)
    with contextlib.redirect_stdout(io.StringIO()):
        policy=ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),
            config['runner']['obs_groups'],18,**options)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    result=apply_action_unit_mapped_actor(policy,checkpoint,canonical['native']['interface'],contract,base,witness)
    assert result['actual_witness_physical_mean_max_error_rad'] <= 2e-6
    actual_std=policy.log_std.detach().exp()
    assert torch.allclose(actual_std,torch.tensor([.2]*12+[.1]*6),atol=1e-7,rtol=0)
    physical_std=actual_std*torch.tensor(config['joint_scale_rad'])
    assert torch.allclose(physical_std,torch.tensor([.03]*12+[.015]*6),atol=1e-7,rtol=0)
    assert not optimizer.state and int(policy.critic_obs_normalizer.count)==0
    assert int(policy.actor_obs_normalizer.count)==12582912
    native=ROOT/'scripts/train_full_racket_residual_ppo_leg15_explore_v4.py'
    helper=ROOT/'scripts/initialize_centered_action_units_leg15_explore_v4.py'
    for path in [Path(__file__),native,helper]:py_compile.compile(str(path),doraise=True)
    save=dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in
        [Path(__file__),native,helper,cfg_path,directory/'policy.pt',directory/'checkpoint.pt',
         base_path,evaluation/'public_and_actions_first2.npz']},actual_source_provenance=proof,transfer=result,
        actual_physical_exploration_std_rad=physical_std.tolist(),fresh_Adam_entries=0,
        fresh_critic_RMS_count=0,source_actor_RMS_count=12582912,
        proposed_total_updates=24,critic_warmup_updates=7,proposed_actual_learning_updates=17,
        proposed_stage_env_steps=24*2048*128,native_execution=False,
        source_selection_gate='Current body search and cold128 must finish; if a better compatible source or ready formal candidate exists, revise or abandon this preparation before any launch.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    result_path.write_text(json.dumps(save,indent=2)+'\n')
    print(json.dumps(dict(actual_witness_rows=1600,physical_mean_max_error_rad=result['actual_witness_physical_mean_max_error_rad'],
        physical_leg_std_rad=float(physical_std[0]),physical_arm_std_rad=float(physical_std[-1]),native_execution=False)),flush=True)


if __name__=='__main__':main()
