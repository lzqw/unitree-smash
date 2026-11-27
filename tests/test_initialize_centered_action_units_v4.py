"""CPU action-unit mapping on actual measured weights/observations, not performance."""
import contextlib
import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.initialize_centered_action_units_v4 import apply_action_unit_mapped_actor
from scripts.initialize_centered_policy_v4 import apply_centered_public_actor

ROOT = Path(__file__).resolve().parents[1]
# Resolve the archived recordings in the primary checkout from any worktree.
MAIN = Path(subprocess.check_output(
    ['git', '-C', str(ROOT), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
    text=True).strip()).parent
REPORTS = MAIN/'reports/G07_full_racket_v4'


@pytest.fixture(scope='module')
def material():
    source = REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
    native = torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    public = torch.load(source/'policy.pt',map_location='cpu',weights_only=False)
    base = torch.load(REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',map_location='cpu',weights_only=False)
    config = json.loads((ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json').read_text())
    contract = copy.deepcopy(native['infos']['residual_contract'])
    contract.update(joint_scale_rad=config['joint_scale_rad'],joint_placement_reward=config['joint_placement_reward'],seed=749301)
    snapshot = REPORTS/'video_snapshot_centered_resume_first_complete_20261026'
    samples = []
    for case in json.loads((snapshot/'cases.json').read_text())['cases']:
        path = snapshot/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        with np.load(path) as arrays:samples.append(torch.from_numpy(unpack_public(arrays['packed_public'])))
    sample = torch.cat(samples);assert sample.shape == (1890,794)
    evaluation = REPORTS/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
    with np.load(evaluation/'public_and_actions_first2.npz') as arrays:
        evaluated_sample = torch.from_numpy(arrays['observations'][arrays['valid_first_episode'].astype(bool)].copy())
    return native,public,base,contract,config,sample,evaluated_sample


def fresh(config):
    options = dict(config['runner']['policy']);options.pop('class_name')
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),config['runner']['obs_groups'],18,**options)


def test_actual_recorded_physical_means_and_fresh_learning_state(material):
    native,public,base,contract,config,sample,evaluated = material
    policy = fresh(config);optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
    before = {name:value.clone() for name,value in policy.state_dict().items()}
    ids = [id(p) for p in policy.parameters()];rng = torch.get_rng_state().clone()
    witness = torch.cat([sample,evaluated])
    result = apply_action_unit_mapped_actor(policy,public,native['native']['interface'],contract,base,witness)
    assert not optimizer.state and torch.equal(rng,torch.get_rng_state())
    assert ids == [id(p) for p in policy.parameters()] and policy.training and policy.actor_obs_normalizer.training
    for name,value in before.items():
        if name.startswith(('critic.','critic_obs_normalizer.')):assert torch.equal(value,policy.state_dict()[name])
    assert int(policy.actor_obs_normalizer.count) == 10485760 and int(policy.critic_obs_normalizer.count) == 0
    assert torch.allclose(policy.log_std.exp(),torch.tensor([.05]*12+[.1]*6),atol=1e-7,rtol=0)
    source_actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
    policy.eval()
    with torch.inference_mode():
        expected = source_actor(witness).clamp(-1,1)*torch.tensor([.03]*12+[.15]*6)
        actual = policy.act_inference(TensorDict({'policy':witness},batch_size=[len(witness)])).clamp(-1,1)*torch.tensor([.06]*12+[.15]*6)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-5)
    assert result['actual_witness_physical_mean_max_error_rad'] <= 2e-6
    assert not result['critic_transferred'] and not result['optimizer_transferred'] and not result['exploration_log_std_transferred']
    assert result['output_action_units_explicitly_changed'] and not result['performance_inherited']


@pytest.mark.parametrize('bad',['different_scale','different_arm','empty','nan','width','nonfresh_std','base'])
def test_invalid_mapping_rejected_before_policy_mutation(material,bad):
    native,public,base,contract,config,sample,_ = material
    contract,base = copy.deepcopy(contract),copy.deepcopy(base)
    policy = fresh(config);witness = sample
    if bad == 'different_scale':contract['joint_scale_rad'][0] = .08
    elif bad == 'different_arm':contract['joint_scale_rad'][-1] = .2
    elif bad == 'empty':witness = torch.empty(0,794)
    elif bad == 'nan':witness = sample.clone();witness[0,0] = float('nan')
    elif bad == 'width':witness = sample[:,:793]
    elif bad == 'nonfresh_std':
        with torch.no_grad():policy.log_std[0] = torch.tensor(.2).log()
    elif bad == 'base':base['parameters'][0] += .01
    before = {name:value.clone() for name,value in policy.state_dict().items()}
    with pytest.raises(ValueError):
        apply_action_unit_mapped_actor(policy,public,native['native']['interface'],contract,base,witness)
    assert all(torch.equal(value,policy.state_dict()[name]) for name,value in before.items())


def test_clipped_source_counterexample_rejected_before_mutation(material):
    native,public,base,contract,config,sample,_ = material
    modified = copy.deepcopy(public)
    actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
    state = actor.state_dict()
    state['1.6.weight'][0].zero_();state['1.6.bias'][0] = 2.
    actor.load_state_dict(state)
    buffer = io.BytesIO();torch.jit.save(actor,buffer)
    modified['residual_actor_jit'] = buffer.getvalue()
    modified['residual_training_metadata']['actor_jit_sha256'] = hashlib.sha256(buffer.getvalue()).hexdigest()
    policy = fresh(config);before = {name:value.clone() for name,value in policy.state_dict().items()}
    with pytest.raises(ValueError,match='clip'):
        apply_action_unit_mapped_actor(policy,modified,native['native']['interface'],contract,base,sample)
    assert all(torch.equal(value,policy.state_dict()[name]) for name,value in before.items())


def test_original_strict_transfer_still_refuses_changed_action_units(material):
    native,public,base,contract,config,_,_ = material
    with pytest.raises(ValueError,match='units'):
        apply_centered_public_actor(fresh(config),public,native['native']['interface'],contract,base)
