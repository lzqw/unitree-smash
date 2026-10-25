"""CPU transfer checks on actual published native weights and recorded inputs.

Large source artifacts are local/release inputs, not small fabricated worlds.
These tests prove model/provenance behavior only, never native performance.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.initialize_centered_policy_v4 import (
    apply_centered_public_actor,digest,select_centered_public_state,verify_measured_source)

MEASURED = Path(os.environ.get('QUADSMASH_MEASURED_ROOT',str(Path(__file__).resolve().parents[2]/'unitree_smash')))
REPORTS = MEASURED/'reports/G07_full_racket_v4'
SOURCE = REPORTS/'residual_ppo_centered_base8_n2048_probe8_02'
EVALUATION = REPORTS/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
BASE = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'


@pytest.fixture(scope='module')
def material():
    required = [SOURCE/'checkpoint.pt',SOURCE/'policy.pt',BASE,EVALUATION/'episodes.json']
    if not all(path.is_file() for path in required):
        pytest.skip('Restore actual centered8 native/release artifacts or set QUADSMASH_MEASURED_ROOT')
    state = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    checkpoint = torch.load(SOURCE/'policy.pt',map_location='cpu',weights_only=False)
    base = torch.load(BASE,map_location='cpu',weights_only=False)
    contract = copy.deepcopy(state['infos']['residual_contract'])
    # Exercise a new TRAIN reward on exactly the old public action meaning.
    contract['joint_placement_reward']['distance_scale_m'] = .75
    contract['seed'] += 100
    configuration = json.loads((MEASURED/'configs/g07_full_racket_residual_ppo_centered_v4.json').read_text())
    sample_root = REPORTS/'video_snapshot_centered_resume_first_complete_20261026'
    cases = json.loads((sample_root/'cases.json').read_text())['cases'];samples = []
    for case in cases:
        path = sample_root/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        with np.load(path) as data:samples.append(torch.from_numpy(unpack_public(data['packed_public'])))
    sample = torch.cat(samples);assert sample.shape == (1890,794)
    return state,checkpoint,base,contract,configuration,sample


def fresh_policy(configuration,actor_dims=None):
    options = dict(configuration['runner']['policy']);options.pop('class_name')
    if actor_dims is not None:options['actor_hidden_dims'] = actor_dims
    observations = TensorDict({'policy':torch.zeros(2,794)},batch_size=[2])
    with contextlib.redirect_stdout(io.StringIO()):
        return ActorCritic(observations,configuration['runner']['obs_groups'],18,**options)


def test_actual_original128_source_is_recomputed(material):
    _,_,_,contract,_,_ = material
    checked = verify_measured_source(MEASURED,SOURCE/'policy.pt',EVALUATION,contract)
    assert checked['source_policy_sha256'] == digest(SOURCE/'policy.pt')
    summary = checked['source_original_TRAIN_summary']
    assert summary['episodes'] == 128 and summary['complete'] == 0
    assert summary['airborne_contacts'] == 26 and summary['conditional_recovery_count'] == 13
    assert summary['attack_outgoing'] == 3 and summary['numerical_failures'] == 0
    assert checked['source_performance_is_not_inherited']


def test_exact_public_mean_and_fresh_nonactor_state(material):
    state,checkpoint,base,contract,configuration,sample = material
    policy = fresh_policy(configuration)
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
    parameters = [id(parameter) for parameter in policy.parameters()]
    before = {name:value.clone() for name,value in policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = apply_centered_public_actor(policy,checkpoint,state['native']['interface'],contract,base)
    assert torch.equal(torch.get_rng_state(),rng) and not optimizer.state
    assert parameters == [id(parameter) for parameter in policy.parameters()]
    assert policy.training and policy.actor_obs_normalizer.training
    assert int(policy.state_dict()['actor_obs_normalizer.count']) == 2097152
    assert int(policy.state_dict()['critic_obs_normalizer.count']) == 0
    for name,value in before.items():
        if not name.startswith(('actor.','actor_obs_normalizer.')):
            assert torch.equal(policy.state_dict()[name],value),name
    source_actor = torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']),map_location='cpu').eval()
    policy.eval()
    with torch.inference_mode():
        actual = policy.act_inference(TensorDict({'policy':sample},batch_size=[len(sample)]))
        expected = source_actor(sample)
    assert torch.equal(actual,expected)
    assert result['actor_export_max_error'] == 0.
    assert not result['critic_transferred'] and not result['exploration_log_std_transferred']
    assert not result['optimizer_transferred'] and not result['performance_inherited']


@pytest.mark.parametrize('change',['source_mode','unexecuted','base','scales','blob','interface'])
def test_incompatible_transfer_rejected_before_mutation(material,change):
    state,checkpoint,base,contract,configuration,_ = material
    checkpoint,base,contract = copy.deepcopy(checkpoint),copy.deepcopy(base),copy.deepcopy(contract)
    current_interface = copy.deepcopy(state['native']['interface'])
    if change == 'source_mode':checkpoint['residual_base_mode'] = 'analytic'
    elif change == 'unexecuted':checkpoint['native_execution'] = False
    elif change == 'base':base['parameters'][0] += .01
    elif change == 'scales':contract['joint_scale_rad'][0] += .01
    elif change == 'blob':checkpoint['residual_training_metadata']['actor_jit_sha256'] = '0'*64
    elif change == 'interface':current_interface['policy_dt'] = .02
    policy = fresh_policy(configuration);before = {name:value.clone()for name,value in policy.state_dict().items()}
    with pytest.raises(ValueError):
        apply_centered_public_actor(policy,checkpoint,current_interface,contract,base)
    assert all(torch.equal(value,policy.state_dict()[name])for name,value in before.items())


def test_wrong_actor_architecture_rejected(material):
    state,checkpoint,base,contract,configuration,_ = material
    policy = fresh_policy(configuration,[128,64])
    with pytest.raises(ValueError,match='architecture'):
        select_centered_public_state(checkpoint,state['native']['interface'],contract,base,policy.state_dict())


def test_CPU_intermediate_export_cannot_borrow_other_native_evaluation(material):
    _,_,_,contract,_,_ = material
    unmeasured = REPORTS/'centered_intermediate_selection_queue01/exports/iter16/policy.pt'
    assert unmeasured.exists()
    with pytest.raises(ValueError,match='matching original TRAIN'):
        verify_measured_source(MEASURED,unmeasured,EVALUATION,contract)


def test_changed_original_protocol_rejected(material):
    _,_,_,contract,_,_ = material
    contract = copy.deepcopy(contract);contract['protocol_sha256'] = '0'*64
    with pytest.raises(ValueError,match='matching original TRAIN'):
        verify_measured_source(MEASURED,SOURCE/'policy.pt',EVALUATION,contract)
