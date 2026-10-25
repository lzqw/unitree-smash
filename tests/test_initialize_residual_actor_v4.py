import copy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from quadsmash.control.full_racket_residual_reward_v4 import EVENT_WEIGHTS
from quadsmash.control.measured_joint_placement_reward_v4 import REWARD_VERSION
from quadsmash.rl.native_checkpoint import SCHEMA, training_signature
import scripts.initialize_residual_actor_v4 as transfer


def example():
    obs = TensorDict({'policy': torch.randn(4, 794)}, batch_size=[4])
    make = lambda: ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 18,
        actor_hidden_dims=[16], critic_hidden_dims=[16], actor_obs_normalization=True,
        critic_obs_normalization=True, noise_std_type='log', init_noise_std=.1)
    source, target = make(), make()
    source.update_normalization(obs)
    contract = dict(base_policy_sha256='CPU-base', protocol_sha256='CPU-protocol',
        joint_scale_rad=[.12] * 12 + [.15] * 6, event_weights=EVENT_WEIGHTS,
        reward_version='native-full-chain-residual-v1', seed=748501)
    target_contract = dict(copy.deepcopy(contract), reward_version=REWARD_VERSION,
        joint_placement_reward=dict(weight=20., distance_scale_m=.25), seed=748503)
    interface = dict(schema=SCHEMA, action_family='bounded-residual-joint-targets-full-racket-v4',
        observation_size=794, asset_sha256='CPU-only-no-native-performance')
    cfg = dict(residual_contract=contract)
    state = dict(iter=32, infos=dict(residual_contract=copy.deepcopy(contract)),
        model_state_dict=source.state_dict(), optimizer_state_dict={'state': {}},
        native=dict(schema=SCHEMA, canonical_restart=True, interface=interface,
            training_config=cfg, training_signature=training_signature(cfg), total_timesteps=1703936,
            environment=dict(num_envs=128, tensors={'episode_length_buf': torch.zeros(128, dtype=torch.long)})))
    return obs, source, target, state, target_contract, interface


def test_only_public_actor_normalizer_and_exploration_are_transferred(tmp_path, monkeypatch):
    obs, source, target, state, contract, interface = example()
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-4)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    parameter_ids = [id(p) for p in target.parameters()]
    resets = []
    env = SimpleNamespace(num_envs=2048, unwrapped=object(), reset=lambda: resets.append('fresh_ground'))
    runner = SimpleNamespace(alg=SimpleNamespace(policy=target, optimizer=optimizer), env=env,
        current_learning_iteration=0, tot_timesteps=0, tot_time=0.,
        original_training_config={'residual_contract': contract})
    monkeypatch.setattr(transfer, 'interface', lambda env: interface)
    path = tmp_path / 'canonical.pt'
    torch.save(state, path)
    rng = torch.get_rng_state().clone()
    result = transfer.initialize_actor(runner, path)
    assert resets == ['fresh_ground'] and torch.equal(torch.get_rng_state(), rng)
    assert [id(p) for p in target.parameters()] == parameter_ids
    assert not optimizer.state and runner.current_learning_iteration == runner.tot_timesteps == runner.tot_time == 0
    for key, value in target.state_dict().items():
        expected = state['model_state_dict'][key] if transfer.actor_key(key) else before[key]
        assert torch.equal(value, expected), key
    source.eval(); target.eval()
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs), atol=1e-7, rtol=1e-7)
    assert result['source_iteration'] == 32 and not result['critic_transferred']
    assert not result['optimizer_transferred'] and not result['old_physical_state_restored']


@pytest.mark.parametrize('change', ['canonical', 'asset', 'protocol', 'baseline', 'scale',
    'clock', 'signature', 'missing_actor', 'nan', 'architecture', 'exploration', 'reward'])
def test_incompatible_or_unsafe_source_is_rejected(change):
    _, _, target, state, contract, interface = example()
    if change == 'canonical': state['native']['canonical_restart'] = False
    elif change == 'asset': interface = dict(interface, asset_sha256='different')
    elif change == 'protocol': contract['protocol_sha256'] = 'different'
    elif change == 'baseline': contract['base_policy_sha256'] = 'different'
    elif change == 'scale': contract['joint_scale_rad'][0] = .25
    elif change == 'clock': state['native']['environment']['tensors']['episode_length_buf'][0] = 1
    elif change == 'signature': state['native']['training_signature'] = 'different'
    elif change == 'missing_actor': del state['model_state_dict']['log_std']
    elif change == 'nan': state['model_state_dict']['log_std'][0] = float('nan')
    elif change == 'architecture': state['model_state_dict']['log_std'] = torch.zeros(17)
    elif change == 'exploration': state['model_state_dict']['log_std'].fill_(1000.)
    elif change == 'reward': contract['reward_version'] = 'native-full-chain-residual-v1'
    with pytest.raises(ValueError):
        transfer.select_actor_state(state, interface, contract, target.state_dict())


def test_existing_learner_counters_cannot_be_discarded_by_actor_initialization():
    runner = SimpleNamespace(current_learning_iteration=1, tot_timesteps=0, tot_time=0.,
        alg=SimpleNamespace(optimizer=SimpleNamespace(state={})))
    with pytest.raises(ValueError, match='fresh learner'):
        transfer.initialize_actor(runner, 'must-not-be-read.pt')


def alignment_contract(contract):
    from quadsmash.control.measured_string_alignment_reward_v4 import REWARD_VERSION as VERSION
    return dict(copy.deepcopy(contract), reward_version=VERSION,
        live_alignment_reward=dict(contact_potential_weight=5., normal_distance_scale_m=.2,
            attack_progress_weight=15., frame_penalty=5., discount=.997),
        target_event_requires_actual_attack=True, fixed_upper_uv_position_shaping=False)


def test_v3_explicit_transfer_still_selects_only_public_actor_not_old_critic():
    _, _, target, state, old_contract, interface = example()
    contract = alignment_contract(old_contract)
    before = {key:value.clone() for key,value in target.state_dict().items()}
    selected = transfer.select_actor_state(state, interface, contract, target.state_dict())
    assert selected and all(transfer.actor_key(key) for key in selected)
    assert not any('critic' in key for key in selected)
    merged = target.state_dict(); merged.update(selected); target.load_state_dict(merged)
    for key,value in target.state_dict().items():
        assert torch.equal(value, state['model_state_dict'][key] if transfer.actor_key(key) else before[key])


@pytest.mark.parametrize('change', ['missing_settings', 'missing_attack_gate', 'old_position_enabled', 'wrong_version'])
def test_v3_cannot_be_enabled_with_an_incomplete_or_v2_labeled_contract(change):
    _, _, target, state, old_contract, interface = example()
    contract = alignment_contract(old_contract)
    if change == 'missing_settings': del contract['live_alignment_reward']
    elif change == 'missing_attack_gate': del contract['target_event_requires_actual_attack']
    elif change == 'old_position_enabled': contract['fixed_upper_uv_position_shaping'] = True
    elif change == 'wrong_version': contract['reward_version'] = REWARD_VERSION
    with pytest.raises(ValueError):
        transfer.select_actor_state(state, interface, contract, target.state_dict())


def test_v3_reward_and_actor_initialization_do_not_authorize_cross_reward_resume():
    _, _, _, _, v2, _ = example()
    v3 = alignment_contract(v2)
    assert training_signature({'residual_contract':v2}) != training_signature({'residual_contract':v3})
    changed = copy.deepcopy(v3); changed['live_alignment_reward']['frame_penalty'] = 6.
    assert training_signature({'residual_contract':changed}) != training_signature({'residual_contract':v3})
