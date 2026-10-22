import copy
import math
import random
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from quadsmash.rl.action_diagnostics import action_summary, policy_std, reset_initial_std, post_update_diagnostics


def make_policy(representation='log'):
    obs = TensorDict({'policy': torch.arange(28, dtype=torch.float32).reshape(4, 7) / 30.}, batch_size=[4])
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 3,
        actor_hidden_dims=[8], critic_hidden_dims=[8], actor_obs_normalization=True,
        critic_obs_normalization=True, noise_std_type=representation, init_noise_std=.6)
    return policy, obs


@pytest.mark.parametrize('representation', ['log', 'scalar'])
def test_loaded_std_reset_changes_only_noise_and_remains_trainable(representation):
    policy, obs = make_policy(representation)
    saved = copy.deepcopy(policy.state_dict()); key = 'log_std' if representation == 'log' else 'std'
    saved[key][:] = math.log(1.49) if representation == 'log' else 1.49
    policy.load_state_dict(saved)
    audit = reset_initial_std(policy, .15)
    assert audit['std_remains_learnable']
    assert torch.allclose(policy_std(policy), torch.full((3,), .15), atol=1e-7)
    for k, v in saved.items():
        if k != key: assert torch.equal(v, policy.state_dict()[k])
    policy.update_distribution(policy.actor_obs_normalizer(policy.get_actor_obs(obs)))
    assert torch.allclose(policy.action_std, torch.full((4, 3), .15), atol=1e-7)
    for bad in (0., -1., float('nan')):
        with pytest.raises(ValueError): reset_initial_std(policy, bad)


def test_action_statistics_match_known_boundaries_and_analytic_gaussian():
    mean = torch.zeros(2, 3); std = torch.ones(3)
    actions = torch.tensor([[-2., -1., 0.], [1., 2., 3.]])
    result = action_summary(mean, std, actions)
    assert result['actual_clipped_fraction'] == .5
    assert result['actual_clipped_fraction_per_joint'] == [.5, .5, .5]
    assert result['mean_outside_bounds_fraction'] == 0.
    assert result['expected_clipping_fraction'] == pytest.approx(math.erfc(1 / math.sqrt(2)), abs=1e-6)
    assert max(result['clipped_mean_vs_expected_action_gap_per_joint']) < 1e-6
    saturated = action_summary(torch.tensor([[2., -2.]]), torch.tensor([.15, .15]))
    assert saturated['mean_outside_bounds_fraction'] == 1.
    assert saturated['expected_clipping_fraction'] > .999


def test_telemetry_does_not_change_rng_distribution_normalizers_parameters_or_gradients():
    policy, obs = make_policy(); policy.train(); policy.update_normalization(obs)
    actions = policy.act(obs).detach(); mu = policy.action_mean.detach(); std = policy.action_std.detach()
    logprob = policy.get_actions_log_prob(actions).detach()
    rng = torch.get_rng_state().clone(); python_rng = random.getstate(); numpy_rng = np.random.get_state()
    state = copy.deepcopy(policy.state_dict()); distribution = policy.distribution
    flags = [module.training for module in policy.modules()]
    result = post_update_diagnostics(policy, obs, mu, std, actions, logprob, .2)
    assert result['post_update_kl'] == pytest.approx(0., abs=1e-6)
    assert result['post_update_clip_fraction'] == 0.
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
    assert distribution is policy.distribution and flags == [module.training for module in policy.modules()]
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in state.items())
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_cpu_ppo_unit_update_is_identical_with_readonly_telemetry():
    # Synthetic storage fixture only: no environment, simulator or task training.
    from rsl_rl.algorithms import PPO
    from quadsmash.rl.native_ppo import FinalObservationPPO
    policy, obs = make_policy()
    alg = FinalObservationPPO(policy, device='cpu', num_learning_epochs=1, num_mini_batches=1)
    alg.init_storage('rl', 4, 2, obs, [3])
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
        alg.compute_returns(obs)
    plain = copy.deepcopy(alg)
    before = torch.get_rng_state().clone()
    losses = alg.update(); after = torch.get_rng_state().clone()
    torch.set_rng_state(before); reference = PPO.update(plain)
    assert torch.equal(after, torch.get_rng_state())
    assert losses == reference
    assert all(torch.equal(value, plain.policy.state_dict()[key]) for key, value in alg.policy.state_dict().items())
    assert alg.last_update_stats['behavior_actions']['samples'] == 8
    assert len(alg.last_update_stats['post_update_actions']['std_per_joint']) == 3
