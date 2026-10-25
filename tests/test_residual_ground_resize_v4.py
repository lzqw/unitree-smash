import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quadsmash.rl.native_checkpoint import SCHEMA, training_signature
import scripts.residual_ground_resize_v4 as resize


def checkpoint_fixture():
    torch.manual_seed(17)
    policy = torch.nn.Linear(2, 2)
    policy.register_buffer('observation_mean', torch.tensor([.2, -.7]))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    x = torch.tensor([[1., 2.], [-.4, 1.2]])
    for _ in range(3):
        optimizer.zero_grad()
        policy(x).square().mean().backward()
        optimizer.step()
    configuration = {'seed': 748501, 'residual': 'unchanged'}
    interface = {'action_family': 'bounded-residual-joint-targets-full-racket-v4', 'asset_sha256': 'same-asset'}
    state = dict(iter=8, model_state_dict=copy.deepcopy(policy.state_dict()),
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        native=dict(schema=SCHEMA, canonical_restart=True, interface=interface,
            training_signature=training_signature(configuration),
            environment=dict(num_envs=128, tensors={'episode_length_buf': torch.zeros(128)},
                             root=torch.full((128, 13), 123.)),
            learning_rate=1e-4, scheduler='fixed', total_timesteps=131072, total_training_time=1407.,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                     torch_cpu=torch.get_rng_state(), torch_cuda=[])))
    return state, configuration, interface, policy, optimizer, x


def test_learner_fork_preserves_adam_normalizers_rng_and_uses_fresh_ground(tmp_path, monkeypatch):
    state, configuration, interface, reference, reference_optimizer, x = checkpoint_fixture()
    path = tmp_path / 'checkpoint.pt'
    torch.save(state, path)

    class GroundBatch:
        num_envs = 512
        reset_calls = 0
        root = torch.full((512, 13), 7.)

        @property
        def unwrapped(self):
            return self

        def reset(self):
            self.reset_calls += 1
            torch.rand(19)

    env = GroundBatch()
    policy = torch.nn.Linear(2, 2)
    policy.register_buffer('observation_mean', torch.zeros(2))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    runner = SimpleNamespace(device='cpu', env=env, original_training_config=configuration,
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=.01, schedule='wrong'))
    monkeypatch.setattr(resize, 'interface', lambda _: interface)
    info = resize.load_resized_ground(runner, path)
    assert env.reset_calls == 1 and env.root.eq(7.).all()
    assert info['previous_num_envs'] == 128 and info['num_envs'] == 512
    assert not info['old_physical_state_restored'] and not info['old_environment_counters_restored']
    assert runner.current_learning_iteration == 8 and runner.tot_timesteps == 131072
    assert runner.alg.learning_rate == 1e-4 and runner.alg.schedule == 'fixed'
    assert torch.equal(policy.observation_mean, reference.observation_mean)
    restored_sample = torch.rand(5)
    torch.set_rng_state(state['native']['rng']['torch_cpu'])
    assert torch.equal(restored_sample, torch.rand(5))
    for model, adam in [(reference, reference_optimizer), (policy, optimizer)]:
        adam.zero_grad()
        model(x).square().mean().backward()
        adam.step()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        assert torch.equal(actual, expected), 'An optimizer restart loses the actual Adam continuation'


@pytest.mark.parametrize('change', ['diagnostic', 'mid_episode', 'interface', 'configuration', 'model_nan', 'optimizer_nan'])
def test_ground_resize_rejects_incompatible_or_nonfinite_sources(change):
    state, cfg, interface, *_ = checkpoint_fixture()
    if change == 'diagnostic':
        state['native']['canonical_restart'] = False
    elif change == 'mid_episode':
        state['native']['environment']['tensors']['episode_length_buf'][0] = 1
    elif change == 'interface':
        interface = dict(interface, asset_sha256='other-asset')
    elif change == 'configuration':
        cfg = dict(cfg, seed=0)
    elif change == 'model_nan':
        state['model_state_dict']['weight'][0, 0] = float('nan')
    else:
        next(iter(state['optimizer_state_dict']['state'].values()))['exp_avg'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        resize.validate_ground_resize(state, interface, cfg, 512)


@pytest.mark.parametrize('num_envs', [0, 64, 128, 192])
def test_ground_resize_requires_a_larger_complete_batch_multiple(num_envs):
    state, cfg, interface, *_ = checkpoint_fixture()
    with pytest.raises(ValueError):
        resize.validate_ground_resize(state, interface, cfg, num_envs)
