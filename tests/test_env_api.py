import numpy as np

from quadsmash.tasks import QuadSmashEnv, QuadSmashEnvConfig
from quadsmash.tasks.observations import observation_size


def test_parallel_reset_step_contract():
    env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=32, seed=3))
    obs, info = env.reset(seed=3)
    assert obs.shape == (32, observation_size(18))
    assert np.isfinite(obs).all()
    next_obs, reward, terminated, truncated, infos = env.step(np.zeros((32, 18), dtype=np.float32))
    assert next_obs.shape == obs.shape
    assert reward.shape == (32,)
    assert terminated.shape == truncated.shape == (32,)
    assert np.isfinite(next_obs).all() and np.isfinite(reward).all()
    assert len(infos["episode"]) == 32


def test_seed_reproduces_reset_state():
    a = QuadSmashEnv(QuadSmashEnvConfig(num_envs=4, seed=9)); obs_a, _ = a.reset(seed=11)
    b = QuadSmashEnv(QuadSmashEnvConfig(num_envs=4, seed=9)); obs_b, _ = b.reset(seed=11)
    assert np.array_equal(obs_a, obs_b)

