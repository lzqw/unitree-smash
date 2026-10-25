"""Explicitly carry learner state into a larger, freshly grounded TRAIN batch.

This is a fork at a canonical episode boundary. It never restores old physical
buffers into a different environment layout; strict NativeRunner.load is unchanged.
"""
import torch
from quadsmash.rl.native_checkpoint import SCHEMA, interface, restore_rng, training_signature


def validate_ground_resize(state, current_interface, configuration, num_envs):
    native = state.get('native', {})
    if native.get('schema') != SCHEMA or not native.get('canonical_restart'):
        raise ValueError('Ground resizing requires a canonical native checkpoint')
    if native.get('interface') != current_interface:
        raise ValueError('Ground resizing must preserve the observation/action/asset interface')
    if current_interface.get('action_family') != 'bounded-residual-joint-targets-full-racket-v4':
        raise ValueError('This ground fork is specific to full-racket residual PPO')
    if native.get('training_signature') != training_signature(configuration):
        raise ValueError('Ground resizing cannot change PPO, seed, baseline, scales or rewards')
    previous_count = native['environment']['num_envs']
    if previous_count < 1 or num_envs <= previous_count or num_envs % previous_count:
        raise ValueError('Require a larger positive integer multiple of the original environment count')
    if native['environment']['tensors']['episode_length_buf'].any():
        raise ValueError('The source must be at the declared ground episode boundary')
    for value in state['model_state_dict'].values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite source model/normalizer')
    for value in state['optimizer_state_dict']['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in value.values()):
            raise ValueError('Nonfinite source optimizer')
    return previous_count


def load_resized_ground(runner, path):
    state = torch.load(path, map_location=runner.device, weights_only=False)
    previous_count = validate_ground_resize(state, interface(runner.env.unwrapped),
        runner.original_training_config, runner.env.num_envs)
    native = state['native']
    runner.alg.policy.load_state_dict(state['model_state_dict'])
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    runner.alg.learning_rate = native['learning_rate']
    runner.alg.schedule = native['scheduler']
    runner.current_learning_iteration = state['iter']
    runner.tot_timesteps = native['total_timesteps']
    runner.tot_time = native['total_training_time']
    # Invoke the existing real ground reset, including per-lane controller,
    # history and action-limiter reset. Old root/joint/origin buffers are unused.
    runner.env.reset()
    restore_rng(native['rng'])
    return dict(mode='canonical_learner_fork_on_fresh_ground', previous_num_envs=previous_count,
        num_envs=runner.env.num_envs, initial_iteration=state['iter'],
        previous_learner_control_env_steps=native['total_timesteps'],
        policy_optimizer_normalizers_rng_preserved=True, old_physical_state_restored=False,
        old_environment_counters_restored=False, current_job_steps_start_at_zero=True)
