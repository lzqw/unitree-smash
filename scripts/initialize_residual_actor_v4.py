"""Explicit actor transfer into a fresh learner with a changed TRAIN reward.

This never calls NativeRunner.load or restores old physics, critic, Adam, RNG,
or experiment counters. Strict canonical resume and resizing remain separate.
"""
import hashlib
import io
from pathlib import Path

import torch

from quadsmash.control.measured_joint_placement_reward_v4 import REWARD_VERSION, placement_settings
from quadsmash.control.measured_string_alignment_reward_v4 import REWARD_VERSION as ALIGNMENT_REWARD_VERSION,alignment_settings
from quadsmash.rl.native_checkpoint import SCHEMA, interface, training_signature


def actor_key(name):
    return name == 'log_std' or name.startswith(('actor.', 'actor_obs_normalizer.'))


def select_actor_state(state, current_interface, target_contract, current_model):
    native = state.get('native', {})
    if native.get('schema') != SCHEMA or not native.get('canonical_restart'):
        raise ValueError('Actor initialization requires an actual canonical native source')
    if native.get('interface') != current_interface:
        raise ValueError('Actor transfer must preserve the observation/action/asset interface')
    if current_interface.get('action_family') != 'bounded-residual-joint-targets-full-racket-v4':
        raise ValueError('This transfer is specific to the full-racket residual policy')
    configuration = native.get('training_config', {})
    source_contract = state.get('infos', {}).get('residual_contract')
    if (not isinstance(source_contract, dict) or configuration.get('residual_contract') != source_contract
            or native.get('training_signature') != training_signature(configuration)):
        raise ValueError('Source training configuration/contract signature is inconsistent')
    for key in ['base_policy_sha256', 'protocol_sha256', 'joint_scale_rad', 'event_weights']:
        if key not in source_contract or source_contract[key] != target_contract.get(key):
            raise ValueError('Actor transfer changes a compatible-task constraint: ' + key)
    version=target_contract.get('reward_version')
    if (version not in [REWARD_VERSION,ALIGNMENT_REWARD_VERSION]
            or placement_settings(target_contract.get('joint_placement_reward')) is None):
        raise ValueError('Use the explicit opt-in joint-placement reward contract')
    alignment=alignment_settings(target_contract.get('live_alignment_reward'))
    if version==ALIGNMENT_REWARD_VERSION:
        if (alignment is None or target_contract.get('target_event_requires_actual_attack') is not True
                or target_contract.get('fixed_upper_uv_position_shaping') is not False):
            raise ValueError('Use the complete opt-in measured strings/attack contract')
    elif alignment is not None:
        raise ValueError('Measured strings/attack settings require their own reward version')
    environment = native.get('environment', {})
    clock = environment.get('tensors', {}).get('episode_length_buf')
    count = environment.get('num_envs', 0)
    if (not isinstance(count, int) or count < 128 or count % 128
            or not torch.is_tensor(clock) or clock.shape != (count,) or clock.any()):
        raise ValueError('Source must be the declared canonical ground boundary')
    source_model = state.get('model_state_dict', {})
    for value in source_model.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite source model or normalizer')
    keys = {key for key in current_model if actor_key(key)}
    if ('log_std' not in keys or not any(k.startswith('actor.') for k in keys)
            or not any(k.startswith('actor_obs_normalizer.') for k in keys)
            or keys != {key for key in source_model if actor_key(key)}):
        raise ValueError('Public actor/normalizer/exploration state is incomplete or incompatible')
    for key in keys:
        source, target = source_model[key], current_model[key]
        if (not torch.is_tensor(source) or not torch.is_tensor(target)
                or source.shape != target.shape or source.dtype != target.dtype):
            raise ValueError('Actor architecture mismatch: ' + key)
    std = source_model['log_std'].exp()
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Source exploration scale cannot be represented')
    return {key: source_model[key] for key in keys}


def initialize_actor(runner, path):
    if (runner.current_learning_iteration != 0 or runner.tot_timesteps != 0 or runner.tot_time != 0
            or runner.alg.optimizer.state or getattr(runner, '_learning_now', False)):
        raise ValueError('Actor initialization requires a fresh learner, optimizer and counters')
    data = Path(path).read_bytes()
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    model = runner.alg.policy.state_dict()
    transferred = select_actor_state(state, interface(runner.env.unwrapped),
        runner.original_training_config['residual_contract'], model)
    model.update(transferred)
    runner.alg.policy.load_state_dict(model, strict=True)
    # Reset through the existing real ground path. No source environment
    # buffers or random-number state are read back into the new simulation.
    runner.env.reset()
    return dict(mode='fresh_learner_with_transferred_public_actor',
        num_envs=runner.env.num_envs, initial_iteration=0, current_job_steps_start_at_zero=True,
        source_checkpoint_sha256=hashlib.sha256(data).hexdigest(), source_iteration=state['iter'],
        source_num_envs=state['native']['environment']['num_envs'],
        source_learner_control_env_steps=state['native']['total_timesteps'],
        public_actor_and_actor_normalizer_transferred=True, exploration_log_std_transferred=True,
        critic_transferred=False, optimizer_transferred=False, source_rng_restored=False,
        old_physical_state_restored=False, old_environment_counters_restored=False,
        previous_learning_counters_restored=False, performance_inherited=False)
