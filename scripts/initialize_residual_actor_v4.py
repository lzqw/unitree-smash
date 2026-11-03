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

RESIDUAL_EXPORT_SCHEMA = 'g07-shared-target-residual-ppo-v1'


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
        # An explicit expanded-authority branch may reuse the learned public
        # actor while changing only the residual-to-joint conversion.  The
        # caller records this in the new contract; all other compatibility
        # checks remain strict.
        if key == 'joint_scale_rad' and target_contract.get('allow_joint_scale_transfer'):
            continue
        if key not in source_contract or source_contract[key] != target_contract.get(key):
            raise ValueError('Actor transfer changes a compatible-task constraint: ' + key)
    version=target_contract.get('reward_version')
    # The success-priority branch keeps the measured-string alignment
    # contract while replacing the legacy 8 m/s term with the authorized
    # saturated 5 m/s term.  It is a compatible actor-transfer target.
    compatible_alignment_versions = [
        ALIGNMENT_REWARD_VERSION,
        'native-full-chain-success-priority-speed5-v1',
    ]
    if (version not in [REWARD_VERSION, *compatible_alignment_versions]
            or placement_settings(target_contract.get('joint_placement_reward')) is None):
        raise ValueError('Use the explicit opt-in joint-placement reward contract')
    alignment=alignment_settings(target_contract.get('live_alignment_reward'))
    if version in compatible_alignment_versions:
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
    transferred = {key: source_model[key].clone() for key in keys}
    # Residual commands are converted to physical joint targets using the
    # configured scale.  When an expanded-authority branch reuses an actor,
    # preserve its initial physical behavior by inversely scaling the final
    # 18 output rows before PPO explores the larger action space.
    source_scale = source_contract.get('joint_scale_rad')
    target_scale = target_contract.get('joint_scale_rad')
    if target_contract.get('allow_joint_scale_transfer') and source_scale != target_scale:
        source_scale = torch.as_tensor(source_scale, dtype=transferred['log_std'].dtype)
        target_scale = torch.as_tensor(target_scale, dtype=transferred['log_std'].dtype)
        if source_scale.shape != (18,) or target_scale.shape != (18,) or (source_scale <= 0).any() or (target_scale <= 0).any():
            raise ValueError('Invalid joint-scale transfer contract')
        output_weights = [name for name, value in transferred.items()
                          if name.startswith('actor.') and name.endswith('.weight')
                          and torch.is_tensor(value) and value.shape[0] == 18]
        output_biases = [name for name, value in transferred.items()
                         if name.startswith('actor.') and name.endswith('.bias')
                         and torch.is_tensor(value) and value.shape == (18,)]
        if len(output_weights) != 1 or len(output_biases) != 1:
            raise ValueError('Could not identify the final 18D residual output layer')
        ratio = (source_scale / target_scale).to(transferred[output_weights[0]])
        transferred[output_weights[0]] = transferred[output_weights[0]] * ratio[:, None]
        transferred[output_biases[0]] = transferred[output_biases[0]] * ratio
    return transferred


def select_exported_actor_state(state, target_contract, current_model):
    """Transfer a measured residual policy export into a fresh learner.

    The export contains the deterministic actor and its public observation
    normalizer, but intentionally omits the critic, optimizer and exploration
    scale.  The fresh learner keeps its declared initial log_std.
    """
    if state.get('schema') != RESIDUAL_EXPORT_SCHEMA or not state.get('native_execution'):
        raise ValueError('Actor initialization requires a native residual policy export')
    metadata = state.get('residual_training_metadata', {})
    if metadata.get('base_policy_sha256') != target_contract.get('base_policy_sha256'):
        raise ValueError('Residual export base policy differs from target contract')
    if state.get('protocol_sha256') != target_contract.get('protocol_sha256'):
        raise ValueError('Residual export protocol differs from target contract')
    if state.get('residual_base_mode') != 'shared_target_bias':
        raise ValueError('Residual export base mode is incompatible')
    source_scale = state.get('residual_joint_scale_rad')
    target_scale = target_contract.get('joint_scale_rad')
    if source_scale != target_scale:
        raise ValueError('Residual export joint scale differs from target contract')
    module = torch.jit.load(io.BytesIO(state['residual_actor_jit']), map_location='cpu').eval()
    source = module.state_dict()
    if 'normalizer._mean' in source:
        # Phase-gated exports use named submodules (normalizer/base/head),
        # unlike the historical Sequential(normalizer, actor) export.
        mapping = {
            'normalizer._mean': 'actor_obs_normalizer._mean',
            'normalizer._var': 'actor_obs_normalizer._var',
            'normalizer._std': 'actor_obs_normalizer._std',
            'normalizer.count': 'actor_obs_normalizer.count',
        }
        for index in (0, 2, 4, 6):
            mapping[f'base.{index}.weight'] = f'actor.{index}.weight'
            mapping[f'base.{index}.bias'] = f'actor.{index}.bias'
        # Preserve an already learned recovery head when a new pre-hit head
        # is added to the target policy.
        for source_prefix,target_prefix in [('head.','recovery_head.'),
                                             ('recovery.','recovery_head.'),
                                             ('approach.','approach_head.')]:
            for key in source:
                if key.startswith(source_prefix):
                    mapping[key]=target_prefix+key[len(source_prefix):]
    else:
        mapping = {
            '0._mean': 'actor_obs_normalizer._mean',
            '0._var': 'actor_obs_normalizer._var',
            '0._std': 'actor_obs_normalizer._std',
            '0.count': 'actor_obs_normalizer.count',
        }
        for index in (0, 2, 4, 6):
            mapping[f'1.{index}.weight'] = f'actor.{index}.weight'
            mapping[f'1.{index}.bias'] = f'actor.{index}.bias'
    transferred = {}
    for source_key, target_key in mapping.items():
        if source_key not in source or target_key not in current_model:
            raise ValueError('Residual export actor state is incomplete: ' + source_key)
        value, target = source[source_key], current_model[target_key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Residual export actor state mismatch: ' + source_key)
        if not torch.isfinite(value).all():
            raise ValueError('Residual export actor state is nonfinite: ' + source_key)
        transferred[target_key] = value.clone()
    return transferred


def initialize_actor(runner, path):
    if (runner.current_learning_iteration != 0 or runner.tot_timesteps != 0 or runner.tot_time != 0
            or runner.alg.optimizer.state or getattr(runner, '_learning_now', False)):
        raise ValueError('Actor initialization requires a fresh learner, optimizer and counters')
    data = Path(path).read_bytes()
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    model = runner.alg.policy.state_dict()
    if state.get('schema') == RESIDUAL_EXPORT_SCHEMA:
        if getattr(runner.alg.policy, 'freeze_verified_policy', False):
            metadata=state.get('residual_training_metadata', {})
            contract=runner.original_training_config['residual_contract']
            for key in ('recovery_assistance','recovery_assistance_execution',
                        'landing_assistance','support_assistance'):
                if metadata.get(key) != contract.get(key):
                    raise ValueError('Frozen v2 initialization changes execution: '+key)
            module=torch.jit.load(io.BytesIO(state['residual_actor_jit']),map_location='cpu')
            if not any(k.startswith('head.') for k in module.state_dict()):
                raise ValueError('Frozen v2 initialization requires the verified recovery head')
        transferred = select_exported_actor_state(state,
            runner.original_training_config['residual_contract'], model)
        mode = 'fresh_learner_with_transferred_exported_residual_actor'
    else:
        transferred = select_actor_state(state, interface(runner.env.unwrapped),
            runner.original_training_config['residual_contract'], model)
        mode = 'fresh_learner_with_transferred_public_actor'
    model.update(transferred)
    runner.alg.policy.load_state_dict(model, strict=True)
    # Reset through the existing real ground path. No source environment
    # buffers or random-number state are read back into the new simulation.
    runner.env.reset()
    if state.get('schema') == RESIDUAL_EXPORT_SCHEMA:
        metadata = state.get('residual_training_metadata', {})
        source_iteration = metadata.get('completed_updates')
        source_num_envs = metadata.get('num_envs')
        source_steps = metadata.get('actual_control_env_steps')
    else:
        source_iteration = state['iter']
        source_num_envs = state['native']['environment']['num_envs']
        source_steps = state['native']['total_timesteps']
    return dict(mode=mode,
        num_envs=runner.env.num_envs, initial_iteration=0, current_job_steps_start_at_zero=True,
        source_checkpoint_sha256=hashlib.sha256(data).hexdigest(), source_iteration=source_iteration,
        source_num_envs=source_num_envs,
        source_learner_control_env_steps=source_steps,
        public_actor_and_actor_normalizer_transferred=True, exploration_log_std_transferred=True,
        critic_transferred=False, optimizer_transferred=False, source_rng_restored=False,
        old_physical_state_restored=False, old_environment_counters_restored=False,
        previous_learning_counters_restored=False, performance_inherited=False)
