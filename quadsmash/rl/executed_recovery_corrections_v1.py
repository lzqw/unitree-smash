"""Learn recovery actions that actually succeeded in teacher-assisted TRAIN.

This auxiliary optimizer does not write PPO actions or log probabilities. Its
labels are effective residuals observed before the production mixer, selected
only from completed full-success episodes and verified by command replay.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .success_retention_v1 import SuccessRetention
from .action_diagnostics import post_update_diagnostics

SCHEMA = 'executed-recovery-correction-learning-v1'
DATA_SCHEMA = 'verified-executed-recovery-corrections-v1'
EXECUTION_KEYS = ('base_policy_sha256', 'protocol_sha256', 'joint_scale_rad',
                  'arm_target_rate_rad_s', 'residual_scale_limit_rad',
                  'recovery_assistance', 'recovery_assistance_execution',
                  'landing_assistance', 'support_assistance')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(configuration, contract, device):
    expected = {'manifest', 'manifest_sha256', 'learning_rate', 'steps_per_update',
                'batch_size', 'anchor_weight', 'seed'}
    if not isinstance(configuration, dict) or set(configuration) != expected:
        raise ValueError('Declare the full executed-correction configuration')
    for key in ['steps_per_update', 'batch_size', 'seed']:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError('Invalid correction sampling value: ' + key)
    for key in ['learning_rate', 'anchor_weight']:
        value = configuration[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0. or (key == 'learning_rate' and value == 0.)):
            raise ValueError('Invalid correction optimizer value: ' + key)
    path = Path(configuration['manifest'])
    if digest(path) != configuration['manifest_sha256']:
        raise ValueError('Executed-correction manifest changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != DATA_SCHEMA
            or manifest.get('target_interface') != 'raw-residual18-before-mixer'
            or manifest.get('target_kind') != 'actually_executed_teacher_mixed_residual'
            or not manifest.get('command_replay_validated')
            or not manifest.get('full_success_only')
            or not manifest.get('recovery_only_and_failed_episodes_excluded')
            or manifest.get('frozen_policy_performance')
            or not manifest.get('native_execution')
            or manifest.get('minimum_speed_mps') != contract['attack_speed_threshold_mps']
            or manifest.get('protocol_sha256') != contract['protocol_sha256']):
        raise ValueError('Require result-filtered native executed recovery labels')
    source_contract = manifest['execution_contract']
    for key in EXECUTION_KEYS:
        if source_contract.get(key) != contract.get(key):
            raise ValueError('Executed correction changes physical/control interface: ' + key)
    tolerance = manifest.get('cpu_gpu_command_tolerance')
    if (not isinstance(tolerance, (float, int)) or not np.isfinite(tolerance)
            or not 0. <= tolerance <= 5e-4
            or not np.isfinite(manifest.get('max_command_error', np.inf))
            or manifest['max_command_error'] > tolerance):
        raise ValueError('Executed correction lacks production-command replay')
    data_path = path.parent / manifest['data_file']
    if data_path.name != manifest['data_file'] or digest(data_path) != manifest['data_sha256']:
        raise ValueError('Executed correction data identity differs')
    with np.load(data_path, allow_pickle=False) as data:
        public = data['observations'].copy()
        targets = data['residual_targets'].copy()
        offsets = data['episode_offsets'].copy()
    episodes = manifest['episodes']
    if (public.shape != (manifest['samples'], 794) or targets.shape != (len(public), 18)
            or not np.isfinite(public).all() or not np.isfinite(targets).all()
            or public.dtype != np.float32 or targets.dtype != np.float32
            or len(episodes) != manifest['episode_count'] or not episodes
            or offsets.shape != (len(episodes) + 1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(public) or not (np.diff(offsets) > 0).all()
            or not (public[:, 774] > .5).all()):
        raise ValueError('Invalid executed recovery arrays or phase labels')
    if any(not e.get('complete5') or not np.isfinite(e.get('max_command_error', np.inf))
           or e['max_command_error'] > tolerance or e['samples'] != offsets[i + 1] - offsets[i]
           for i, e in enumerate(episodes)):
        raise ValueError('Failed/unverified episode cannot supply positive correction labels')
    return (torch.from_numpy(public).to(device), torch.from_numpy(targets).to(device),
            torch.as_tensor(offsets, dtype=torch.int64), manifest)


def attach_executed_recovery_corrections(runner, configuration, contract):
    public, targets, offsets, manifest = load_dataset(configuration, contract, runner.device)
    if not getattr(runner.alg.policy, 'phase_gated_recovery', False):
        raise ValueError('Use an explicit recovery-local policy for executed recovery corrections')
    learning = SuccessRetention(runner.alg.policy, public, targets, offsets, configuration)
    expected = {id(p) for p in runner.alg.policy.recovery_head.parameters()}
    if {id(p) for p in learning.update_parameters} != expected:
        raise ValueError('Executed recovery labels must update only the recovery head')
    original_update, original_save, original_load = runner.alg.update, runner.save, runner.load

    def update():
        losses = original_update()
        stats = learning.step()
        stats.update(target_kind=manifest['target_kind'], positive_labels_from_full_success_only=True)
        runner.alg.last_update_stats['executed_recovery_corrections'] = stats
        # Diagnostics must cover PPO, old-success retention AND this update.
        alg = runner.alg
        with torch.no_grad():
            total = alg.storage.num_transitions_per_env * alg.storage.num_envs
            stride = max(1, total // 4096)
            alg.last_update_stats.update(post_update_diagnostics(alg.policy,
                alg.storage.observations.flatten(0, 1)[::stride],
                alg.storage.mu.flatten(0, 1)[::stride], alg.storage.sigma.flatten(0, 1)[::stride],
                alg.storage.actions.flatten(0, 1)[::stride],
                alg.storage.actions_log_prob.flatten(0, 1)[::stride], alg.clip_param))
        return losses

    def save(path, infos=None):
        combined = dict(infos or {})
        combined['executed_recovery_corrections_state'] = learning.state_dict()
        return original_save(path, infos=combined)

    def load(path, load_optimizer=True, map_location=None):
        if load_optimizer:
            source = torch.load(path, map_location='cpu', weights_only=False)
            saved = source.get('infos', {}).get('executed_recovery_corrections_state')
            if not saved or saved.get('configuration') != configuration:
                raise ValueError('Missing compatible executed-correction optimizer/sampler state')
        result = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            learning.load_state_dict(saved)
        return result

    runner.alg.update, runner.save, runner.load = update, save, load
    runner.executed_recovery_corrections = learning
    return dict(schema=SCHEMA, manifest_sha256=configuration['manifest_sha256'],
        episodes=manifest['episode_count'], cases=len(manifest['case_indices']), samples=manifest['samples'],
        command_replay_validated=True, optimizer_and_sampler_checkpointed=True,
        modifies_ppo_actions_or_log_prob=False, action_override_at_deployment=False)
