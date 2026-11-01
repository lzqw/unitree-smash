"""Actor-only rehearsal of verified complete native residual demonstrations.

PPO still supplies all online actions and rewards. Four small supervised steps
can follow each PPO update; these have their own optimizer and checkpointed
sampler. No teacher is queried for deployed actions.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from quadsmash.control.shared_target_assistance_v6 import RECOVERY_FIELDS
from quadsmash.control.persistent_recovery_assistance_v1 import SCHEMA as RECOVERY_SCHEMA
from quadsmash.rl.action_diagnostics import post_update_diagnostics

SCHEMA = 'complete-native-residual-success-retention-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(configuration, contract, device):
    required = {'manifest', 'manifest_sha256', 'learning_rate', 'steps_per_update',
                'batch_size', 'anchor_weight', 'seed'}
    if not isinstance(configuration, dict) or set(configuration) != required:
        raise ValueError('Declare the complete success-retention configuration')
    for key in ['steps_per_update', 'batch_size', 'seed']:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError('Invalid retention integer: ' + key)
    for key in ['learning_rate', 'anchor_weight']:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid retention scalar: ' + key)
        if value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError('Invalid retention scalar: ' + key)
    path = Path(configuration['manifest'])
    if digest(path) != configuration['manifest_sha256']:
        raise ValueError('Retention manifest changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'verified-fixed-success-residual-retention-v1'
            or manifest.get('target_interface') != 'raw-residual18-before-mixer'
            or manifest.get('minimum_speed_mps') != contract['attack_speed_threshold_mps']
            or manifest.get('protocol_sha256') != contract['protocol_sha256']
            or contract.get('recovery_assistance_execution') != RECOVERY_SCHEMA):
        raise ValueError('Retention dataset and task/execution contract differ')
    public_only_labels = manifest.get('label_validation') == 'actor-public-observation-v1'
    if public_only_labels and any(source.get('command_replay_validated', True) for source in manifest['sources']):
        raise ValueError('Public-observation retention source is inconsistently marked')
    for source in manifest['sources']:
        metadata = source['metadata']
        if (metadata.get('base_policy_sha256') != contract['base_policy_sha256']
                or source['residual_joint_scale_rad'] != contract['joint_scale_rad']
                or source['residual_arm_target_rate_rad_s'] != contract['arm_target_rate_rad_s']
                or metadata.get('support_assistance') != contract.get('support_assistance')
                or metadata.get('landing_assistance') != contract.get('landing_assistance')):
            raise ValueError('Residual labels require the original base and action interface')
        for name, (index, *_bounds) in RECOVERY_FIELDS.items():
            actual = (metadata['recovery_assistance'][name] if public_only_labels
                      else (source['effective_reference_parameters_after_execution'][index] if index is not None
                            else metadata['recovery_assistance'][name]))
            if not abs(contract['recovery_assistance'][name] - actual) <= 1e-7:
                raise ValueError('Retention recovery execution differs: ' + name)
    data_path = path.parent / manifest['data_file']
    if digest(data_path) != manifest['data_sha256']:
        raise ValueError('Retention data changed')
    with np.load(data_path, allow_pickle=False) as data:
        public, targets, offsets = [data[k].copy() for k in
                                   ['observations', 'residual_targets', 'episode_offsets']]
    if (public.shape != (manifest['samples'], 794) or targets.shape != (len(public), 18)
            or not np.isfinite(public).all() or not np.isfinite(targets).all()
            or len(manifest['episodes']) != manifest['episode_count']
            or offsets.shape != (manifest['episode_count']+1,)
            or offsets[0] != 0 or offsets[-1] != len(public) or not (np.diff(offsets) > 0).all()):
        raise ValueError('Invalid complete-episode retention arrays')
    if any(e['samples'] != offsets[i+1]-offsets[i] for i,e in enumerate(manifest['episodes'])):
        raise ValueError('Retention episode boundaries are invalid')
    if (not public_only_labels and any(e['max_command_error'] > manifest['cpu_gpu_command_tolerance']
           for e in manifest['episodes'])):
        raise ValueError('Retention episodes lack verified command replay')
    return (torch.from_numpy(public).to(device), torch.from_numpy(targets).to(device),
            torch.as_tensor(offsets, dtype=torch.int64), manifest)


class SuccessRetention:
    def __init__(self, policy, public, targets, offsets, configuration):
        self.policy, self.public, self.targets = policy, public, targets
        self.offsets = offsets.cpu().clone()
        self.configuration = copy.deepcopy(configuration)
        self.optimizer = torch.optim.Adam(policy.actor.parameters(), lr=configuration['learning_rate'])
        self.generator = torch.Generator(device='cpu').manual_seed(configuration['seed'])
        self.completed_steps = 0

    def state_dict(self):
        return dict(schema=SCHEMA, configuration=self.configuration,
                    optimizer=self.optimizer.state_dict(), sampler_state=self.generator.get_state(),
                    completed_steps=self.completed_steps)

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA or state.get('configuration') != self.configuration:
            raise ValueError('Retention resume configuration differs')
        self.optimizer.load_state_dict(state['optimizer'])
        self.generator.set_state(state['sampler_state'].cpu())
        self.completed_steps = state['completed_steps']

    def step(self):
        if not any(p.requires_grad for p in self.policy.actor.parameters()):
            return dict(skipped_actor_warmup=True, completed_steps=self.completed_steps)
        if not all(p.requires_grad for p in self.policy.actor.parameters()):
            raise ValueError('Partially frozen actors require a separate retention contract')
        cfg = self.configuration
        count = cfg['steps_per_update'] * cfg['batch_size']
        episode = torch.randint(len(self.offsets)-1, (count,), generator=self.generator)
        lengths = self.offsets[episode+1]-self.offsets[episode]
        within = (torch.rand(count, generator=self.generator)*lengths).long()
        ids = (self.offsets[episode]+within).to(self.public.device)
        observations = self.public[ids]
        targets = self.targets[ids]
        # A soft anchor to this update's PPO actor keeps the rehearsal local.
        with torch.no_grad():
            anchor = self.policy.act_inference(TensorDict({'policy': observations}, batch_size=[count])).detach()
        before = float((anchor-targets).square().mean())
        fit_values, anchor_values, gradients = [], [], []
        for start in range(0, count, cfg['batch_size']):
            sl = slice(start, start+cfg['batch_size'])
            obs = TensorDict({'policy': observations[sl]}, batch_size=[cfg['batch_size']])
            pred = self.policy.act_inference(obs)
            fit = (pred-targets[sl]).square().mean()
            drift = (pred-anchor[sl]).square().mean()
            loss = fit + cfg['anchor_weight']*drift
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite retention loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            if not all(torch.isfinite(p).all() for p in self.policy.actor.parameters()):
                raise RuntimeError('Nonfinite retention actor')
            self.completed_steps += 1
            fit_values.append(float(fit.detach())); anchor_values.append(float(drift.detach()))
            gradients.append(float(norm))
        with torch.no_grad():
            final = self.policy.act_inference(TensorDict({'policy': observations}, batch_size=[count]))
            after = float((final-targets).square().mean())
        return dict(skipped_actor_warmup=False, completed_steps=self.completed_steps,
                    sampled_rows=count, episode_balanced_sampling=True,
                    mse_before=before, mse_after=after,
                    mean_fit_loss=sum(fit_values)/len(fit_values),
                    mean_anchor_loss=sum(anchor_values)/len(anchor_values),
                    max_gradient_norm_before_clip=max(gradients))


def attach_success_retention(runner, configuration, contract):
    public, targets, offsets, manifest = load_dataset(configuration, contract, runner.device)
    retention = SuccessRetention(runner.alg.policy, public, targets, offsets, configuration)
    original_update, original_save, original_load = runner.alg.update, runner.save, runner.load

    def update():
        losses = original_update()
        stats = retention.step()
        runner.alg.last_update_stats['success_retention'] = stats
        # Recompute diagnostics after both updates, on the existing rollout.
        alg = runner.alg
        with torch.no_grad():
            total = alg.storage.num_transitions_per_env*alg.storage.num_envs
            stride = max(1, total//4096)
            runner.alg.last_update_stats.update(post_update_diagnostics(alg.policy,
                alg.storage.observations.flatten(0,1)[::stride],
                alg.storage.mu.flatten(0,1)[::stride], alg.storage.sigma.flatten(0,1)[::stride],
                alg.storage.actions.flatten(0,1)[::stride],
                alg.storage.actions_log_prob.flatten(0,1)[::stride], alg.clip_param))
        return losses

    def save(path, infos=None):
        combined = dict(infos or {})
        combined['success_retention_state'] = retention.state_dict()
        return original_save(path, infos=combined)

    def load(path, load_optimizer=True, map_location=None):
        if load_optimizer:
            source = torch.load(path, map_location='cpu', weights_only=False)
            saved = source.get('infos', {}).get('success_retention_state')
            if not saved or saved.get('configuration') != configuration or saved.get('schema') != SCHEMA:
                raise ValueError('Missing compatible retention optimizer/sampler restart state')
        result = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            retention.load_state_dict(saved)
        return result

    runner.alg.update, runner.save, runner.load = update, save, load
    runner.success_retention = retention
    return dict(schema=SCHEMA, manifest_sha256=configuration['manifest_sha256'],
                episodes=manifest['episode_count'], samples=manifest['samples'],
                optimizer_and_sampler_checkpointed=True, teacher_action_override=False,
                dataset_execution_validated=True)
