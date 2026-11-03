"""Small actor-only guidance updates from causal recovery-failure states."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

SCHEMA = 'failure-recovery-guidance-v1'


def actor_update_parameters(policy):
    """Return parameters explicitly exposed by a phase-local recovery policy."""
    method = getattr(policy, 'actor_update_parameters', None)
    return list(method()) if method is not None else list(policy.actor.parameters())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(configuration, contract, device):
    required = {'manifest', 'manifest_sha256', 'learning_rate', 'steps_per_update',
                'batch_size', 'anchor_weight', 'seed'}
    if not isinstance(configuration, dict) or set(configuration) != required:
        raise ValueError('Declare the complete failure-guidance configuration')
    for key in ['steps_per_update', 'batch_size', 'seed']:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Invalid guidance integer: ' + key)
    for key in ['learning_rate', 'anchor_weight']:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid guidance scalar: ' + key)
    path = Path(configuration['manifest'])
    if digest(path) != configuration['manifest_sha256']:
        raise ValueError('Failure-guidance manifest changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('target_interface') != 'raw-residual18-before-mixer'
            or not manifest.get('failure_states_are_not_expert_success')):
        raise ValueError('Invalid failure-guidance manifest')
    data_path = path.parent / manifest['data_file']
    if digest(data_path) != manifest['data_sha256']:
        raise ValueError('Failure-guidance data changed')
    with np.load(data_path, allow_pickle=False) as data:
        public = data['observations'].copy()
        targets = data['residual_targets'].copy()
    if (public.shape != (manifest['samples'], 794) or targets.shape != (len(public), 18)
            or not np.isfinite(public).all() or not np.isfinite(targets).all()):
        raise ValueError('Invalid failure-guidance arrays')
    return torch.from_numpy(public).to(device), torch.from_numpy(targets).to(device), manifest


class FailureRecoveryGuidance:
    def __init__(self, policy, public, targets, configuration, target_kind):
        self.policy, self.public, self.targets = policy, public, targets
        self.configuration = copy.deepcopy(configuration)
        self.target_kind = str(target_kind)
        self.update_parameters = actor_update_parameters(policy)
        self.optimizer = torch.optim.Adam(self.update_parameters, lr=configuration['learning_rate'])
        self.generator = torch.Generator(device='cpu').manual_seed(configuration['seed'])
        self.completed_steps = 0

    def state_dict(self):
        return dict(schema=SCHEMA, configuration=self.configuration,
                    optimizer=self.optimizer.state_dict(), sampler_state=self.generator.get_state(),
                    completed_steps=self.completed_steps)

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA or state.get('configuration') != self.configuration:
            raise ValueError('Failure-guidance resume configuration differs')
        self.optimizer.load_state_dict(state['optimizer'])
        self.generator.set_state(state['sampler_state'].cpu())
        self.completed_steps = state['completed_steps']

    def step(self):
        if not any(p.requires_grad for p in self.update_parameters):
            return dict(skipped_actor_warmup=True, completed_steps=self.completed_steps)
        if not all(p.requires_grad for p in self.update_parameters):
            raise ValueError('Partially frozen actors require a separate failure-guidance contract')
        cfg = self.configuration
        count = cfg['steps_per_update'] * cfg['batch_size']
        ids = torch.randint(len(self.public), (count,), generator=self.generator)
        observations, targets = self.public[ids.to(self.public.device)], self.targets[ids.to(self.public.device)]
        with torch.no_grad():
            anchor = self.policy.act_inference(TensorDict({'policy': observations}, batch_size=[count])).detach()
        before = float((anchor - targets).square().mean())
        fit_values, anchor_values, gradients = [], [], []
        for start in range(0, count, cfg['batch_size']):
            sl = slice(start, start + cfg['batch_size'])
            obs = TensorDict({'policy': observations[sl]}, batch_size=[cfg['batch_size']])
            pred = self.policy.act_inference(obs)
            fit = (pred - targets[sl]).square().mean()
            drift = (pred - anchor[sl]).square().mean()
            loss = fit + cfg['anchor_weight'] * drift
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite failure-guidance loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.update_parameters, 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.completed_steps += 1
            fit_values.append(float(fit.detach())); anchor_values.append(float(drift.detach())); gradients.append(float(norm))
        with torch.no_grad():
            final = self.policy.act_inference(TensorDict({'policy': observations}, batch_size=[count]))
            after = float((final - targets).square().mean())
        return dict(completed_steps=self.completed_steps, sampled_rows=count, phase_guidance=True,
                    target_kind=self.target_kind, mse_before=before, mse_after=after,
                    mean_fit_loss=sum(fit_values) / len(fit_values),
                    mean_anchor_loss=sum(anchor_values) / len(anchor_values),
                    max_gradient_norm_before_clip=max(gradients))


def attach_failure_recovery_guidance(runner, configuration, contract):
    public, targets, manifest = load_dataset(configuration, contract, runner.device)
    guidance = FailureRecoveryGuidance(runner.alg.policy, public, targets, configuration,
                                       manifest.get('target_kind', 'unknown'))
    original_update, original_save, original_load = runner.alg.update, runner.save, runner.load

    def update():
        losses = original_update()
        stats = guidance.step()
        runner.alg.last_update_stats['failure_recovery_guidance'] = stats
        return losses

    def save(path, infos=None):
        combined = dict(infos or {})
        combined['failure_recovery_guidance_state'] = guidance.state_dict()
        return original_save(path, infos=combined)

    def load(path, load_optimizer=True, map_location=None):
        if load_optimizer:
            source = torch.load(path, map_location='cpu', weights_only=False)
            saved = source.get('infos', {}).get('failure_recovery_guidance_state')
            if not saved or saved.get('configuration') != configuration or saved.get('schema') != SCHEMA:
                raise ValueError('Missing compatible failure-guidance restart state')
        result = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            guidance.load_state_dict(saved)
        return result

    runner.alg.update, runner.save, runner.load = update, save, load
    runner.failure_recovery_guidance = guidance
    return dict(schema=SCHEMA, manifest_sha256=configuration['manifest_sha256'],
                episodes=manifest['episode_count'], samples=manifest['samples'],
                optimizer_and_sampler_checkpointed=True, teacher_action_override=False,
                failure_states_are_not_expert_success=True)
