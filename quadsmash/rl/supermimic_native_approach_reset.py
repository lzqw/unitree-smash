"""Candidate native restart from a control boundary, checked against continuation."""
from dataclasses import dataclass
import copy

import numpy as np
import torch

from .supermimic_g07_memory_preview_v1 import clone_execution_graph
from .supermimic_imagination_contract_v1 import TIMING_FIELDS


def _tensors(owner):
    return {name: value.detach().clone() for name, value in vars(owner).items()
            if isinstance(value, torch.Tensor)}


def _restore_tensors(owner, values):
    for name, saved in values.items():
        target = getattr(owner, name)
        if target.shape != saved.shape or target.dtype != saved.dtype:
            raise ValueError(f'Native restart tensor changed shape or dtype: {name}')
        # Some static buffers are expanded views of the same underlying lane.
        # Restore only the one physically stored slice in such dimensions.
        for dimension in reversed(range(target.ndim)):
            if target.shape[dimension] > 1 and target.stride(dimension) == 0:
                first = saved.select(dimension, 0).unsqueeze(dimension)
                if not torch.allclose(saved, first.expand_as(saved), atol=0., rtol=0., equal_nan=True):
                    raise ValueError(f'Cannot restore different lanes into shared buffer: {name}')
                target, saved = target.select(dimension, 0), saved.select(dimension, 0)
        target.copy_(saved)


def _task_owners(env):
    return {name: getattr(env, name) for name in (
        'rally', 'contact_recovery', 'contact_events', 'policy_history',
        '_sm_native_returns', 'approach_ball_termination') if getattr(env, name, None) is not None}


@dataclass
class NativeApproachRestart:
    """Exposed PhysX state and owned task/control memory for one complete scene.

    Callers must validate continuation because PhysX does not expose its
    internal contact, constraint solver and sensor caches.
    """
    scene_state: dict
    buffers: dict
    tasks: dict
    clocks: dict
    history_sizes: dict
    first_contacts: dict | None
    controller: object
    drive_targets: tuple
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor
    numpy_rng: tuple

    @classmethod
    @torch.inference_mode()
    def capture(cls, env):
        from .supermimic_physical_loop_loading_v1 import freeze_execution_modules

        freeze_execution_modules(env._sm_residual_controller)
        return cls(
            scene_state=env.scene.get_state(is_relative=True),
            buffers=_tensors(env),
            tasks={name: _tensors(owner) for name, owner in _task_owners(env).items()},
            clocks={name: int(getattr(env, name)) for name in (*TIMING_FIELDS, 'common_step_counter')},
            history_sizes={name: len(getattr(env, name)) for name in (
                'rally_records', 'contact_records', 'racket_plane_records')},
            first_contacts=copy.deepcopy(getattr(env, 'first_approach_contact_motion', None)),
            controller=clone_execution_graph(env._sm_residual_controller),
            drive_targets=tuple(getattr(env.robot.data, key).detach().clone() for key in
                                ('joint_pos_target', 'joint_vel_target', 'joint_effort_target')),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(env.device),
            numpy_rng=np.random.get_state())

    @torch.inference_mode()
    def restore(self, env):
        env.scene.reset_to(self.scene_state, is_relative=True)
        _restore_tensors(env, self.buffers)
        for name, values in self.tasks.items():
            _restore_tensors(getattr(env, name), values)
        for name, value in self.clocks.items():
            setattr(env, name, value)
        for name, size in self.history_sizes.items():
            del getattr(env, name)[size:]
        if self.first_contacts is not None:
            env.first_approach_contact_motion = copy.deepcopy(self.first_contacts)
        env._sm_residual_controller = clone_execution_graph(self.controller)
        position_target, velocity_target, effort_target = self.drive_targets
        env.robot.set_joint_position_target(position_target)
        env.robot.set_joint_velocity_target(velocity_target)
        env.robot.set_joint_effort_target(effort_target)
        env.scene.write_data_to_sim()
        env.sim.forward()
        torch.set_rng_state(self.cpu_rng)
        torch.cuda.set_rng_state(self.cuda_rng, env.device)
        np.random.set_state(self.numpy_rng)


@torch.inference_mode()
def observe_restart_state(env, observation):
    return dict(
        public=observation['policy'].detach().clone(),
        joint=env.robot.data.joint_pos.detach().clone(),
        root=env.robot.data.root_link_state_w.detach().clone(),
        shuttle=env.shuttle.data.root_state_w.detach().clone(),
        executed_target=env._sm_joint_targets.detach().clone(),
        episode=env.episode_index.detach().clone(),
        contact=torch.stack((env.rally.had_contact, env.rally.frame_hit, env.rally.string_hit,
                             env.rally.released, env.rally.crossed_net)).detach().clone(),
    )


def restart_comparison(expected, actual):
    if len(expected) != 64 or len(actual) != 64:
        raise ValueError('Restart validation requires 64 native control ticks')
    tolerances = dict(public=2e-4, joint=1e-5, root=1e-5, shuttle=2e-5, executed_target=2e-5)
    errors = {field: max(float((a[field] - b[field]).abs().amax()) for a, b in zip(expected, actual))
              for field in tolerances}
    contact_equal = all(torch.equal(a['contact'], b['contact']) for a, b in zip(expected, actual))
    episodes_equal = all(torch.equal(a['episode'], b['episode']) for a, b in zip(expected, actual))
    passed = contact_equal and episodes_equal and all(errors[k] <= tolerances[k] for k in errors)
    return dict(passed=passed, ticks=64, maximum_absolute_error=errors, tolerances=tolerances,
                contact_equal=contact_equal, episodes_equal=episodes_equal,
                collection_mode='verified_scene_restart' if passed else 'native_ground_prefix_replay')


@torch.inference_mode()
def verify_resident_snapshot(env, wrapper, observation, request):
    """Recover the same scene only when uninterrupted replay agrees at all audit ticks."""
    initial_observation = {name: value.clone() for name, value in observation.items()}

    def failed(exc):
        return initial_observation, dict(passed=False, collection_mode='native_ground_prefix_replay',
                                         error_type=type(exc).__name__, error=str(exc))

    try:
        snapshot = NativeApproachRestart.capture(env)
    except (RuntimeError, ValueError) as exc:
        return failed(exc)
    commands, expected, actual = [], [], []
    for _ in range(64):
        command = request(observation)
        commands.append(command.clone())
        observation, _, _, _ = wrapper.step(command)
        expected.append(observe_restart_state(env, observation))
    try:
        snapshot.restore(env)
    except (RuntimeError, ValueError) as exc:
        return failed(exc)
    for command in commands:
        observation, _, _, _ = wrapper.step(command)
        actual.append(observe_restart_state(env, observation))
    result = restart_comparison(expected, actual)
    if result['passed']:
        try:
            snapshot.restore(env)
        except (RuntimeError, ValueError) as exc:
            return failed(exc)
    return initial_observation, result
