"""Proposal-only shared swing parameters over intact source actors and guards."""
import torch

from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_prepose_branch_v4 import validate_overrides


class MeasuredSwingController:
    def __init__(self, source, count, device, overrides):
        if set(overrides) - {'swing_joint_velocity_rad_s', 'desired_normal_world'}:
            raise ValueError('This proposal changes only shared swing direction and optional normal')
        self.overrides = validate_overrides(overrides)
        self.source = make_rally_controller(source, count, device)
        self.reference = self.source.reference_controller
        self.runtime_matrices = []
        node = self.source
        visited = set()
        while id(node) not in visited:
            visited.add(id(node))
            if hasattr(node, 'runtime_parameters'):
                self.runtime_matrices.append(node.runtime_parameters)
            if hasattr(node, 'source'):
                node = node.source
            elif hasattr(node, 'base') and not isinstance(node.base, torch.Tensor):
                node = node.base
            else:
                break
        if len(self.runtime_matrices) != 2:
            raise ValueError('Require the measured gain/body and preparation runtime matrices')
        if self.overrides:
            self._apply(self.reference.parameters)
            for parameters in self.runtime_matrices:
                self._apply(parameters)
            if 'desired_normal_world' in self.overrides:
                self.reference.teacher.normal_world = self.reference.teacher.arm.new_tensor(
                    self.overrides['desired_normal_world'])

    def _apply(self, parameters):
        if 'swing_joint_velocity_rad_s' in self.overrides:
            parameters[:, 7] = 1.
            parameters[:, 26:32] = parameters.new_tensor(self.overrides['swing_joint_velocity_rad_s'])
        if 'desired_normal_world' in self.overrides:
            parameters[:, 37] = 0.

    @property
    def reference_controller(self):
        return self.reference

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    def reset(self, ids=None):
        self.source.reset(ids)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)
