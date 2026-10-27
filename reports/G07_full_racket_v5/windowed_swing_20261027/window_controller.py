"""Change swing feedforward velocity only inside the causal stroke window.

The instantiated teacher receives a temporary parameter tensor. Original
preparation, source tensors, actors, feedback, tool guards and slew limits stay
in its execution path. This does not prescribe actual physical head velocity.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

from quadsmash.control.full_racket_arm_search_v4 import arm_velocity
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller

HERE = Path(__file__).resolve().parent
SCHEMA = 'g07-windowed-swing-contact-recovery-v5'


def make_source(source, count, device):
    if not source.get('native_execution'):
        raise ValueError('Require an actually executed source')
    if source['schema'] in ['g07-measured-swing-contact-recovery-v5',
                            'g07-public-early-lateral-intercept-v5']:
        path = HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py'
        spec = importlib.util.spec_from_file_location('window_preserved_lateral_factory', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if source['schema'] == 'g07-public-early-lateral-intercept-v5':
            if hashlib.sha256(path.read_bytes()).hexdigest() != source['factory_sha256']:
                raise ValueError('Measured source factory changed')
            return module.EarlyLateralController(source['source_public_policy'], count, device,
                source['early_lateral']['gain'], source['early_lateral']['start_clock_s'])
        return module.source_controller(source, count, device)
    return make_rally_controller(source, count, device)


def window_envelope(offset, half_window):
    relative = offset / half_window
    return torch.where(abs(relative) < 1., (1. - relative.square()).square(), torch.zeros_like(relative))


class WindowedSwingController:
    def __init__(self, source, count, device, joint_velocity_rad_s=None):
        self.source = make_source(source, count, device)
        self.reference = self.source.reference_controller
        self.teacher = self.reference.teacher
        if self.teacher.preparation is not None:
            raise ValueError('Cartesian preparation also uses velocity gain; require measured joint preparation')
        if self.teacher.joint_preparation is None or self.teacher.arm_target_rate.rate != 5.:
            raise ValueError('Preserve measured complete-tool guard and5rad/s limiter')
        self.velocity = None
        if joint_velocity_rad_s is not None:
            velocity = torch.as_tensor(joint_velocity_rad_s, device=device, dtype=torch.float32)
            if velocity.shape != (6,) or not torch.isfinite(velocity).all() or (abs(velocity) > 5.).any():
                raise ValueError('Require one shared finite6D velocity within original5rad/s seed bounds')
            self.velocity = velocity
        self.original_teacher_action = self.teacher.action
        # Only this instantiated analytic teacher is wrapped; no class, source
        # file, checkpoint, reference parameters or simulator object is changed.
        self.teacher.action = self.teacher_action
        self.last_diagnostic = None

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
        if ids is None:
            self.last_diagnostic = None

    def teacher_action(self, observation, parameters, default, limits, event_time=None, target_offset_world=None):
        time = observation['policy'][:, 775]
        event = torch.full_like(time, 2.75) if event_time is None else event_time
        center = event - parameters[:, 14]
        half_window = parameters[:, 19]
        if not (half_window > 0.).all():
            raise ValueError('Require the original positive swing half-window')
        envelope = window_envelope(time - center, half_window)
        active = envelope > 0.
        effective = parameters
        old_velocity = arm_velocity(parameters, self.teacher.arm_velocity)
        effective_velocity = old_velocity
        if self.velocity is not None and bool(active.any()):
            effective_velocity = old_velocity + envelope[:, None] * (self.velocity[None] - old_velocity)
            effective = parameters.clone()
            effective[active, 7] = 1.
            effective[active, 26:32] = effective_velocity[active]
        self.last_diagnostic = dict(center_s=center.clone(), half_window_s=half_window.clone(),
            envelope=envelope.clone(), active=active.clone(), original_velocity=old_velocity.clone(),
            effective_velocity=effective_velocity.clone(),
            feedforward_joint_displacement_rad=(time-center)[:, None] * (effective_velocity-old_velocity))
        return self.original_teacher_action(observation, effective, default, limits,
            event_time=event_time, target_offset_world=target_offset_world)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)
