"""Past observed air/contact memory for the existing command support guard.

The actual794D input and neural actor inputs remain untouched. A temporary
planning feature carries a bounded memory of a phase that actually occurred.
Native rally rules, legality and collision scoring receive no modified input.
"""
import importlib.util
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def make_base(source, count, device):
    path = HERE.parent / 'recovery_motion_prior_20261027/prior_controller.py'
    spec = importlib.util.spec_from_file_location('preserved_origin0_source_factory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_base(source, count, device)


class ObservedContactMemoryGuard:
    def __init__(self, original, count, device, seconds):
        if not math.isfinite(seconds) or not 0 < seconds <= .6:
            raise ValueError('Bound the extra planning memory to at most0.6s')
        self.original = original
        self.config = original.config
        self.seconds = float(seconds)
        self.seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_clock = torch.zeros(count, device=device)
        self.last_extended = torch.zeros_like(self.seen)

    def reset(self, ids=None):
        if ids is None:
            self.seen.zero_()
            self.first_clock.zero_()
            self.last_extended.zero_()
        else:
            self.seen[ids] = False
            self.first_clock[ids] = 0.
            self.last_extended[ids] = False

    def __call__(self, arm, observation, limits, kinematics, actual_joints=None, post_stroke_mask=None):
        time = observation[:, 775]
        observed = observation[:, 773] > .5
        first = observed & (~self.seen)
        self.first_clock[first] = time[first]
        self.seen |= observed
        extended = self.seen & (~observed) & (time - self.first_clock <= self.seconds)
        self.last_extended.copy_(extended)
        if not bool(extended.any()):
            return self.original(arm, observation, limits, kinematics, actual_joints, post_stroke_mask)
        planning = observation.clone()
        planning[:, 773] = (observed | extended).to(observation.dtype)
        return self.original(arm, planning, limits, kinematics, actual_joints, post_stroke_mask)


class ContactMemoryController:
    def __init__(self, source, count, device, seconds):
        if seconds not in [0., .4, .6]:
            raise ValueError('Require a declared memory setting')
        self.source = make_base(source, count, device)
        self.reference = self.source.reference_controller
        self.guard = None
        if seconds:
            original = self.reference.teacher.joint_preparation
            if original is None or not original.config['protect_post_hit_net_volume']:
                raise ValueError('Preserve the source full-tool post-contact net-volume guard')
            self.guard = ObservedContactMemoryGuard(original, count, device, seconds)
            self.reference.teacher.joint_preparation = self.guard

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
        if self.guard is not None:
            self.guard.reset(ids)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)
