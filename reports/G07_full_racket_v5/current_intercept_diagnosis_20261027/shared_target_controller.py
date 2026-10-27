"""Shared world y/z reference bias through the unchanged teacher path.

This changes a control reference only. The causal forecast, hardware/asset,
actors, native physics, full-tool guard and final arm limiter are preserved.
There is no simulator, case, future contact point or ground-truth pose input.
"""
import math

import torch


def candidates():
    return [dict(candidate_id=i, name=name, bias_world_m=list(bias))
        for i, (name, bias) in enumerate([
            ('unchanged_source', (0., 0., 0.)),
            ('shared_y_minus005', (0., -.05, 0.)),
            ('shared_y_minus005_z_plus004', (0., -.05, .04)),
            ('shared_y_minus010_z_plus004', (0., -.10, .04)),
            ('shared_y_minus010_z_plus008', (0., -.10, .08)),
        ])]


class SharedTargetController:
    def __init__(self, source_controller, count, device, *, bias_world_m):
        if len(bias_world_m) != 3 or not all(math.isfinite(x) for x in bias_world_m):
            raise ValueError('Require a finite shared world3D reference bias')
        allowed = [c['bias_world_m'] for c in candidates()]
        if list(bias_world_m) not in allowed:
            raise ValueError('Require a declared bounded shared y/z reference bias')
        self.source = source_controller
        self.reference = self.source.reference_controller
        self.teacher = self.reference.teacher
        if self.teacher.joint_preparation is None or self.teacher.arm_target_rate.rate != 5.:
            raise ValueError('Require the preserved complete-tool guard and5rad/s arm target limiter')
        self.bias = torch.as_tensor(bias_world_m, device=device, dtype=torch.float32)
        self.changed = bool((self.bias != 0).any())
        self.valid_plan = torch.zeros(count, device=device, dtype=torch.bool)
        self.plan_cursor = 0
        self.last_diagnostic = None
        self._inside_action = False
        self._original_offset = None
        self._teacher_calls = 0
        self.original_teacher_action = self.teacher.action
        if self.changed:
            self.teacher.action = self.teacher_action

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
        if self._inside_action:
            raise RuntimeError('Reset must not interrupt a reference action')
        self.source.reset(ids)
        if ids is None:
            self.valid_plan.zero_()
            self.plan_cursor = 0
            self.last_diagnostic = None
        else:
            self.valid_plan[ids] = False

    def teacher_action(self, observation, parameters, default, limits, event_time=None, target_offset_world=None):
        if not self._inside_action or self._teacher_calls:
            raise RuntimeError('Require exactly one original analytic teacher call per public action')
        self._teacher_calls += 1
        public = observation['policy']
        if public.ndim != 2 or public.shape[1] != 794:
            raise ValueError('Require the original794D public input')
        if target_offset_world is not self._original_offset:
            raise ValueError('Require the original shared causal target-offset tensor')
        for entry in self.reference.logs[self.plan_cursor:]:
            ids = torch.as_tensor(entry['env_ids'], device=public.device, dtype=torch.long)
            self.valid_plan[ids] = torch.as_tensor(entry['valid'], device=public.device, dtype=torch.bool)
        self.plan_cursor = len(self.reference.logs)
        time = public[:, 775]
        event = self.reference.event_time
        envelope = ((time-2.)/.12).clamp(0., 1.)
        active = (self.valid_plan & (public[:, 67] > .5) & (public[:, 773] < .5)
                  & (time >= 2.) & (time <= event+.12))
        amount = envelope*active
        effective = target_offset_world+amount[:, None]*self.bias[None]
        # The same temporary tensor must be visible to the original teacher
        # AND source early-lateral controller after its nested base action.
        # The original causal-plan tensor is restored by action's finally.
        self.reference.target_offset_world = effective
        self.last_diagnostic = dict(active=active.clone(), envelope=envelope.clone(),
            added_world_target_bias_m=(amount[:, None]*self.bias[None]).clone(),
            causal_offset_world_m=target_offset_world.clone(), effective_offset_world_m=effective.clone())
        return self.original_teacher_action(observation, parameters, default, limits,
            event_time=event_time, target_offset_world=effective)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        if not self.changed:
            return self.source.action(observation, default, limits)
        if self._inside_action:
            raise RuntimeError('Reference control is not reentrant')
        self._inside_action = True
        self._teacher_calls = 0
        self._original_offset = self.reference.target_offset_world
        try:
            action = self.source.action(observation, default, limits)
            if self._teacher_calls != 1:
                raise RuntimeError('Expected one original analytic teacher call')
            return action
        finally:
            self.reference.target_offset_world = self._original_offset
            self._original_offset = None
            self._inside_action = False
