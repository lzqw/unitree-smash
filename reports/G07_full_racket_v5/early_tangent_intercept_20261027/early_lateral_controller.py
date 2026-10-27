"""Public causal early lateral interception over an unchanged measured teacher.

Only six arm position commands change. The source actors, native physics,
forecast model, material contact point, complete tool support guard and5rad/s
target limiter are retained. This is an unexecuted control proposal.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import torch

from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_arm_target_rate_v4 import ArmTargetRateV4

SCHEMA = 'g07-public-early-lateral-intercept-v5'


def candidates():
    result = [dict(candidate_id=0, name='unchanged_source', gain=0., start_clock_s=2.0)]
    for start in [2.0, 2.3]:
        for gain in [.5, 1., 2., 4.]:
            result.append(dict(candidate_id=len(result), name=f'lateral_start{start:g}_gain{gain:g}',
                               gain=gain, start_clock_s=start))
    return result


def source_controller(source, count, device):
    if source.get('schema') == 'g07-measured-swing-contact-recovery-v5':
        path = Path(__file__).resolve().parents[1] / 'origin0_measured_swing_velocity_20261027/swing_controller.py'
        if not source.get('native_execution') or hashlib.sha256(path.read_bytes()).hexdigest() != source['factory_sha256']:
            raise ValueError('Require a measured swing source with the unchanged exact factory')
        spec = importlib.util.spec_from_file_location('preserved_measured_swing_factory', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MeasuredSwingController(source['source_public_policy'], count, device, source['runtime_overrides'])
    return make_rally_controller(source, count, device)


class EarlyLateralController:
    def __init__(self, source, count, device, gain, start_clock_s):
        if not math.isfinite(gain) or gain not in [0., .5, 1., 2., 4.]:
            raise ValueError('Require a declared bounded lateral gain')
        if start_clock_s not in [2.0, 2.3]:
            raise ValueError('Require a declared public-clock preparation start')
        self.source = source_controller(source, count, device)
        self.reference = self.source.reference_controller
        self.teacher = self.reference.teacher
        if self.teacher.joint_preparation is None or self.teacher.arm_target_rate.rate != 5.:
            raise ValueError('Preserve the source complete-tool support guard and5rad/s rate limit')
        self.gain, self.start = float(gain), float(start_clock_s)
        self.rate = ArmTargetRateV4(count, device, 5.)
        self.valid_plan = torch.zeros(count, dtype=torch.bool, device=device)
        self.ever_active = torch.zeros_like(self.valid_plan)
        self.plan_cursor = 0
        self.eye = torch.eye(6, device=device)[None]
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
        self.rate.reset(ids)
        if ids is None:
            self.valid_plan.zero_()
            self.ever_active.zero_()
            self.plan_cursor = 0
            self.last_diagnostic = None
        else:
            self.valid_plan[ids] = False
            self.ever_active[ids] = False

    @torch.inference_mode()
    def action(self, observation, default, limits):
        obs = observation['policy']
        base = self.source.action(observation, default, limits)
        if self.gain == 0.:
            return base
        # The source forecast is computed from the present794D public input
        # at ball appearance. Read only that already causal plan, never labels
        # or an environment/simulator/case object.
        for entry in self.reference.logs[self.plan_cursor:]:
            ids = torch.tensor(entry['env_ids'], device=obs.device, dtype=torch.long)
            self.valid_plan[ids] = torch.tensor(entry['valid'], device=obs.device, dtype=torch.bool)
        self.plan_cursor = len(self.reference.logs)
        time, event = obs[:, 775], self.reference.event_time
        envelope = ((time - self.start) / .12).clamp(0., 1.)
        envelope *= ((event + .05 - time) / .10).clamp(0., 1.)
        active = self.valid_plan & (obs[:, 67] > .5) & (obs[:, 773] < .5) & (envelope > 0)
        self.ever_active |= active
        arm_limits = limits[12:]
        desired = arm_limits[:, 0] + .5 * (base[:, 12:] + 1) * (arm_limits[:, 1] - arm_limits[:, 0])
        actual = (default[12:][None] + obs[:, 21:27]).clamp(arm_limits[:, 0], arm_limits[:, 1])
        point, normal, jp, jn = self.teacher.kinematics(actual, self.teacher.contact_offset)
        forward, up = obs[:, 763:766], -obs[:, 6:9]
        left = torch.linalg.cross(up, forward)
        left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-6)
        known = obs.new_tensor([1.10, 0., 1.80])
        offset = self.teacher.target_world[None] + self.reference.target_offset_world - known
        target = obs[:, 760:763] + offset[:, :1] * forward + offset[:, 1:2] * left + offset[:, 2:3] * up
        error_y = ((target - point) * left).sum(1)
        error = torch.cat((left * error_y[:, None], torch.zeros_like(point)), 1)
        jacobian = torch.cat((jp, .3 * jn), 1)
        solved = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + .04**2 * self.eye, error[:, :, None])
        delta = (jacobian.transpose(1, 2) @ solved).squeeze(-1)
        correction = (self.gain * envelope[:, None] * delta).clamp(-.4, .4)
        correction = torch.where(active[:, None], correction, torch.zeros_like(correction))
        candidate = (desired + correction).clamp(arm_limits[:, 0], arm_limits[:, 1])
        center = event - self.reference.parameters[:, 14]
        half_window = self.reference.parameters[:, 19]
        guarded = self.teacher.joint_preparation(candidate, obs, arm_limits,
            self.teacher.kinematics, actual, post_stroke_mask=time >= center + half_window)
        # The original source path stays exact before any new correction.
        # Once this lane has changed, retain a final independent slew limiter
        # while fading back to the original arm commands after the stroke.
        requested = torch.where(active[:, None], guarded, desired)
        bounded = self.rate(requested, actual, time)
        result = base.clone()
        arm_action = (2 * (bounded - arm_limits[:, 0]) / (arm_limits[:, 1] - arm_limits[:, 0]) - 1).clamp(-1., 1.)
        result[:, 12:] = torch.where(self.ever_active[:, None], arm_action, base[:, 12:])
        self.last_diagnostic = dict(active=active.clone(), public_forecast_target_body_m=target.clone(),
            measured_material_point_body_m=point.clone(), error_world_y_m=error_y.clone(),
            requested_joint_correction_rad=correction.clone(), requested_arm_after_original_guard=guarded.clone(),
            actual_limited_arm_target_rad=bounded.clone())
        return result
