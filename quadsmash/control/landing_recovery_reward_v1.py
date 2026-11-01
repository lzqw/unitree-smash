"""Optional TRAIN feedback after the flight containing the actual strike.

This does not change the recovery event or its independent acceptance rules.
Support quality gives smooth feedback while the original longest continuous
stable window supplies bounded progress credit (short windows cannot stack).
"""
import math

import torch


VERSION = 'contact-flight-landing-feedback-v1'
DEFAULTS = dict(quality_weight_per_s=6., stable_progress_weight=20.,
                tilt_scale_rad=.5, horizontal_speed_scale_mps=.6,
                angular_speed_scale_rad_s=2., joint_speed_scale_rad_s=4.)


def settings(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('Declare all landing recovery TRAIN feedback settings')
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in value.values()):
        raise ValueError('Landing feedback settings must be positive finite numbers')
    return dict(value)


class LandingRecoveryRewardV1:
    def __init__(self, count, device, config, stable_window_s):
        self.config = settings(config)
        if not math.isfinite(stable_window_s) or stable_window_s <= 0:
            raise ValueError('Require the original continuous stable window')
        self.window = float(stable_window_s)
        self.credited = torch.zeros(count, device=device)
        self.paid_quality = torch.zeros((), dtype=torch.float64, device=device)
        self.paid_progress = torch.zeros_like(self.paid_quality)
        self.support_steps = torch.zeros((), dtype=torch.long, device=device)

    def reset(self, ids):
        self.credited[ids] = 0

    @torch.inference_mode()
    def step(self, *, legal_air, associated_landing, supported, robot_fault,
             body_ground_seen, tilt_rad, horizontal_speed_mps,
             angular_speed_rad_s, joint_speed_rad_s, longest_stable_s, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Require positive policy timestep')
        cfg = self.config
        measurements = torch.stack((tilt_rad, horizontal_speed_mps,
                                    angular_speed_rad_s, joint_speed_rad_s), 1)
        finite = torch.isfinite(measurements).all(1) & torch.isfinite(longest_stable_s)
        eligible = (legal_air & associated_landing & supported & ~robot_fault
                    & ~body_ground_seen & finite)
        scales = measurements.new_tensor([cfg['tilt_scale_rad'],
            cfg['horizontal_speed_scale_mps'], cfg['angular_speed_scale_rad_s'],
            cfg['joint_speed_scale_rad_s']])
        quality = torch.exp(-(torch.nan_to_num(measurements) / scales).square()).mean(1)
        quality_pay = cfg['quality_weight_per_s'] * quality * eligible * dt
        fraction = (torch.nan_to_num(longest_stable_s) / self.window).clamp(0., 1.)
        # No credit from a different jump or after a body/rule violation. Once
        # seen, a stable prefix is never paid again until that lane resets.
        progress = torch.where(eligible, (fraction-self.credited).clamp_min(0.), 0.)
        self.credited.add_(progress)
        progress_pay = cfg['stable_progress_weight'] * progress
        self.paid_quality.add_(quality_pay.double().sum())
        self.paid_progress.add_(progress_pay.double().sum())
        self.support_steps.add_(eligible.sum())
        return quality_pay + progress_pay

    def statistics(self):
        return dict(version=VERSION, quality_paid=float(self.paid_quality),
                    stable_progress_paid=float(self.paid_progress),
                    eligible_support_control_steps=int(self.support_steps),
                    original_stable_window_s=self.window)
