"""Relaxed recovery for the flight containing the legal airborne string hit."""
import math

import torch

from .contact_flight_recovery_v5 import ContactFlightRecoveryV5


FAILURE_FIELDS = ('support', 'roll_pitch', 'horizontal_speed', 'roll_pitch_speed',
                  'leg_joint_speed', 'body_ground', 'deadline')


def stability_failures(limits, *, support, roll_pitch, horizontal_speed, roll_pitch_speed,
                       leg_joint_speed, body_ground):
    return torch.stack((
        support < limits['minimum_support_feet'],
        roll_pitch.abs().amax(-1) > math.radians(limits['maximum_abs_roll_pitch_deg']),
        horizontal_speed > limits['maximum_horizontal_root_speed_m_s'],
        roll_pitch_speed > limits['maximum_roll_pitch_speed_rad_s'],
        leg_joint_speed > limits['maximum_abs_leg_joint_velocity_rad_s'],
        body_ground,
    ), dim=-1)


class ContactFlightRecoveryV6(ContactFlightRecoveryV5):
    def __init__(self, count, device, strict_protocol, relaxed_protocol):
        super().__init__(count, device, strict_protocol)
        self.limits = relaxed_protocol['recovery']
        self.relaxed_run = torch.zeros_like(self.stable_run)
        self.relaxed_longest = torch.zeros_like(self.stable_run)
        self.relaxed_success = torch.zeros_like(self.bound)
        self.reset_counts = torch.zeros((count, len(FAILURE_FIELDS)), dtype=torch.long, device=device)
        self.first_reset_mask = torch.zeros(count, dtype=torch.long, device=device)
        self.reason_bits = tuple(1 << i for i in range(len(FAILURE_FIELDS)))

    def relaxed_metric(self, jump, failures, dt):
        elapsed = jump.jump_clock - self.landing_time
        deadline = elapsed > self.limits['landing_deadline_s'] + 1e-6
        reasons = torch.cat((failures, deadline[:, None]), dim=-1)
        active = self.qualified_landing & ~self.relaxed_success
        eligible = active & ~reasons.any(-1)
        reset = active & (self.relaxed_run > 0) & ~eligible
        self.reset_counts.add_(reset[:, None] & reasons)
        mask = sum(reasons[:, i].long() * bit for i, bit in enumerate(self.reason_bits))
        self.first_reset_mask.copy_(torch.where(reset & (self.first_reset_mask == 0), mask, self.first_reset_mask))
        # The sample that established the landing cannot earn pre-landing time.
        run = torch.minimum(self.relaxed_run + dt, elapsed.clamp_min(0.))
        self.relaxed_run.copy_(torch.where(active, torch.where(eligible, run, 0.), self.relaxed_run))
        torch.maximum(self.relaxed_longest, self.relaxed_run, out=self.relaxed_longest)
        self.relaxed_success |= self.relaxed_run >= self.limits['continuous_stable_window_s'] - 1e-6

    def relaxed_record(self, index):
        first = int(self.first_reset_mask[index])
        return dict(success=bool(self.relaxed_success[index]), stable_s=float(self.relaxed_longest[index]),
                    reset_counts=dict(zip(FAILURE_FIELDS, self.reset_counts[index].tolist())),
                    first_reset_reasons=[field for field, bit in zip(FAILURE_FIELDS, self.reason_bits) if first & bit])
