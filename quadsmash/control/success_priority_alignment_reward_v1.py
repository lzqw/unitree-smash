"""Keep measured contact shaping; stop paying for extra speed above task speed."""
import math

import torch

from quadsmash.control.measured_string_alignment_reward_v4 import (
    MeasuredStringAlignmentRewardV4,
    alignment_settings,
)

REWARD_VERSION = 'native-full-chain-success-priority-speed5-v1'


class SuccessPriorityAlignmentRewardV1(MeasuredStringAlignmentRewardV4):
    def __init__(self, *args, speed_saturation_mps, **kwargs):
        value = float(speed_saturation_mps)
        if not math.isfinite(value) or value != 5.0:
            raise ValueError('This version implements the user-authorized 5 m/s objective')
        self.speed_saturation_mps = value
        super().__init__(*args, **kwargs)

    def step(self, **measured):
        # Capture the actual first-release mask before the parent updates its
        # event history. Geometry, potential, frame penalties and reset rules
        # are inherited unchanged from the frozen implementation.
        first_release = measured['released'] & (~self.seen_release)
        super().step(**measured)
        outgoing = measured['outgoing_velocity_mps']
        finite = torch.isfinite(outgoing).all(1)
        velocity = torch.where(finite[:, None], outgoing, torch.zeros_like(outgoing))
        eligible = (first_release & measured['legal_air'] & (~measured['faulted'])
                    & finite & (velocity[:, 0] > 0.))
        speed = (velocity.norm(dim=1) / self.speed_saturation_mps).clamp(0., 1.)
        downward = (-velocity[:, 2]).clamp(0., 1.)
        progress = torch.where(eligible,
            self.settings['attack_progress_weight'] * speed * downward,
            torch.zeros_like(speed))
        # Replace the legacy speed term in BOTH the paid reward and telemetry.
        # The discarded 8 m/s calculation never reaches the returned reward.
        change = progress - self.last_components[:, 1]
        self.job_sums[1].add_(change.double().sum())
        self.last_components[:, 1].copy_(progress)
        return self.last_components.sum(1)

    def statistics(self):
        result = super().statistics()
        result.update(reward_version=REWARD_VERSION,
                      release_speed_saturation_mps=self.speed_saturation_mps,
                      extra_speed_above_task_threshold_rewarded=False)
        return result
