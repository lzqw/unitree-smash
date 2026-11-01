"""Versioned reward-only training branch for the authorized success objective."""
from quadsmash.tasks.stand.full_racket_residual_env_v4 import FullRacketResidualTrainingEnvV4
from quadsmash.control.success_priority_alignment_reward_v1 import SuccessPriorityAlignmentRewardV1


class SuccessPriorityTrainingEnvV1(FullRacketResidualTrainingEnvV4):
    def __init__(self, *args, success_priority_reward, **kwargs):
        expected = {'release_speed_saturation_mps': 5.0,
                    'independent_racket_speed_shaping_enabled': True}
        if success_priority_reward != expected:
            raise ValueError('Declare the authorized success-priority reward settings')
        if kwargs.get('attack_speed_threshold_mps') != 5.0:
            raise ValueError('Task event speed must match the primary 5 m/s requirement')
        racket_speed_reward = kwargs.get('racket_speed_reward')
        if (not isinstance(racket_speed_reward, dict)
                or racket_speed_reward.get('target_mps') != 5.0):
            raise ValueError('Retain the independent racket-speed reward, saturated at5m/s')
        super().__init__(*args, **kwargs)
        previous = self.live_alignment_reward
        if previous is None:
            raise ValueError('Retain the measured contact and release shaping')
        if previous.job_event_counts.any() or previous.job_sums.any():
            raise ValueError('Install the new reward before any episode is stepped')
        self.live_alignment_reward = SuccessPriorityAlignmentRewardV1(
            self.num_envs, self.device, previous.settings,
            semiaxes_yz_m=previous.semiaxes,
            contact_plane_offset_m=previous.plane_offset,
            speed_saturation_mps=success_priority_reward['release_speed_saturation_mps'])
