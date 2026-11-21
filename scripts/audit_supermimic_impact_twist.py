"""First-impact velocity diagnostic. Keep the calibrated detection, normal, lever arm and slip law fixed."""
import torch

from evaluate_supermimic_contact_rollout import racket_motion_at_fraction
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_imagination_adapter_v1 import matrix_quaternion


class EndpointTwistPrior(BallImpulsePriorV1):
    def contact_state_at_time(self, ball_position, ball_velocity, initial_rotation,
                              initial_center, initial_velocity, initial_angular_velocity,
                              impact_time, *, endpoint_rotation=None, endpoint_center=None,
                              endpoint_velocity=None, endpoint_angular_velocity=None, **local_path):
        result = super().contact_state_at_time(
            ball_position, ball_velocity, initial_rotation, initial_center, initial_velocity,
            initial_angular_velocity, impact_time, endpoint_rotation=endpoint_rotation,
            endpoint_center=endpoint_center, endpoint_velocity=endpoint_velocity,
            endpoint_angular_velocity=endpoint_angular_velocity, **local_path)
        if endpoint_rotation is None:
            return result
        start = (initial_center, matrix_quaternion(initial_rotation), initial_velocity, initial_angular_velocity)
        end = (endpoint_center, matrix_quaternion(endpoint_rotation), endpoint_velocity, endpoint_angular_velocity)
        _, _, linear, angular = racket_motion_at_fraction(
            start, end, impact_time[:, None]/self.dt, self.dt, rotation_hermite=True)
        lever = result['rotation'] @ (result['local'] - self.preview.radius*result['normal_local'])[..., None]
        result['surface_velocity'] = linear + torch.linalg.cross(angular, lever.squeeze(-1))
        return result
