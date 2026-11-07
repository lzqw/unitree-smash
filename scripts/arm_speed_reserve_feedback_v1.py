"""Recovery-only arm reference speed reserve; hardware limits remain unchanged."""
from types import SimpleNamespace
import numpy as np
from scripts.arm_velocity_recovery_feedback_v1 import ArmVelocityRecoveryFeedbackV1


class ArmSpeedReserveFeedbackV1(ArmVelocityRecoveryFeedbackV1):
    def __init__(self, policy, reference, interface, *, damping_nm_per_rad_s=4., max_offset_rad=.02):
        # Only the arm-feedback reference is changed. The leg trajectory and
        # underlying policy retain their original reference and parameters.
        velocity = np.asarray(reference.v_reference).copy()
        velocity[:, 12:] = np.clip(velocity[:, 12:], -4.5, 4.5)
        super().__init__(policy, SimpleNamespace(v_reference=velocity), interface,
                         damping_nm_per_rad_s=damping_nm_per_rad_s, max_offset_rad=max_offset_rad)
