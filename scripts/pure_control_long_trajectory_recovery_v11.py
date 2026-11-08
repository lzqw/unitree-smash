"""Predictive command selection over the full optimized recovery reference.

Keeps the original physical interface and causal release-state coverage test.
The reference duration is data, not a fixed 600 ms truncation. Native success
must still be established independently; this is not global sequence MPC.
"""
import numpy as np
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.pure_control_moving_foot_recovery_v7 import PureControlMovingFootRecoveryV7
from scripts.recovery_reference_library_v2 import RecoveryReferenceLibraryV2
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1, CourtCollisionPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from quadsmash.physics.court_v4 import load_court


class PureControlLongTrajectoryRecoveryV11(PureControlTrajectoryRecoveryV9):
    def __init__(self, interface, nominal, library):
        PureControlMovingFootRecoveryV7.__init__(self, interface, nominal)
        self.library = RecoveryReferenceLibraryV2(library, self.model)
        self.chosen = None
        self.selection_done = False
        self.selection_score = None
        self.reference_index = None
        self.path_geometry = FootSeparationPathConstraintV1(
            CourtCollisionPathConstraintV1(self.geometry.ground,
                AuthoredNetClearanceV1(self.geometry.ground, load_court())),
            self.geometry.spheres, margin_m=0.)

    def _reference_time(self):
        if self.chosen is None:
            return None, 0.
        steps = len(self.chosen['trajectory'])
        duration = .01 * steps
        if self.elapsed >= duration - 1e-9:
            return None, 0.
        index = int(np.clip(np.floor(self.elapsed / .01 + 1e-6), 0, steps - 1))
        return index, float(min(1., (duration - self.elapsed) / .10))

    def _candidate(self, state, target, k):
        predicted = state
        cost = 0.
        horizon = min(2, len(self.chosen['trajectory']) - k)
        for offset in range(horizon):
            command = target if offset == 0 else np.asarray(self.chosen['trajectory'][k + offset]['target'])
            try:
                result = predict_self_contact_spheres(self.model, self.interface, predicted, command,
                    self.geometry.spheres, geometry=self.path_geometry, substeps=16)
            except (ValueError, np.linalg.LinAlgError) as error:
                return dict(accepted=False, error=repr(error), horizon_step=offset)
            violations = {n: v for n, v in result['diagnostics'].items() if
                (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
                (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                       'motor_torque_violation_nm'] and v > 3e-4)}
            if violations:
                return dict(accepted=False, violations=violations, horizon_step=offset)
            predicted = result['state']
            predicted.previous_command_target = command.copy()
            cost += self._tracking_cost(predicted, self.library.state(self.chosen, k + offset))
        return dict(accepted=True, cost=cost, prediction_intervals=horizon, native_certified=False)

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_long_trajectory_recovery_v11',
            model_reference_duration_s=.01 * len(self.chosen['trajectory']) if self.chosen is not None else None)
        return target, diagnostic
