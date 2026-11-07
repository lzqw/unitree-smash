"""Experimental support rollout sharing an implicit command QP and predictor.

Sequential QP rollout, not joint trajectory optimization. Predicted touchdown
is the existing event approximation. Native effectiveness is unverified.
"""
import numpy as np
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.implicit_command_support_qp_v1 import ImplicitCommandSupportQP
from scripts.implicit_support_prediction_v1 import predict_implicit_support


class ImplicitPredictionRejected(ValueError):
    pass


class LandingImplicitCommandMPCV1(LandingCommandHistoryMPCV1):
    def __init__(self, interface, *, arm_target_rate_rad_s=5., **kwargs):
        super().__init__(interface, arm_target_rate_rad_s=arm_target_rate_rad_s, **kwargs)
        old = self.qp
        self.qp = ImplicitCommandSupportQP(interface, old.settings,
            arm_target_rate_rad_s=arm_target_rate_rad_s,
            continuous_reference=old.continuous_reference,
            preserve_arm_command=old.preserve_arm_command,
            reference_frequency_rad_s=old.reference_frequency_rad_s,
            load_aware=old.load_aware, load_force_floor_n=old.load_force_floor_n)
        self.prediction_checks = []

    def _integrate(self, state, acceleration, plan=None):
        if plan is None:
            raise ImplicitPredictionRejected('Full actual command required by implicit predictor')
        target = self.admit_command_target(state, plan['joint_position_target'])
        result = predict_implicit_support(self.qp.model, self.qp.interface, state, target,
            duration_s=self.settings.control_dt_s, substeps=1,
            contact_velocity_decay_s=self.qp.settings.contact_velocity_decay_s)
        diagnostics = result['diagnostics'].copy()
        diagnostics['qp_prediction_acceleration_disagreement'] = float(np.max(abs(
            (result['state'].v-state.v)/self.settings.control_dt_s-np.asarray(acceleration))))
        self.prediction_checks.append(diagnostics)
        quantities = ['normal_force_violation_n', 'friction_violation_n',
            'joint_velocity_violation_rad_s', 'joint_position_violation_rad',
            'motor_torque_violation_nm', 'dynamics_contact_residual_max']
        if max(diagnostics[k] for k in quantities) > self.qp.settings.residual_tolerance:
            raise ImplicitPredictionRejected('Admitted implicit prediction violates original physical constraints')
        if diagnostics['qp_prediction_acceleration_disagreement'] > 1e-2:
            raise ImplicitPredictionRejected('QP/predictor response differs after target admission')
        following = result['state']
        following.previous_command_target = target.copy()
        return following

    def solve(self, state, joint_reference, **kwargs):
        self.prediction_checks = []
        try:
            result = super().solve(state, joint_reference, **kwargs)
        except ImplicitPredictionRejected as error:
            # Do not expose an unchecked partial horizon as an executable plan.
            result = dict(accepted=False, intervened=True, horizon=[],
                rejection=str(error), constraint_failure=True, solver_failure=False,
                risk=self.risk(state), native_execution=False, mpc_contract=self.contract())
        result['implicit_prediction_checks'] = list(self.prediction_checks)
        return result

    def contract(self):
        result = super().contract()
        result.update(schema='landing-implicit-command-mpc-v1',
            formulation='sequential-QP-with-matching-backward-Euler-PD',
            drive_branch='unsaturated-with-original-effort-constraints',
            nonlinear_contact_and_impact_model_complete=False,
            native_recovery_validated=False)
        return result
