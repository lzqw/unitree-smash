"""Experimental position/velocity envelope inside the loaded command QP.

At every implicit substep: -k(q-q_min) <= v <= k(q_max-q).
This anticipates joint limits without relaxing any physical bound. It is a
conservative engineering envelope, NOT an acceleration/viability certificate.
States outside the envelope can be rejected even when physically recoverable.
The live interval implementation is left unchanged while its evaluation runs.
"""
import numpy as np
from scipy import sparse

from scripts.condensed_implicit_drive_v1 import condense_implicit_drive
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.landing_arm_feasibility_mpc_v1 import (
    ArmFeasibilityIntervalQP, LandingArmFeasibilityMPCV1)


def joint_damper_constraints(stages, selection, limits, gain_s_inv):
    """Return physical held-command inequalities, not normalized action bounds."""
    limits = np.asarray(limits, dtype=float)
    if (not np.isfinite(gain_s_inv) or gain_s_inv <= 0 or not stages or
            limits.shape != (selection.shape[1], 2) or
            not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError('Finite positive damper gain and original bounded joints required')
    matrices, lows, highs = [], [], []
    for stage in stages:
        matrix = selection.T @ stage['velocity_map'] + gain_s_inv * stage['joint_position_map']
        bias = selection.T @ stage['velocity_bias'] + gain_s_inv * stage['joint_position_bias']
        matrices.append(matrix)
        lows.append(gain_s_inv * limits[:, 0] - bias)
        highs.append(gain_s_inv * limits[:, 1] - bias)
    return np.vstack(matrices), np.concatenate(lows), np.concatenate(highs)


class JointDamperIntervalQP(ArmFeasibilityIntervalQP):
    def __init__(self, *args, damper_gain_s_inv=20., **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(damper_gain_s_inv) or damper_gain_s_inv <= 0:
            raise ValueError('Positive finite gain required')
        self.damper_gain_s_inv = float(damper_gain_s_inv)
        self._damper_context = None

    def _make_solver(self):
        # A local solver adapter avoids mutating the running/base QP class.
        if self._damper_context is None:
            raise RuntimeError('A loaded state must precede solver construction')
        matrix, low, high, center, scale = self._damper_context

        class AugmentedSolver(DenseActiveSetQP):
            def setup(self, **kwargs):
                scaled = matrix * scale[None, :]
                row_scale = 1. / np.maximum(np.max(abs(scaled), axis=1), 1e-12)
                kwargs['A'] = sparse.vstack((kwargs['A'], sparse.csc_matrix(scaled * row_scale[:, None])), format='csc')
                kwargs['l'] = np.r_[kwargs['l'], (low - matrix @ center) * row_scale]
                kwargs['u'] = np.r_[kwargs['u'], (high - matrix @ center) * row_scale]
                return super().setup(**kwargs)

        return AugmentedSolver()

    def solve(self, state, joint_reference, *, static_equilibrium=False):
        interface, cfg = self.interface, self.settings
        interface.bind_previous_command(state.previous_command_target)
        bounds = interface.admitted_target_bounds(state.joint_position, cfg.control_dt_s)
        center = np.clip(np.asarray(joint_reference, dtype=float), *bounds)
        snap = self.model.snapshot(state.q, state.v)
        feet = np.flatnonzero(state.contacts)
        jac = snap['foot_jacobian'][feet].reshape(-1, 24)
        jdot = (snap['foot_jacobian_dot'][feet] @ state.v).reshape(-1)
        selection = np.zeros((24, 18))
        selection[self.model.v_indices, np.arange(18)] = 1.
        stages = condense_implicit_drive(snap['mass_matrix'], snap['nonlinear'], selection,
            interface.kp, interface.kd, jac, jdot, state.v, state.joint_position,
            duration_s=cfg.control_dt_s, substeps=self.substeps,
            contact_velocity_decay_s=cfg.contact_velocity_decay_s)
        matrix, low, high = joint_damper_constraints(
            stages, selection, interface.limits, self.damper_gain_s_inv)
        self._damper_context = (matrix, low, high, center, .5 * np.diff(interface.limits, axis=1).ravel())
        try:
            result = super().solve(state, joint_reference, static_equilibrium=static_equilibrium)
        finally:
            self._damper_context = None
        result['joint_damper_gain_s_inv'] = self.damper_gain_s_inv
        result['joint_damper_constraint_rows'] = len(low)
        result['constraint_rows'] += len(low)
        result['joint_damper_is_viability_certificate'] = False
        if 'joint_position_target' in result:
            response = matrix @ result['joint_position_target']
            error = float(max(0., np.max(low - response), np.max(response - high)))
            result['residuals']['joint_damper_rad_s'] = error
            if error > cfg.residual_tolerance:
                result['accepted'] = False
                result['rejection'] = 'Independent joint damper check failed'
        return result


class LandingJointDamperMPCV1(LandingArmFeasibilityMPCV1):
    qp_type = JointDamperIntervalQP

    def contract(self):
        result = super().contract()
        result.update(schema='landing-joint-damper-mpc-v1',
            joint_damper_gain_s_inv=self.qp.damper_gain_s_inv,
            damper_in_all_implicit_substeps=True, physical_constraints_relaxed=False,
            viability_or_native_recovery_certified=False)
        return result
