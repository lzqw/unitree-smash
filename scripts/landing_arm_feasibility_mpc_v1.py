"""Retain the arm command when feasible; permit bounded recovery otherwise.

Only the equality to the requested arm command may be removed, and only after
the identical interval QP is diagnosed primal-infeasible. All original physical
and command-history bounds remain. This is not yet a native-validated teacher.
"""
import numpy as np
from scripts.landing_active_set_interval_mpc_v1 import (
    ActiveSetIntervalCommandSupportQP, LandingActiveSetIntervalMPCV1)


class ArmFeasibilityIntervalQP(ActiveSetIntervalCommandSupportQP):
    def solve(self, state, joint_reference, *, static_equilibrium=False):
        before = self.reference_state()
        fixed = super().solve(state, joint_reference, static_equilibrium=static_equilibrium)
        fixed['arm_feasibility_fallback_attempted'] = False
        if (not self.preserve_arm_command or fixed['accepted'] or
                fixed.get('solver_status') != 'primal infeasible'):
            return fixed
        after = self.reference_state()
        self.restore_reference_state(before)
        self.preserve_arm_command = False
        try:
            free = super().solve(state, joint_reference, static_equilibrium=static_equilibrium)
        finally:
            self.preserve_arm_command = True
            # A second solve must never advance the live/speculative reference
            # twice. Both proposals use the same single reference transition.
            self.restore_reference_state(after)
        event = dict(fixed_status=fixed['solver_status'], free_status=free.get('solver_status'),
                     free_accepted=bool(free['accepted']))
        if free['accepted']:
            event['arm_change_from_preserved_reference_max_rad'] = float(np.max(abs(
                free['joint_position_target'][12:] - np.asarray(fixed['admitted_reference'])[12:])))
        if hasattr(self, 'fallback_events'):
            self.fallback_events.append(event)
        result = free if free['accepted'] else fixed
        result['arm_feasibility_fallback_attempted'] = True
        result['arm_feasibility_fallback'] = event
        result['physical_constraints_relaxed'] = False
        return result


class LandingArmFeasibilityMPCV1(LandingActiveSetIntervalMPCV1):
    qp_type = ArmFeasibilityIntervalQP

    def solve(self, *args, **kwargs):
        self.qp.fallback_events = []
        result = super().solve(*args, **kwargs)
        result['arm_feasibility_fallbacks'] = list(self.qp.fallback_events)
        return result

    def contract(self):
        result = super().contract()
        result.update(schema='landing-arm-feasibility-interval-mpc-v1',
            arm_command_semantics='preserve-when-feasible; bounded-free-arm-only-after-primal-infeasible',
            preserve_arm_position_command='conditional', physical_constraints_relaxed=False,
            arm_reference_is_still_soft_posture_objective=True, native_recovery_validated=False)
        return result
