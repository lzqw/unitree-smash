"""Contact-aware support with continuous COM references and prescribed arm PD.

CPU candidate, not a validated landing controller. Full loaded dynamics and all
original limits come from the existing reduced QP. The optional arm prescription
retains the supplied six arm position commands through the ORIGINAL PD; it does
not freeze arm joints, remove their dynamics, or add external forces. The stance
reference keeps position/velocity continuous across observed contact changes.
A new controller instance is required per environment, with reset per episode.
"""
import numpy as np

from scripts.reduced_whole_body_support_qp_v1 import ReducedWholeBodySupportQP
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


class PrescribedArmPDInterface(PositionPDInterface):
    """Same actuator model, with an explicitly supplied arm position command."""
    def __init__(self, source):
        super().__init__(source.limits, source.kp, source.kd, source.effort,
                         source.velocity, source.friction)
        self.arm_reference = None

    def torque_bounds(self, q, v, control_dt_s=None):
        low, high = super().torque_bounds(q, v, control_dt_s)
        if self.arm_reference is not None:
            actual = np.clip(self.kp[12:]*(self.arm_reference-q[12:])-
                            self.kd[12:]*v[12:], -self.effort[12:], self.effort[12:])
            low[12:] = actual
            high[12:] = actual
        return low, high

    def target_from_torque(self, torque, q, v):
        target, _ = super().target_from_torque(torque, q, v)
        if self.arm_reference is not None:
            # Saturated PD can admit several position targets. Preserve the
            # caller's exact target rather than another equivalent instant.
            target[12:] = self.arm_reference
        achieved = np.clip(self.kp*(target-q)-self.kd*v, -self.effort, self.effort)
        return target, achieved


class ContinuousSupportQP(ReducedWholeBodySupportQP):
    def __init__(self, interface, settings=SupportSettings(), *,
                 continuous_reference=True, preserve_arm_command=True,
                 reference_frequency_rad_s=10.):
        if (type(continuous_reference) is not bool or type(preserve_arm_command) is not bool
                or not np.isfinite(reference_frequency_rad_s) or reference_frequency_rad_s <= 0):
            raise ValueError('Explicit Boolean branches and positive reference frequency required')
        super().__init__(PrescribedArmPDInterface(interface), settings)
        self.continuous_reference = continuous_reference
        self.preserve_arm_command = preserve_arm_command
        self.reference_frequency_rad_s = float(reference_frequency_rad_s)
        self.reset()

    def reset(self):
        self.reference_position = None
        self.reference_velocity = None
        self.support_anchors = np.zeros((4, 3))
        self.previous_contacts = np.zeros(4, dtype=bool)
        self.interface.arm_reference = None
        self._static_fixture = False

    def reference(self, state, snapshot, joint_reference):
        raw = super().reference(state, snapshot, joint_reference)
        if not self.continuous_reference or self._static_fixture:
            return raw
        if not state.contacts.any():
            self.reference_position = None
            self.reference_velocity = None
            self.previous_contacts[:] = False
            return raw
        feet_ids = [self.model.names.index(name) for name in self.model.feet]
        feet = snapshot['body_positions_world'][feet_ids]
        newly = state.contacts & ~self.previous_contacts
        self.support_anchors[newly] = feet[newly]
        self.previous_contacts = state.contacts.copy()
        # Preserve the full loaded-posture support height. Contact anchors stop
        # a planted foot's small measured drift from moving its support target.
        goal = raw['com_position'] + (self.support_anchors[state.contacts]-
                                      feet[state.contacts]).mean(0)
        if self.reference_position is None:
            self.reference_position = snapshot['com_world'].copy()
            self.reference_velocity = snapshot['com_velocity_world'].copy()
        position, velocity = self.reference_position.copy(), self.reference_velocity.copy()
        omega, dt = self.reference_frequency_rad_s, self.settings.control_dt_s
        acceleration = omega**2*(goal-position)-2.*omega*velocity
        cfg = self.settings
        raw.update(com_position=position, com_velocity=velocity,
            com_acceleration=acceleration+cfg.com_kp_s2*(position-snapshot['com_world'])-
                cfg.com_kd_s*(snapshot['com_velocity_world']-velocity),
            com_goal_before_filter=goal.copy(), com_reference_acceleration=acceleration,
            active_support_anchors=self.support_anchors[state.contacts].copy())
        # Exact critical-damping transition for a held goal over one native
        # command interval. A changed goal affects acceleration, not a reset of
        # position/velocity. No synthetic ground force is introduced.
        displacement = position-goal
        linear = velocity+omega*displacement
        decay = np.exp(-omega*dt)
        self.reference_position = goal+(displacement+linear*dt)*decay
        self.reference_velocity = (velocity-omega*linear*dt)*decay
        return raw

    def solve(self, state, joint_reference, *, static_equilibrium=False):
        target = np.asarray(joint_reference, dtype=np.float64)
        if target.shape != (18,) or not np.isfinite(target).all():
            raise ValueError('Require a finite full18-joint command reference')
        self.interface.arm_reference = target[12:].copy() if self.preserve_arm_command else None
        self._static_fixture = bool(static_equilibrium)
        try:
            plan = super().solve(state, target, static_equilibrium=static_equilibrium)
        finally:
            self._static_fixture = False
        plan.update(continuous_support_reference=self.continuous_reference,
            preserve_supplied_arm_position_command=self.preserve_arm_command,
            reference_frequency_rad_s=self.reference_frequency_rad_s,
            full_hold_pd_realization_verified=False)
        if plan['accepted'] and self.preserve_arm_command:
            error = float(np.max(abs(plan['joint_position_target'][12:]-target[12:])))
            if error != 0.:
                raise ValueError('Solver replaced the supplied arm position commands')
            plan['prescribed_arm_command_error_rad'] = error
        return plan
