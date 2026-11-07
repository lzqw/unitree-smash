#!/usr/bin/env python3
"""Continuous-reference support MPC for post-touchdown recovery.

V3 keeps the support-margin gates from v2 but replaces the per-tick COM target
reset with ``ContinuousSupportQP``.  Its critically damped reference retains
position/velocity across observed contact changes and preserves the supplied
arm position command, which is important for the Go2+PiPER load after a hit.
"""
import numpy as np

from scripts.admitted_arm_support_qp_v1 import AdmittedArmSupportQP
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2, LandingMPCSettingsV2


class LandingSupportMPCV3(LandingSupportMPCV2):
    """Support-margin MPC using a continuous whole-body reference."""

    def __init__(self, interface, *, qp_settings=None,
                 settings=LandingMPCSettingsV2(), preserve_arm_command=True,
                 load_aware_support=False, load_force_floor_n=2.):
        super().__init__(interface, qp_settings=qp_settings, settings=settings)
        if type(preserve_arm_command) is not bool:
            raise ValueError('preserve_arm_command must be boolean')
        self.preserve_arm_command = preserve_arm_command
        self.qp = AdmittedArmSupportQP(interface, settings=self.qp.settings,
            continuous_reference=True, preserve_arm_command=preserve_arm_command,
            reference_frequency_rad_s=10., load_aware=load_aware_support,
            load_force_floor_n=load_force_floor_n)
        self._live_reference = {}
        self._prediction_reference = {}
        self._observed_tick = {}

    def _posture_reference(self, state, joint_reference):
        reference = super()._posture_reference(state, joint_reference)
        if self.preserve_arm_command:
            # Both live-reference observation and speculative QPs must use
            # the arm goal that the final position interface can actually send.
            reference[12:] = self.admit_command_target(state, reference)[12:]
        return reference

    def reset(self, lane=None):
        """Reset a single episode without clearing other environments."""
        if lane is None:
            self._live_reference.clear()
            self._prediction_reference.clear()
            self._observed_tick.clear()
        else:
            for memory in (self._live_reference, self._prediction_reference,
                           self._observed_tick):
                memory.pop(int(lane), None)
        self.qp.reset()

    def observe(self, state, joint_reference, *, lane=0, control_tick):
        """Advance live control memory once per actual tick, including holds.

        A reference branch for planning begins before that tick's advance.
        Failed solves and speculative contact modes never commit their memory.
        """
        lane, control_tick = int(lane), int(control_tick)
        previous = self._observed_tick.get(lane)
        if previous == control_tick:
            return
        if previous is not None and control_tick < previous:
            raise ValueError('reset the lane before its control clock restarts')
        if lane in self._live_reference:
            self.qp.restore_reference_state(self._live_reference[lane])
        else:
            self.qp.reset()
        before = self.qp.reference_state()
        try:
            snapshot = self.qp.model.snapshot(state.q, state.v)
            self.qp.reference(state, snapshot, self._posture_reference(state, joint_reference))
            self._live_reference[lane] = self.qp.reference_state()
            self._prediction_reference[lane] = before
            self._observed_tick[lane] = control_tick
        except Exception:
            self.qp.restore_reference_state(before)
            raise

    def solve(self, state, joint_reference, *, force=False, lane=0, control_tick=None):
        lane = int(lane)
        if control_tick is None:
            control_tick = self._observed_tick.get(lane, -1) + 1
        self.observe(state, joint_reference, lane=lane, control_tick=control_tick)
        self.qp.restore_reference_state(self._prediction_reference[lane])
        try:
            return super().solve(state, joint_reference, force=force)
        finally:
            self.qp.restore_reference_state(self._live_reference[lane])

    def contract(self):
        result = super().contract()
        result['schema'] = 'landing-support-mpc-v3-continuous-reference'
        result['formulation'] = 'fixed-contact-sequential-continuous-reference-QP'
        result['continuous_reference'] = True
        result['preserve_arm_position_command'] = self.preserve_arm_command
        result['arm_command_semantics'] = 'preserve-after-native-target-admission'
        result['load_aware_support'] = self.qp.load_aware
        result['load_force_floor_n'] = self.qp.load_force_floor_n
        result['reference_memory'] = 'per-lane-live-tick-with-speculative-rollback'
        return result


__all__ = ['LandingSupportMPCV3']
