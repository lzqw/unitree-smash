#!/usr/bin/env python3
"""Continuous-reference support MPC for post-touchdown recovery.

V3 keeps the support-margin gates from v2 but replaces the per-tick COM target
reset with ``ContinuousSupportQP``.  Its critically damped reference retains
position/velocity across observed contact changes and preserves the supplied
arm position command, which is important for the Go2+PiPER load after a hit.
"""
import numpy as np

from scripts.continuous_support_qp_v1 import ContinuousSupportQP
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
        self.qp = ContinuousSupportQP(interface, settings=self.qp.settings,
            continuous_reference=True, preserve_arm_command=preserve_arm_command,
            reference_frequency_rad_s=10., load_aware=load_aware_support,
            load_force_floor_n=load_force_floor_n)

    def reset(self):
        if hasattr(self.qp, 'reset'):
            self.qp.reset()

    def contract(self):
        result = super().contract()
        result['schema'] = 'landing-support-mpc-v3-continuous-reference'
        result['formulation'] = 'fixed-contact-sequential-continuous-reference-QP'
        result['continuous_reference'] = True
        result['preserve_arm_position_command'] = self.preserve_arm_command
        result['load_aware_support'] = self.qp.load_aware
        result['load_force_floor_n'] = self.qp.load_force_floor_n
        return result


__all__ = ['LandingSupportMPCV3']
