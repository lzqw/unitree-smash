#!/usr/bin/env python3
"""Soft arm-command tracking variant of the support-margin landing MPC."""
from scripts.command_tracking_support_qp_v1 import CommandTrackingSupportQP
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2, LandingMPCSettingsV2


class LandingSupportMPCV4(LandingSupportMPCV2):
    """V2 support-margin MPC with a continuous COM reference and soft arm cost."""

    def __init__(self, interface, *, qp_settings=None,
                 settings=LandingMPCSettingsV2(), arm_tracking_weight=None):
        super().__init__(interface, qp_settings=qp_settings, settings=settings)
        self.qp = CommandTrackingSupportQP(interface, settings=self.qp.settings,
            continuous_reference=True, arm_tracking_weight=arm_tracking_weight,
            reference_frequency_rad_s=10.)

    def reset(self):
        if hasattr(self.qp, 'reset'):
            self.qp.reset()

    def contract(self):
        result = super().contract()
        result['schema'] = 'landing-support-mpc-v4-soft-arm'
        result['formulation'] = 'fixed-contact-sequential-continuous-reference-soft-arm-QP'
        result['continuous_reference'] = True
        result['soft_arm_command_tracking'] = True
        return result


__all__ = ['LandingSupportMPCV4']
