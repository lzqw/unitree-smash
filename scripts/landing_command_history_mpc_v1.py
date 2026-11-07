"""Sustained support controller with explicit executed-command history."""
import numpy as np
from scripts.landing_support_mpc_v3 import LandingSupportMPCV3
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.command_history_support_qp_v1 import CommandHistorySupportQP


class LandingCommandHistoryMPCV1(LandingSupportMPCV3):
    def __init__(self,interface,*,arm_target_rate_rad_s=5.,**kwargs):
        if kwargs.get("qp_settings") is None:
            kwargs["qp_settings"]=SupportSettings()
        super().__init__(interface,**kwargs)
        if self.settings.command_rate_scale!=1.:
            raise ValueError('Command-history controller uses declared command slew, not measured-position rate scaling')
        old=self.qp
        self.qp=CommandHistorySupportQP(interface,old.settings,
            arm_target_rate_rad_s=arm_target_rate_rad_s,
            continuous_reference=old.continuous_reference,
            preserve_arm_command=old.preserve_arm_command,
            reference_frequency_rad_s=old.reference_frequency_rad_s,
            load_aware=old.load_aware,load_force_floor_n=old.load_force_floor_n)

    def admit_command_target(self,state,target):
        target=np.asarray(target,dtype=np.float64)
        if target.shape!=(18,) or not np.isfinite(target).all():
            raise ValueError('Require finite 18D target')
        self.qp.interface.bind_previous_command(state.previous_command_target)
        low,high=self.qp.interface.admitted_target_bounds(state.joint_position,self.settings.control_dt_s)
        return np.clip(target,low,high)

    def _integrate(self,state,acceleration,plan=None):
        following=super()._integrate(state,acceleration,plan)
        if plan is not None:
            following.previous_command_target=self.admit_command_target(state,plan['joint_position_target']).copy()
        return following

    def contract(self):
        result=super().contract()
        result.update(schema='landing-command-history-mpc-v1',
            command_interface='leg-position-range;arm-slew-from-last-final-command',
            arm_target_rate_rad_s=self.qp.interface.arm_rate,
            physical_velocity_constraints_unchanged=True)
        return result
