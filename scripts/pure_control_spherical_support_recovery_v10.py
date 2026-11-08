"""Trajectory feedback with support forces applied at the authored foot surface."""
import numpy as np
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.pure_control_interval_spherical_support_v4 import PureControlIntervalSphericalSupportV4


class PureControlSphericalSupportRecoveryV10(PureControlTrajectoryRecoveryV9):
    def solve(self,state,joint_reference,**kwargs):
        k,weight=self._reference_time();self.reference_index=k
        if k is not None:
            target=np.asarray(self.chosen['trajectory'][k]['target'])
            joint_reference=weight*target+(1-weight)*joint_reference
        plan=PureControlIntervalSphericalSupportV4.solve(self,state,joint_reference,**kwargs)
        self.last_hip_plan=plan;return plan

    def command(self,*args,**kwargs):
        target,diagnostic=super().command(*args,**kwargs)
        diagnostic.update(controller='pure_control_spherical_support_recovery_v10',
            support_point='authored_sphere_bottom',independent_validation_uses_same_support_point=True)
        return target,diagnostic
