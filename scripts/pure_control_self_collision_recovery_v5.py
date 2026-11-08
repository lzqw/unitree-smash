"""Full-authority recovery with explicit hip/mount collision avoidance.

This experiment prevents the interval planner from relying on motion through
the mounting plate. Native collision response and hardware remain unchanged.
It is one-interval predictive WBC, not sequence MPC or a validated teacher.
"""
from scripts.pure_control_court_recovery_v4 import PureControlCourtRecoveryV4
from scripts.pure_control_interval_self_collision_v2 import PureControlIntervalSelfCollisionV2
from scripts.hip_mount_planning_constraints_v1 import HipMountPlanningConstraintsV1


class PureControlSelfCollisionRecoveryV5(PureControlCourtRecoveryV4):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.hip_collision=HipMountPlanningConstraintsV1(self.model)

    def solve(self,*args,**kwargs):
        plan=PureControlIntervalSelfCollisionV2.solve(self,*args,**kwargs)
        self.last_hip_plan=plan
        return plan

    def command(self,state,phase,time_s):
        target,diagnostic=super().command(state,phase,time_s)
        diagnostic.update(controller='pure_control_self_collision_recovery_v5',
            hip_mount_state_gaps_rad=self.hip_collision.gaps(state.joint_position).tolist(),
            hip_mount_avoidance_in_planning=True,
            predicted_minimum_hip_mount_gap_rad=self.last_hip_plan.get('minimum_hip_mount_gap_rad'),
            native_self_collision_response_unchanged=True,
            fallback_collision_avoidance_certified=False)
        return target,diagnostic
