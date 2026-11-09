"""Experimental clearance cost on contact-dynamic predicted states.

Replaces V25's constant-velocity surface extrapolation with five actual
interface prediction intervals. Candidate first command plus nominal tail is
still a restricted continuation assumption, not a whole-policy certificate.
"""
import numpy as np
from scripts.pure_control_clearance_reference_v25 import PureControlClearanceReferenceV25, surface_margin_penalty
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_settled_stance_model_v2 import ForceRecorder


class PureControlDynamicClearanceV26(PureControlClearanceReferenceV25):
    def _tracking_cost(self, state, desired):
        cost = PureControlStanceFootReferenceV15._tracking_cost(self, state, desired)
        if self.clearance_weight:
            penalty, _ = surface_margin_penalty(authored_clearance_state(self.geometry, state), approach_s=0.)
            cost += self.clearance_weight * penalty
        return cost

    def _candidate(self, state, target, k):
        if not self.clearance_weight:
            return PureControlStanceFootReferenceV15._candidate(self, state, target, k)
        predicted=state;cost=0.;horizon=min(5,len(self.chosen['trajectory'])-k)
        for offset in range(horizon):
            command=target if offset==0 else np.asarray(self.chosen['trajectory'][k+offset]['target'])
            recorder=ForceRecorder(self.model)
            try:
                result=predict_self_contact_spheres(self.model,self.interface,predicted,command,
                    self.geometry.spheres,geometry=self.path_geometry,substeps=16,collision_response=recorder)
            except (ValueError,np.linalg.LinAlgError) as error:
                return dict(accepted=False,error=repr(error),horizon_step=offset)
            bad={n:v for n,v in result['diagnostics'].items() if
                 (n=='nonfoot_ground_violation_m' and v>1e-6) or
                 (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
            if bad:return dict(accepted=False,violations=bad,horizon_step=offset)
            predicted=result['state'];predicted.previous_command_target=command.copy()
            predicted.measured_force_world=recorder.last_force.copy()
            predicted.contacts=np.where(predicted.contacts,recorder.last_force[:,2]>.5,recorder.last_force[:,2]>1.)
            cost+=self._tracking_cost(predicted,self.library.state(self.chosen,k+offset))
        return dict(accepted=True,cost=cost,prediction_intervals=horizon,
                    dynamic_ground_margin=True,native_certified=False)

    def command(self,state,phase,time_s):
        target,diagnostic=super().command(state,phase,time_s)
        penalty,surfaces=surface_margin_penalty(authored_clearance_state(self.geometry,state),approach_s=0.)
        diagnostic.update(controller='pure_control_dynamic_clearance_v26',
                          surface_margin_soft_cost=penalty,surface_approach_time_s=0.,
                          dynamic_prediction_time_s=.05,
                          nearest_approach_surface=min(surfaces,key=lambda row:row['gap_m']),
                          extrapolated_surface_rate_not_in_objective=True)
        return target,diagnostic
