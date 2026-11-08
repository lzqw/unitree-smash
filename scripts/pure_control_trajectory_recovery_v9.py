"""Select executable reference/feedback commands using coupled prediction.

The WBC request is a candidate, not an unconditional replacement of a feasible
planned command. Two actual-interface intervals evaluate tracking error and
physical path constraints. This is restricted candidate predictive control,
not a globally optimized MPC or a native safety certificate.
"""
import numpy as np
import pinocchio as pin
from scripts.pure_control_trajectory_recovery_v8 import PureControlTrajectoryRecoveryV8
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


class PureControlTrajectoryRecoveryV9(PureControlTrajectoryRecoveryV8):
    def _tracking_cost(self,state,desired):
        dq=pin.difference(self.model.model,desired.q,state.q)
        # Compare floating-base velocities in world coordinates.
        dv=np.r_[state.root_rotation_world@state.v[:3]-desired.root_rotation_world@desired.v[:3],
            state.root_rotation_world@state.v[3:6]-desired.root_rotation_world@desired.v[3:6],
            state.joint_velocity-desired.joint_velocity]
        scaled=np.r_[dq[:3]/.05,dq[3:6]/.15,dq[6:]/.20,dv[:3]/.5,dv[3:6]/2.,dv[6:]/5.]
        return float(scaled@scaled)

    def _candidate(self,state,target,k):
        predicted=state;cost=0.;last=None
        for offset in range(min(2,60-k)):
            command=target if offset==0 else np.asarray(self.chosen['trajectory'][k+offset]['target'])
            try:
                last=predict_self_contact_spheres(self.model,self.interface,predicted,command,
                    self.geometry.spheres,geometry=self.path_geometry,substeps=16)
            except (ValueError,np.linalg.LinAlgError) as error:
                return dict(accepted=False,error=repr(error),horizon_step=offset)
            bad={n:v for n,v in last['diagnostics'].items() if
                (n=='nonfoot_ground_violation_m' and v>1e-6) or
                (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
            if bad:return dict(accepted=False,violations=bad,horizon_step=offset)
            predicted=last['state'];predicted.previous_command_target=command.copy()
            cost+=self._tracking_cost(predicted,self.library.state(self.chosen,k+offset))
        return dict(accepted=True,cost=cost,prediction_intervals=min(2,60-k),native_certified=False)

    def command(self,state,phase,time_s):
        wbc,diagnostic=super().command(state,phase,time_s)
        k,weight=self._reference_time();records=[];selected=None
        if k is not None:
            planned=np.asarray(self.chosen['trajectory'][k]['target'],float)
            desired=self.library.state(self.chosen,k-1)
            correction=(desired.joint_position-state.joint_position)+.05*(desired.joint_velocity-state.joint_velocity)
            candidates=[('planned',planned),('state_joint_feedback',planned+correction),
                ('wbc',wbc),('half_wbc',.5*(planned+wbc))]
            best=None
            for name,request in candidates:
                target=np.clip(request,self.interface.limits[:,0],self.interface.limits[:,1])
                result=self._candidate(state,target,k);records.append(dict(name=name,**result))
                if result['accepted']:
                    # Prefer the exact reference only for numerically tied
                    # predictions; state-error improvement otherwise decides.
                    cost=result['cost']+1e-8*float(np.sum((target-planned)**2))
                    if best is None or cost<best[0]:best=(cost,name,target)
            if best is not None:
                _,selected,wbc=best;diagnostic['source']='predictive_reference_selection:'+selected
        diagnostic.update(controller='pure_control_trajectory_recovery_v9',
            predictive_candidates=records,selected_predictive_candidate=selected,
            reference_command_preserved_as_candidate=True,
            no_feasible_predictive_candidate=k is not None and selected is None,
            global_sequence_optimum=False)
        return wbc,diagnostic
