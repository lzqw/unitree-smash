"""Optimized full-body reference plus continuous original-interface feedback.

The library supplies offline model plans, not expert demonstrations. Native
feedback and event phases persist until the episode ends. This is offline
trajectory optimization with WBC tracking, not online sequence MPC.
"""
import numpy as np
import pinocchio as pin
from scripts.pure_control_moving_foot_recovery_v7 import PureControlMovingFootRecoveryV7
from scripts.recovery_reference_library_v1 import RecoveryReferenceLibraryV1
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1,CourtCollisionPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from quadsmash.physics.court_v4 import load_court


class PureControlTrajectoryRecoveryV8(PureControlMovingFootRecoveryV7):
    def __init__(self,interface,nominal,library):
        super().__init__(interface,nominal)
        self.library=RecoveryReferenceLibraryV1(library,self.model)
        self.chosen=None;self.selection_done=False;self.selection_score=None;self.reference_index=None
        self.path_geometry=FootSeparationPathConstraintV1(
            CourtCollisionPathConstraintV1(self.geometry.ground,AuthoredNetClearanceV1(self.geometry.ground,load_court())),
            self.geometry.spheres,margin_m=0.)

    def _reference_time(self):
        if self.chosen is None or self.elapsed>=.60-1e-9:return None,0.
        k=int(np.clip(np.floor(self.elapsed/.01+1e-6),0,59))
        # Blend reference objectives during the last 100 ms; the final output
        # is always one newly solved feedback command, never student mixing.
        return k,float(min(1.,(.60-self.elapsed)/.10))

    def solve(self,state,joint_reference,**kwargs):
        k,weight=self._reference_time();self.reference_index=k
        if k is not None:
            target=np.asarray(self.chosen['trajectory'][k]['target'])
            joint_reference=weight*target+(1-weight)*joint_reference
        return super().solve(state,joint_reference,**kwargs)

    def reference(self,state,snapshot,joint_reference):
        base=super().reference(state,snapshot,joint_reference)
        k,weight=self._reference_time()
        if k is None:return base
        desired=self.library.state(self.chosen,k)
        before=self.library.state(self.chosen,k-1)
        snap_d=self.model.snapshot(desired.q,desired.v);snap_b=self.model.snapshot(before.q,before.v)
        com_acc=(snap_d['com_velocity_world']-snap_b['com_velocity_world'])/.01
        com_acc+=35.*(snap_b['com_world']-snapshot['com_world'])+12.*(snap_b['com_velocity_world']-snapshot['com_velocity_world'])
        rd=desired.root_rotation_world;rb=before.root_rotation_world;r=state.root_rotation_world
        omega_d=rd@desired.v[3:6];omega_b=rb@before.v[3:6]
        angular=r.T@((omega_d-omega_b)/.01)+35.*pin.log3(r.T@rb)+12.*(r.T@omega_b-state.v[3:6])
        joint=(desired.joint_velocity-before.joint_velocity)/.01
        joint+=25.*(before.joint_position-state.joint_position)+8.*(before.joint_velocity-state.joint_velocity)
        foot_ids=[self.model.names.index(n) for n in self.model.feet]
        foot_velocity_before=snap_b['foot_jacobian']@before.v
        foot_velocity_after=snap_d['foot_jacobian']@desired.v
        planned=dict(com_position=snap_b['com_world'],com_acceleration=com_acc,
            angular_acceleration_body=angular,joint_acceleration=joint,
            swing_foot_target_world=snap_b['body_positions_world'][foot_ids],
            swing_foot_velocity_world=foot_velocity_before,
            swing_foot_acceleration_world=(foot_velocity_after-foot_velocity_before)/.01)
        for key in planned:base[key]=weight*planned[key]+(1-weight)*base[key]
        self.reference_diagnostics.update(offline_model_reference=True,reference_weight=weight,
            reference_index=k,model_position_tracking_error_m=float(np.linalg.norm(before.q[:3]-state.q[:3])),
            model_attitude_tracking_error_rad=float(np.linalg.norm(pin.log3(r.T@rb))))
        return base

    def command(self,state,phase,time_s):
        if not self.selection_done:
            self.chosen,self.selection_score=self.library.match(state);self.selection_done=True
        target,diagnostic=super().command(state,phase,time_s)
        k,weight=self._reference_time();replay=None
        if k is not None and not diagnostic['solver_accepted']:
            candidate=np.asarray(self.chosen['trajectory'][k]['target'],float)
            try:
                result=predict_self_contact_spheres(self.model,self.interface,state,candidate,
                    self.geometry.spheres,geometry=self.path_geometry,substeps=16)
                bad={name:value for name,value in result['diagnostics'].items() if
                    (name=='nonfoot_ground_violation_m' and value>1e-6) or
                    (name in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and value>3e-4)}
                replay=dict(accepted=not bad,violations=bad,substeps=16,native_certified=False)
                if not bad:
                    target=candidate.copy();diagnostic['source']='planned_command_with_actual_state_model_replay'
            except (ValueError,np.linalg.LinAlgError) as error:replay=dict(accepted=False,error=repr(error))
        bounded=np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        correction=float(np.max(abs(bounded-target)))
        if correction>1e-8:raise ValueError('Command outside original limits beyond floating-point tolerance')
        target=bounded
        diagnostic['original_limit_roundoff_correction_rad']=correction
        diagnostic.update(controller='pure_control_trajectory_recovery_v8',
            offline_reference_source_key=self.chosen['source_key'] if self.chosen is not None else None,
            initial_state_match_score=self.selection_score,reference_active=k is not None,
            offline_trajectory_not_native_expert=True,sequence_mpc=False,command_replay=replay)
        return target,diagnostic
