"""Build same-candidate finite-mass impact evidence from a full-body state."""
import hashlib,json
import numpy as np
import pinocchio as pin
from scripts.contact_terminal_attribution_v1 import contact_kinematics
from quadsmash.control.strict_impact_admission_v1 import StrictImpactAdmissionV1

class StrikeStateImpactBuilderV1:
    def __init__(self, model, impact_admission, *, initial_state_hash, command_sequence_hash):
        self.model=model;self.admission=impact_admission
        self.initial_state_hash=str(initial_state_hash);self.command_sequence_hash=str(command_sequence_hash)
    def __call__(self, terminal, *, ball_position, ball_velocity, contact_point_local, contact_time_s, support_mode):
        q=np.asarray(terminal['q'],float);v=np.asarray(terminal['v'],float)
        if q.shape!=(self.model.model.nq,) or v.shape!=(self.model.model.nv,) or not np.isfinite(np.r_[q,v]).all(): raise ValueError('Finite terminal full-body state required')
        offset=np.asarray(contact_point_local,float).copy();offset[0]+=self.admission.evaluator.physics['collision_radius_m']
        position,normal,jac,jnormal=contact_kinematics(self.model,q,offset)
        support_mask=np.asarray(terminal.get('contacts'),dtype=bool)
        if support_mask.shape!=(4,):raise ValueError('Actual four-foot contact mask required')
        # Foot contact Jacobian is assembled from the current loaded model frames.
        pin.computeJointJacobians(self.model.model,self.model.data,q);pin.updateFramePlacements(self.model.model,self.model.data)
        rows=[]
        for foot in self.model.feet:
            frame=self.model.body_frames[foot];J=pin.getFrameJacobian(self.model.model,self.model.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED);rows.append(J[:3])
        support=np.vstack([rows[i] for i in range(4) if support_mask[i]]) if support_mask.any() else np.zeros((0,self.model.model.nv))
        mass=pin.crba(self.model.model,self.model.data,q);mass=.5*(mass+mass.T)
        state={'mass_matrix':mass,'contact_jacobian_world':jac,'support_jacobian':support,'support_mode':support_mode,
               'support_model':'same_candidate_pinocchio_state','initial_state_hash':self.initial_state_hash,
               'command_sequence_hash':self.command_sequence_hash,'terminal_state_hash':hashlib.sha256(np.r_[q,v].tobytes()).hexdigest()}
        # Record mass-model construction independently of contact validity.
        jn=normal@jac; response=np.linalg.solve(mass,jn)
        if support.size:
            inv_support=np.linalg.solve(mass,support.T)
            response-=inv_support@np.linalg.pinv(support@inv_support)@(support@response)
        inverse_mass=float(jn@response)
        mass_status='FINITE_MASS_MODEL_COMPUTED' if inverse_mass>1e-14 else 'EFFECTIVE_MASS_UNVERIFIED'
        result=self.admission.evaluate(shuttle_position=ball_position,shuttle_velocity_before=ball_velocity,
            racket_contact_position=position-self.admission.evaluator.physics['collision_radius_m']*normal,
            racket_contact_velocity=jac@v,racket_normal=normal,contact_point_local=contact_point_local,
            robot_state=state,task_context={'event_order':True,'no_numerical_failure':True})
        if result.get('strict_status')=='EFFECTIVE_MASS_UNVERIFIED' and result.get('rejection_reason') in ('ball_string_contact_position_mismatch','outside_authored_string_surface'):
            result['strict_status']=mass_status
        result.update(mass_model_computed=True,inverse_effective_mass_kg_inv=inverse_mass,same_candidate_state=True,terminal_state_hash=state['terminal_state_hash'],contact_time_s=float(contact_time_s),support_mode=support_mode,
                      actual_contact_position_world_m=position.tolist(),actual_contact_normal_world=normal.tolist(),actual_contact_velocity_world_m_s=(jac@v).tolist())
        return result
