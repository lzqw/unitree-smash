"""Bounded support-kinematic tracking of a feasible full-body reference.

This is a candidate feedback execution layer, not an MPC or support guarantee.
The input reference is a planned trajectory, never a future simulator state.
Only currently observed supporting feet enter the kinematic correction.
"""
import numpy as np
import pinocchio as pin


class LandingReferenceTrackingV1:
    def __init__(self, base, reference_q, reference_v, *, position_gain=.4,
                 velocity_gain_s=.06, orientation_gain=.2, angular_gain_s=.02,
                 max_translation_m=.025, max_rotation_rad=.04, max_joint_delta_rad=.08):
        self.base=base; self.model=base.model; self.interface=base.interface
        self.reference_q=np.asarray(reference_q,float).copy()
        self.reference_v=np.asarray(reference_v,float).copy()
        if (self.reference_q.shape!=(520,25) or self.reference_v.shape!=(520,24)
                or not np.isfinite(self.reference_q).all() or not np.isfinite(self.reference_v).all()):
            raise ValueError('Finite pre-command reference states for all 520 steps required')
        values=[position_gain,velocity_gain_s,orientation_gain,angular_gain_s,
                max_translation_m,max_rotation_rad,max_joint_delta_rad]
        if (not np.isfinite(values).all() or min(values)<0 or position_gain>1
                or velocity_gain_s>.2 or orientation_gain>1 or angular_gain_s>.2
                or not 0<max_translation_m<=.05 or not 0<max_rotation_rad<=.1
                or not 0<max_joint_delta_rad<=.08):
            raise ValueError('Bounded feedback gains and displacement limits required')
        self.gains=np.asarray(values[:4]);self.limits=np.asarray(values[4:])
        self.last_diagnostic=None

    @staticmethod
    def bounded_vector(value,limit):
        return value*min(1.,limit/max(float(np.linalg.norm(value)),1e-30))

    def action(self,offset,state,previous):
        step=offset-80
        if type(step) is not int or not 0<=step<520:
            raise ValueError('Original early recovery clock required')
        target,mode=self.base.action(offset,state,previous)
        self.last_diagnostic=dict(active=False,reference_step=step)
        contacts=np.asarray(state.contacts,bool)
        if not contacts.any() or step==0:return target,mode
        rq=self.reference_q[step];rv=self.reference_v[step]
        rotation=state.root_rotation_world;ref_rotation=pin.Quaternion(rq[3:7]).matrix()
        dp=rq[:3]-state.q[:3]
        dv=ref_rotation@rv[:3]-rotation@state.v[:3]
        dr=pin.log3(ref_rotation@rotation.T)
        dw=ref_rotation@rv[3:6]-rotation@state.v[3:6]
        # Do not add roundoff corrections to an exactly reproduced reference.
        if max(np.linalg.norm(x) for x in (dp,dv,dr,dw))<1e-12:return target,mode
        translation=self.bounded_vector(self.gains[0]*dp+self.gains[1]*dv,self.limits[0])
        angular=self.gains[2]*dr+self.gains[3]*dw;angular[2]=0.
        angular=self.bounded_vector(angular,self.limits[1])
        desired=np.r_[rotation.T@translation,rotation.T@angular]
        snapshot=self.model.snapshot(state.q,state.v)
        jac=snapshot['foot_jacobian'][contacts].reshape(-1,self.model.model.nv)
        legs=jac[:,self.model.v_indices[:12]]
        rhs=-jac[:,:6]@desired
        delta=legs.T@np.linalg.solve(legs@legs.T+.003**2*np.eye(len(rhs)),rhs)
        delta=np.clip(delta,-self.limits[2],self.limits[2])
        proposed=target.copy()
        proposed[:12]=np.clip(target[:12]+delta,self.interface.limits[:12,0],self.interface.limits[:12,1])
        self.last_diagnostic=dict(active=True,reference_step=step,contact_indices=np.flatnonzero(contacts).tolist(),
            position_error_world_m=dp.tolist(),velocity_error_world_m_s=dv.tolist(),
            attitude_error_world_rad=dr.tolist(),angular_velocity_error_world_rad_s=dw.tolist(),
            requested_virtual_root_delta_local=desired.tolist(),joint_delta_rad=(proposed[:12]-target[:12]).tolist(),
            scope='Kinematic command correction; actual dynamics must validate it')
        return proposed,'sustained_support_reference_tracking_candidate'
