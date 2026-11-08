"""Structured whole-body recovery reference through the actual position interface.

Current front-foot locations anchor the support plan. Rear feet receive a
declared landing footprint while a virtual base pose settles and the arm
brakes. IK only proposes joint commands: full coupled rollout still decides
dynamic feasibility, clearances and the complete stable-window requirement.
"""
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator

from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1


PARAMETER_NAMES = ['base_settle_s','terminal_root_height_m','root_forward_shift_m',
    'root_lateral_shift_m','rear_forward_offset_m','rear_half_width_m',
    'front_outward_shift_m','arm_settle_s'] + [f'arm_offset_{i}_rad' for i in range(6)]
LOWER = np.r_[.15,.20,-.15,-.12,-.08,.10,0.,.12,np.full(6,-.30)]
UPPER = np.r_[.90,.36,.15,.12,.08,.22,.08,.80,np.full(6,.30)]
NOMINAL = np.r_[.45,.28,0.,0.,0.,.14,0.,.35,np.zeros(6)]


class FootAnchoredRecoveryReferenceV1:
    def __init__(self, model, interface, initial, spheres, parameters, *, control_tick):
        self.model=model;self.interface=interface;self.initial=initial
        self.tick=int(control_tick);self.parameters=np.asarray(parameters,float)
        p=self.parameters
        if (p.shape!=(14,) or not np.isfinite(p).all() or np.any(p<LOWER-1e-12)
                or np.any(p>UPPER+1e-12) or not 280<=self.tick<700
                or [s['body'] for s in spheres]!=model.feet):
            raise ValueError('Declared bounded reference parameters and actual control clock required')
        rotation=initial.root_rotation_world
        yaw=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        upright=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        angle=pin.log3(rotation.T@upright)
        final_root=initial.q[:3]+upright@np.array([p[2],p[3],0.])
        final_root[2]=p[1]
        self.spheres=spheres
        current=self.centers(initial.q)
        feet=current.copy()
        # Front support points start from measured locations, not a different
        # trial's nominal root trajectory. Outward moves are explicit variables.
        feet[0]+=upright@np.array([0.,p[6],0.])
        feet[1]+=upright@np.array([0.,-p[6],0.])
        for index,side in [(2,1.),(3,-1.)]:
            feet[index]=final_root+upright@np.array([-.19+p[4],side*p[5],0.])
        feet[:,2]=[x['radius_m'] for x in spheres]
        self.foot_center_targets_world=feet
        times=np.array([0.,.04,.10,.20,.40,.70,1.20,2.,3.5])
        values=[initial.previous_command_target.copy()];seed=initial.q.copy();diagnostics=[]
        smooth=lambda x:np.clip(x,0.,1.)**2*(3.-2.*np.clip(x,0.,1.))
        for t in times[1:]:
            blend=float(smooth(t/p[0]));arm_blend=float(smooth(t/p[7]))
            seed[:3]=(1.-blend)*initial.q[:3]+blend*final_root
            seed[3:7]=pin.Quaternion(rotation@pin.exp3(blend*angle)).coeffs()
            arm_goal=np.clip(initial.joint_position[12:]+p[8:],interface.limits[12:,0],interface.limits[12:,1])
            seed[model.q_indices[12:]]=(1.-arm_blend)*initial.previous_command_target[12:]+arm_blend*arm_goal
            # Keep early feet continuous as the virtual base moves. References
            # are proposals only: unreachable targets retain an explicit error.
            foot_blend=float(smooth(t/p[0]))
            desired=(1.-foot_blend)*current+foot_blend*feet
            for _ in range(16):
                centers,jacobian=self.centers(seed,jacobian=True)
                error=(desired-centers).reshape(-1)
                if np.max(abs(error))<1e-5:break
                legs=jacobian.reshape(12,model.model.nv)[:,model.v_indices[:12]]
                delta=legs.T@np.linalg.solve(legs@legs.T+.003**2*np.eye(12),error)
                seed[model.q_indices[:12]]=np.clip(seed[model.q_indices[:12]]+np.clip(delta,-.18,.18),
                    interface.limits[:12,0],interface.limits[:12,1])
            error=np.linalg.norm(desired-self.centers(seed),axis=1)
            diagnostics.append(dict(time_s=float(t),maximum_ik_residual_m=float(error.max()),
                per_foot_residual_m=error.tolist(),virtual_root=seed[:7].tolist()))
            values.append(seed[model.q_indices].copy())
        self.ik_diagnostics=diagnostics
        self.curve=PchipInterpolator(times,np.asarray(values),axis=0,extrapolate=False)

    def centers(self,q,*,jacobian=False):
        m=self.model
        if jacobian:pin.computeJointJacobians(m.model,m.data,q)
        else:pin.forwardKinematics(m.model,m.data,q)
        pin.updateFramePlacements(m.model,m.data)
        centers=[];rows=[]
        for sphere in self.spheres:
            frame=m.body_frames[sphere['body']];pose=m.data.oMf[frame]
            offset=pose.rotation@np.asarray(sphere['center_body_m'])
            centers.append(pose.translation+offset)
            if jacobian:
                j=pin.getFrameJacobian(m.model,m.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                rows.append(j[:3]-pin.skew(offset)@j[3:])
        return (np.asarray(centers),np.asarray(rows)) if jacobian else np.asarray(centers)

    def action(self,offset,state,previous):
        tick=offset-80+280
        if type(tick) is not int or not self.tick<=tick<800:raise ValueError('Original episode clock required')
        if tick==self.tick:
            if not np.array_equal(previous,self.initial.previous_command_target):raise ValueError('Changed actual history')
            return previous.copy(),'actual_command_handoff'
        target=np.clip(self.curve(min(.01*(tick-self.tick),3.5)),self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,previous[12:]+.01*self.interface.arm_rate)
        return target,'foot_anchored_whole_body_reference'


class FootAnchoredRecoveryFeedbackV1:
    def __init__(self,reference):
        self.reference=reference;self.model=reference.model;self.interface=reference.interface
        self.feedback=TerminalAttitudeFeedbackV1(reference,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)

    def action(self,offset,state,previous):
        tick=offset-80+280
        if tick==self.reference.tick:return self.reference.action(offset,state,previous)
        return self.feedback.action(offset,state,previous)
