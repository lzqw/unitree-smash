"""Public-state foot preparation and compliant landing reference prototype.

Inspired by Roscia et al. (RA-L 2023) and Ding et al. (TRO 2024).
This is a bounded joint-position adapter, NOT their torque WBC or a proof of
dynamic feasibility. All forces remain supplied by the original actuators.
The full loaded asset supplies FK, COM and COM velocity. A constant-height
capture approximation supplies horizontal placement; actual per-foot contact
switches flight placement to a critically damped body-reference trajectory.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'configs/go2_piper_racket_landing_model_v1.json'
DEFAULTS = dict(mode='prepare_and_buffer', blend_s=.08, settling_s=.7,
                landing_height_m=.30, capture_gain=1., ik_iterations=5,
                joint_target_rate_rad_s=20., ik_damping=.025)


def settings(value):
    if value is None:return None
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('Declare every reactive landing field explicitly')
    result=dict(value)
    if result['mode'] not in ('prepare_only', 'prepare_and_buffer', 'prepare_and_buffer_vhsip',
                              'post_touchdown_buffer_vhsip'):
        raise ValueError('Unknown landing comparison mode')
    bounds=dict(blend_s=(.01,.3),settling_s=(.2,2.),landing_height_m=(.15,.42),
                capture_gain=(0.,2.),joint_target_rate_rad_s=(1.,30.),ik_damping=(.001,.1))
    for key,(lo,hi) in bounds.items():
        v=result[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError('Invalid landing setting: '+key)
    if type(result['ik_iterations']) is not int or not 1<=result['ik_iterations']<=12:
        raise ValueError('Invalid landing IK iterations')
    return result


def rotate(r,v):return (r@v.unsqueeze(-1)).squeeze(-1)


def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


class LoadedKinematics:
    """Batched FK/Jacobian of the actual composed USD, no simulator reads."""
    def __init__(self,device='cpu',dtype=torch.float32):
        self.record=json.loads(MODEL.read_text());self.sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest()
        self.names=self.record['body_names'];self.ids={n:i for i,n in enumerate(self.names)}
        self.feet=[self.ids[n+'_foot'] for n in ('FL','FR','RL','RR')]
        self.leg_coordinates=torch.tensor([[i,i+4,i+8] for i in range(4)],device=device)
        self.tensor=lambda v:torch.as_tensor(v,device=device,dtype=dtype)
        self.eye=torch.eye(3,device=device,dtype=dtype)
        self.mass=self.tensor([b['mass_kg'] for b in self.record['bodies']])
        self.com=self.tensor([b['center_of_mass_m'] for b in self.record['bodies']])
        self.edges=[];pending=list(self.record['joints']);seen={self.record['root_body']}
        joint_ids={n:i for i,n in enumerate(self.record['joint_order'])}
        while pending:
            ready=[j for j in pending if j['parent'] in seen]
            if not ready:raise ValueError('Disconnected asset model')
            for j in ready:
                axis=self.tensor([0.,0.,0.])
                if j['axis'] is not None:axis['XYZ'.index(j['axis'])]=1.
                x,y,z=axis.unbind();zero=torch.zeros_like(x)
                skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                self.edges.append((self.ids[j['parent']],self.ids[j['child']],joint_ids.get(j['name']),
                    self.tensor(j['parent_position_m']),self.tensor(j['child_position_m']),
                    quat_matrix(self.tensor(j['parent_rotation_wxyz'])),quat_matrix(self.tensor(j['child_rotation_wxyz'])),axis,skew))
                seen.add(j['child']);pending.remove(j)

    def forward(self,q,qd=None,root_v=None,root_w=None):
        n=len(q);b=len(self.names)
        p=q.new_zeros((n,b,3));r=self.eye.expand(n,b,3,3).clone()
        v=torch.zeros_like(p);w=torch.zeros_like(p)
        root=self.ids[self.record['root_body']]
        if root_v is not None:v[:,root]=root_v
        if root_w is not None:w[:,root]=root_w
        origin=q.new_zeros((n,18,3));axis_world=torch.zeros_like(origin)
        for parent,child,j,p0,p1,r0,r1,axis,skew in self.edges:
            joint_r=r[:,parent]@r0
            offset0=rotate(r[:,parent],p0)
            if j is None:motion=self.eye;added_w=0.
            else:
                angle=q[:,j,None,None]
                motion=self.eye+angle.sin()*skew+(1-angle.cos())*(skew@skew)
                axis_world[:,j]=rotate(joint_r,axis);origin[:,j]=p[:,parent]+offset0
                added_w=axis_world[:,j]*qd[:,j,None] if qd is not None else 0.
            r[:,child]=joint_r@motion@r1.T
            offset1=rotate(r[:,child],p1)
            p[:,child]=p[:,parent]+offset0-offset1
            w[:,child]=w[:,parent]+added_w
            v[:,child]=v[:,parent]+torch.linalg.cross(w[:,parent],offset0)-torch.linalg.cross(w[:,child],offset1)
        offset=rotate(r,self.com)
        center=((p+offset)*self.mass[None,:,None]).sum(1)/self.mass.sum()
        center_v=((v+torch.linalg.cross(w,offset))*self.mass[None,:,None]).sum(1)/self.mass.sum()
        feet=p[:,self.feet]
        columns=torch.linalg.cross(axis_world[:,self.leg_coordinates],feet[:,:,None]-origin[:,self.leg_coordinates])
        return dict(feet=feet,jacobian=columns.transpose(-1,-2),center=center,center_v=center_v,
                    position=p,rotation=r)


class ReactiveLandingV1:
    def __init__(self,count,device,config,asset_sha256):
        self.config=settings(config);self.model=LoadedKinematics(device)
        if asset_sha256 != self.model.record['asset_sha256']:raise ValueError('Landing asset mismatch')
        self.count=count;self.device=device
        for name in ('hit','active','landed'):
            setattr(self,name,torch.zeros(count,device=device,dtype=torch.bool))
        for name in ('start','touchdown','last_time'):
            setattr(self,name,torch.zeros(count,device=device))
        self.anchors=torch.zeros((count,4,3),device=device)
        # Contact support is latched after the first measured touchdown.  The
        # upstream landing controller keeps the touchdown trajectory active
        # after the contact detector changes; using the instantaneous mask here
        # made a one-foot landing immediately lose its anchor when PhysX
        # hysteresis flickered.
        self.support=torch.zeros((count,4),device=device,dtype=torch.bool)
        self.c0=torch.zeros((count,3),device=device);self.v0=torch.zeros_like(self.c0)
        self.goal=torch.zeros_like(self.c0);self.r0=torch.eye(3,device=device).expand(count,3,3).clone()
        self.landing_plane=torch.zeros_like(self.c0)
        self.last_q=torch.zeros((count,12),device=device)
        self.nominal_feet=None
        self.last_diagnostic={}

    def reset(self,ids=None):
        if ids is None:ids=slice(None)
        for name in ('hit','active','landed','start','touchdown','last_time','anchors','support','c0','v0','goal','landing_plane','last_q'):
            getattr(self,name)[ids]=0
        self.r0[ids]=self.model.eye

    def _vhsip_capture(self, velocity_xy, position_xy, height, vertical_velocity):
        """Batched VHSIP/ZMP solve following Roscia et al. (RA-L 2023).

        This is the analytical one-input OCP from the reference controller,
        evaluated with torch tensors so it can run alongside the policy on the
        selected environments.  It returns a horizontal capture/ZMP offset;
        the downstream IK and original position-PD limits remain unchanged.
        """
        # The paper uses a 1.8 s settling horizon.  The current position-PD
        # adapter only needs the near-touchdown capture reference; keeping a
        # 0.6 s receding window avoids hundreds of tiny Python-side launches
        # per native tick.  The post-touchdown buffer still uses settling_s.
        dt=.01; settling=.6; gravity=9.81; compression=.4*height.clamp_min(.15)
        mass=self.model.mass.sum().to(height)
        stiffness=mass*(8./settling)**2
        eig=-torch.sqrt(stiffness/mass)
        limit=torch.e*compression*eig
        fast=vertical_velocity < limit
        fast_k=mass*(vertical_velocity/(torch.e*compression).clamp_min(1e-4))**2
        stiffness=torch.where(fast,fast_k,stiffness)
        damping=2.*torch.sqrt(mass*stiffness)
        eig=-torch.sqrt(stiffness/mass)
        steps=int(round(settling/dt))
        time=torch.arange(steps,device=height.device,dtype=height.dtype)*dt
        et=torch.exp(eig[:,None]*time[None])
        vz_t=et*(eig[:,None]*time[None]+1.)*vertical_velocity[:,None]
        z=et*time[None]*vertical_velocity[:,None]+height[:,None]
        az=-(damping/mass)[:,None]*vz_t-(stiffness/mass)[:,None]*(et*time[None]*vertical_velocity[:,None])
        omega=((gravity+az)/z.clamp_min(.12)).clamp_min(0.)
        batch=height.shape[0]
        P=torch.eye(2,device=height.device,dtype=height.dtype).expand(batch,2,2).clone()
        U=torch.zeros((batch,2,1),device=height.device,dtype=height.dtype)
        for k in range(steps-1):
            om=omega[:,k]
            A=torch.zeros((batch,2,2),device=height.device,dtype=height.dtype)
            # State order is [velocity, position].  Keep both the acceleration
            # response and the kinematic position integral in the discrete
            # VHSIP model; omitting A[1,0] makes position remain constant.
            A[:,0,0]=1.; A[:,1,1]=1.; A[:,0,1]=dt*om; A[:,1,0]=dt
            B=torch.zeros((batch,2,1),device=height.device,dtype=height.dtype)
            B[:,0,0]=-dt*om
            P=A@P; U=A@U+B
        cvp=P[:,0,:]; cpp=P[:,1,:]
        cvu=U[:,0,0]; cpu=U[:,1,0]
        qu=cvu*cvu+(cpu-1.)**2+1e-6
        qxu=cvu[:,None]*cvp+(cpu-1.)[:,None]*cpp
        x0=torch.stack((velocity_xy,position_xy),dim=-1)
        # x0 is [batch, axis(x/y), (velocity, position)]; keep the
        # per-batch cost row aligned with both horizontal axes.
        return -(qxu[:,None,:]*x0).sum(-1)/qu[:,None]

    @torch.inference_mode()
    def action(self,base,public,default,limits,enabled_mask=None,native_env=None):
        if public.shape!=(self.count,794) or base.shape!=(self.count,18):raise ValueError('Require public794D/18D native interface')
        if enabled_mask is None:
            enabled_mask=torch.ones((self.count,),device=public.device,dtype=torch.bool)
        else:
            enabled_mask=torch.as_tensor(enabled_mask,device=public.device,dtype=torch.bool)
            if enabled_mask.shape!=(self.count,):raise ValueError('Require one landing enable flag per environment')
        clock=public[:,775]
        self.hit|=(public[:,773]>.5)&enabled_mask
        if native_env is None:
            forward=public[:,763:766];up=-public[:,6:9]
            left=torch.linalg.cross(up,forward)
            rotation=torch.stack((forward,left,up),1)
            q=default[None]+public[:,9:27]
            actual=self.model.forward(q,public[:,27:45],public[:,:3],public[:,3:6])
            root=public.new_tensor([1.10,0.,1.80])-rotate(rotation,public[:,760:763])
            contacts=public[:,57:61]>.5
        else:
            # Use the same live state as the native landing MPC adapter.  The
            # ball/phase clocks remain policy observables, but FK, COM velocity,
            # world feet and contacts must come from Isaac's current tensors.
            data=native_env.robot.data
            q=data.joint_pos[:,native_env.joint_ids]
            qd=data.joint_vel[:,native_env.joint_ids]
            root=data.root_link_pos_w-native_env.scene.env_origins
            quat=data.root_link_quat_w
            w,x,y,z=quat.unbind(-1)
            rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
            omega=data.root_com_ang_vel_w
            com_v=data.root_com_lin_vel_w
            com_offset=rotate(rotation,data.body_com_pos_b[:,0])
            link_v=com_v-torch.linalg.cross(omega,com_offset)
            root_v=rotate(rotation.transpose(-1,-2),link_v)
            root_w=rotate(rotation.transpose(-1,-2),omega)
            actual=self.model.forward(q,qd,root_v,root_w)
            contacts=native_env.observed_foot_contact
        cv=rotate(rotation,actual['center_v'])
        newly=self.hit&(~self.active)&(cv[:,2]<-.05)
        self.start[newly]=clock[newly];self.last_q[newly]=q[newly,:12];self.last_time[newly]=clock[newly]-.01
        self.active|=newly
        active=self.active&enabled_mask
        if not bool(active.any()):return base
        ids=torch.where(active)[0];cfg=self.config;n=len(ids)
        r=rotation[ids];current=q[ids];t=clock[ids]
        root=root[ids]
        center=root+rotate(r,actual['center'][ids]);vel=cv[ids]
        world_feet=root[:,None]+rotate(r[:,None],actual['feet'][ids])
        contacts=contacts[ids]
        first=contacts.any(1)&(~self.landed[ids])
        added=contacts&(~self.support[ids])
        self.anchors[ids]=torch.where(added[:,:,None],world_feet,self.anchors[ids])
        # Keep touchdown anchors for the whole recovery buffer.  New feet can
        # join the support set, but a transient force-threshold drop must not
        # erase the already planned landing reference.
        self.support[ids]|=contacts
        support=self.support[ids]
        new_ids=ids[first]
        self.touchdown[new_ids]=t[first];self.c0[new_ids]=center[first];self.v0[new_ids]=vel[first];self.r0[new_ids]=r[first]
        self.landed[new_ids]=True
        if cfg['mode']=='post_touchdown_buffer_vhsip':
            # The post-touchdown-only variant must not inherit an airborne
            # blend clock or a stale pre-contact target.  Start its bounded
            # joint transition from the command that was actually present at
            # the measured touchdown tick.
            self.start[new_ids]=t[first]
            self.last_q[new_ids]=q[new_ids,:12]
            self.last_time[new_ids]=t[first]
        # A horizontal foot rectangle follows the current yaw, not body tilt.
        yaw=torch.atan2(r[:,1,0],r[:,0,0]);c,s=yaw.cos(),yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
        ryaw=torch.stack((c,-s,z,s,c,z,z,z,one),1).reshape(n,3,3)
        # The default joint posture is fixed by the admitted native interface;
        # cache its foot geometry instead of re-running FK at every 100 Hz tick.
        if self.nominal_feet is None or self.nominal_feet.shape[0] != self.count:
            self.nominal_feet=self.model.forward(default[None].expand(self.count,-1))['feet']
        nominal=self.nominal_feet[ids]
        shape=nominal-nominal.mean(1,keepdim=True);shape[:,:,2]=0.
        shape=rotate(ryaw[:,None],shape)
        com_offset=rotate(ryaw,actual['center'][ids])
        h=cfg['landing_height_m']+com_offset[:,2]
        has_contacts=support.any(1)
        support_center=(self.anchors[ids]*support[:,:,None]).sum(1)/support.sum(1).clamp_min(1)[:,None]
        support_center=torch.where(has_contacts[:,None],support_center,center)
        if cfg['mode']=='prepare_and_buffer_vhsip':
            position_xy=center[:,:2]-support_center[:,:2]
            capture=cfg['capture_gain']*self._vhsip_capture(vel[:,:2],position_xy,h,vel[:,2])
            capture=capture.clamp(-.35,.35)
            plane_center=support_center.clone();plane_center[:,:2]+=capture
            # ``plane_center`` is a foot-plane reference.  When no measured
            # support exists, the fallback support center is the COM, so the
            # foot plane must be below it by the nominal COM height.  The COM
            # target is reconstructed separately as ground + h below.
            measured_ground=(self.anchors[ids,:,2]*support).sum(1)/support.sum(1).clamp_min(1)
            ground=torch.where(has_contacts,measured_ground,center[:,2]-h)
            plane_center[:,2]=ground
        else:
            capture=cfg['capture_gain']*(h.clamp_min(.1)/9.81).sqrt()[:,None]*vel[:,:2]
            plane_center=center.clone();plane_center[:,:2]+=capture;plane_center[:,2]-=h
        feet_goal=plane_center[:,None]+shape
        # Actual contact anchors determine surface height; four simultaneous
        # contacts are never required. Subsequent feet enter support separately.
        measured_ground=(self.anchors[ids,:,2]*support).sum(1)/support.sum(1).clamp_min(1)
        ground=torch.where(has_contacts,measured_ground,center[:,2]-h)
        self.goal[new_ids,:2]=plane_center[first,:2]
        self.goal[new_ids,2]=(ground+h)[first]
        self.landing_plane[new_ids]=torch.cat((plane_center[first,:2],ground[first,None]),1)
        buffered=self.landed[ids]&(cfg['mode'] in ('prepare_and_buffer','prepare_and_buffer_vhsip',
                                                    'post_touchdown_buffer_vhsip'))
        feet_goal=torch.where(self.landed[ids,None,None],self.landing_plane[ids,None]+shape,feet_goal)
        tau=(t-self.touchdown[ids]).clamp_min(0.)
        rate=7./cfg['settling_s'];decay=(-rate*tau).exp()
        displacement=self.c0[ids]-self.goal[ids]
        planned_center=self.goal[ids]+(displacement+(self.v0[ids]+rate*displacement)*tau[:,None])*decay[:,None]
        # Smooth orientation reference from measured touchdown orientation to
        # upright. SVD projection keeps the interpolated rotation on SO(3).
        amount=1.-(1.+rate*tau)*decay
        mixed=torch.lerp(self.r0[ids],ryaw,amount[:,None,None])
        u,_,vh=torch.linalg.svd(mixed)
        sign=torch.linalg.det(u@vh)
        u=u.clone();u[:,:,-1]*=sign[:,None]
        planned_rotation=u@vh
        stance_root=planned_center-rotate(planned_rotation,actual['center'][ids])
        target_world=torch.where(support[:,:,None],self.anchors[ids],feet_goal)
        target_body=rotate(r.transpose(-1,-2)[:,None],feet_goal-root[:,None])
        stance_body=rotate(planned_rotation.transpose(-1,-2)[:,None],target_world-stance_root[:,None])
        target_body=torch.where(buffered[:,None,None],stance_body,target_body)
        # For the preparation-only ablation, retain the original source as soon
        # as any foot lands. Blending is shared so the comparison is explicit.
        if cfg['mode']=='post_touchdown_buffer_vhsip':
            enabled=self.landed[ids]
        else:
            enabled=(~self.landed[ids])|buffered
        desired=limits[:,0]+.5*(base[ids]+1)*(limits[:,1]-limits[:,0])
        solution=desired.clone();coords=self.model.leg_coordinates
        eye=self.model.eye
        for _ in range(cfg['ik_iterations']):
            k=self.model.forward(solution);j=k['jacobian'];error=target_body-k['feet']
            delta=rotate(j.transpose(-1,-2),torch.linalg.solve(j@j.transpose(-1,-2)+cfg['ik_damping']**2*eye,error.unsqueeze(-1)).squeeze(-1))
            # Numerical solver trust region; final target rate and physical
            # joint bounds remain independently enforced.
            solution[:,coords]+=delta.clamp(-.3,.3)
            solution[:,:12]=solution[:,:12].clamp(limits[:12,0],limits[:12,1])
        blend_clock=self.touchdown[ids] if cfg['mode']=='post_touchdown_buffer_vhsip' else self.start[ids]
        blend=((t-blend_clock)/cfg['blend_s']).clamp(0,1)
        blend=blend.square()*(3-2*blend)*enabled
        blended=torch.lerp(desired[:,:12],solution[:,:12],blend[:,None])
        dt=(t-self.last_time[ids]).clamp(0.,.02)
        rate_limit=cfg['joint_target_rate_rad_s']*dt[:,None]
        bounded=self.last_q[ids]+(blended-self.last_q[ids]).clamp(-rate_limit,rate_limit)
        bounded=bounded.clamp(limits[:12,0],limits[:12,1])
        self.last_q[ids]=bounded;self.last_time[ids]=t
        result=base.clone()
        result[ids,:12]=torch.where(enabled[:,None],2*(bounded-limits[:12,0])/(limits[:12,1]-limits[:12,0])-1,base[ids,:12])
        achieved=self.model.forward(torch.cat((bounded,desired[:,12:]),1))['feet']
        self.last_diagnostic=dict(env_ids=ids,active=enabled,buffering=buffered,feet_contact=contacts,
            desired_feet_body_m=target_body,commanded_feet_body_m=achieved,
            foot_reference_error_m=(target_body-achieved).norm(dim=-1),
            measured_com_world_m=center,measured_com_velocity_world_mps=vel,
            planned_com_world_m=planned_center,source='public-only position-interface prototype')
        return result
