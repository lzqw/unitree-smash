"""Contact-state centroidal feedback through the original joint PD interface.

Prototype: full loaded inertia/momentum and joint gravity are modeled, while
joint Coriolis/acceleration inverse dynamics are not. Feasible planned support
forces are not measured forces. PhysX execution remains the necessary test.
No aerial commands, base forces or actuator parameters are changed here.
"""
import math

import torch

from .reactive_landing_v1 import LoadedKinematics,quat_matrix,rotate


DEFAULTS=dict(root_height_m=.30,horizontal_gain_s2=6.,horizontal_damping_s=5.,
    vertical_gain_s2=80.,vertical_damping_s=18.,upright_gain_s2=50.,
    momentum_damping_s=8.,blend_s=.05,solver_iterations=80,
    wrench_length_m=.30,force_regularization=.005)
LEG_KP=25.
LEG_KD=.5
LEG_TORQUE_LIMIT=23.5
# Asset foot colliders have no bound physics material: the scene default is
# .5, ground is .8, and both use average combination. Validate live at admission.
FRICTION=.65


def settings(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!=set(DEFAULTS):
        raise ValueError('Declare every centroidal support setting')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in value.values()):
        raise ValueError('Support settings must be finite positive numbers')
    if type(value['solver_iterations']) is not int or value['solver_iterations']>100:
        raise ValueError('Declare a positive integer solver iteration budget at most100')
    return dict(value)


def cross_matrix(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)


class LoadedCentroidalModel(LoadedKinematics):
    def __init__(self,device='cpu',dtype=torch.float32):
        super().__init__(device,dtype)
        ancestry=torch.zeros((len(self.names),18),device=device,dtype=dtype)
        for parent,child,j,*_ in self.edges:
            ancestry[child]=ancestry[parent]
            if j is not None:ancestry[child,j]=1
        self.ancestry=ancestry
        principal=quat_matrix(self.tensor([b['principal_axes_wxyz'] for b in self.record['bodies']]))
        diagonal=torch.diag_embed(self.tensor([b['diagonal_inertia_kg_m2'] for b in self.record['bodies']]))
        self.link_inertia=principal@diagonal@principal.transpose(-1,-2)

    def dynamics(self,q,qd,root_v_body,root_w_body,rotation_world):
        k=self.forward(q)
        p,r=k['position'],k['rotation'];n=len(q)
        origin=q.new_zeros((n,18,3));axes=torch.zeros_like(origin)
        for parent,_,j,p0,_,r0,_,axis,_ in self.edges:
            if j is not None:
                origin[:,j]=p[:,parent]+rotate(r[:,parent],p0)
                axes[:,j]=rotate(r[:,parent]@r0,axis)
        pc=p+rotate(r,self.com)
        jac=torch.linalg.cross(axes[:,None],pc[:,:,None]-origin[:,None])*self.ancestry[None,:,:,None]
        vc=(root_v_body[:,None]+torch.linalg.cross(root_w_body[:,None],pc)
            +(jac*qd[:,None,:,None]).sum(2))
        wc=root_w_body[:,None]+torch.einsum('bj,njk,nj->nbk',self.ancestry,axes,qd)
        center=(pc*self.mass[None,:,None]).sum(1)/self.mass.sum()
        cv=(vc*self.mass[None,:,None]).sum(1)/self.mass.sum()
        relative=pc-center[:,None]
        inertia=r@self.link_inertia@r.transpose(-1,-2)
        momentum=(rotate(inertia,wc)+torch.linalg.cross(relative,
            self.mass[None,:,None]*(vc-cv[:,None]))).sum(1)
        composite=(inertia+self.mass[None,:,None,None]*(
            relative.square().sum(-1)[...,None,None]*self.eye
            -relative.unsqueeze(-1)*relative.unsqueeze(-2))).sum(1)
        gravity_body=rotate(rotation_world.transpose(-1,-2),q.new_tensor([0.,0.,-9.81]))
        gravity_torque=-(jac*(self.mass[None,:,None,None]*gravity_body[:,None,None])).sum((1,3))
        return dict(k,center=center,center_v=cv,
            momentum_world=rotate(rotation_world,momentum),
            inertia_world=rotation_world@composite@rotation_world.transpose(-1,-2),
            joint_gravity_torque=gravity_torque,body_com_positions=pc)


def project_friction(forces,contacts,mu=FRICTION):
    """Euclidean projection onto fz>=0, norm(fxy)<=mu*fz."""
    horizontal=forces[...,:2];normal=forces[...,2]
    length=horizontal.norm(dim=-1)
    inside=(normal>=0)&(length<=mu*normal)
    z=((mu*length+normal)/(1.+mu*mu)).clamp_min(0.)
    xy=horizontal*(mu*z/length.clamp_min(1e-12))[...,None]
    projected=torch.cat((xy,z[...,None]),-1)
    return torch.where((contacts&inside)[...,None],forces,
        torch.where(contacts[...,None],projected,0.))


def support_forces(lever_world,jacobian_body,rotation_world,gravity,torque_low,torque_high,
                   contacts,desired_wrench,previous,config):
    """Constrained convex wrench least squares with a feasible incumbent.

    ADMM separates the friction cones and linear torque boxes. Final scaling keeps
    friction/torque limits exact when zero support is torque-feasible; otherwise
    that leg is explicitly rejected and its source command is retained. Fixed
    iterations are not a convergence certificate; retain the best feasible point.
    """
    n=len(lever_world);eye=torch.eye(3,device=lever_world.device,dtype=lever_world.dtype)
    eligible=contacts&((gravity>=torque_low)&(gravity<=torque_high)).all(-1)
    a=torch.cat((eye.expand(n,4,3,3),cross_matrix(lever_world)/config['wrench_length_m']),2)
    a=(a*eligible[:,:,None,None]).transpose(1,2).reshape(n,6,12)
    b=desired_wrench.clone();b[:,3:]/=config['wrench_length_m']
    moment_map=jacobian_body.transpose(-1,-2)@rotation_world.transpose(-1,-2)[:,None]
    low=gravity-torque_high;high=gravity-torque_low
    reg=config['force_regularization']
    def project(f):
        for j in range(3):
            row=moment_map[:,:,j]
            value=(row*f).sum(-1)
            residual=value-value.clamp(low[:,:,j],high[:,:,j])
            f=f-(residual/row.square().sum(-1).clamp_min(1e-12))[...,None]*row
        return project_friction(f,eligible)
    # Start from the lower-cost projected unconstrained solution or warm start.
    dual=torch.linalg.solve(a@a.transpose(-1,-2)+reg*torch.eye(6,device=a.device,dtype=a.dtype),b[...,None])
    fresh=project((a.transpose(-1,-2)@dual).reshape(n,4,3))
    warm=project(previous)
    def objective(f):
        return ((a@f.reshape(n,12,1)).squeeze(-1)-b).square().sum(1)+reg*f.square().sum((1,2))
    f=torch.where((objective(fresh)<objective(warm))[:,None,None],fresh,warm)
    step=1./(a.square().sum((1,2))+reg)
    for _ in range(min(20,config['solver_iterations'])):
        error=(a@f.reshape(n,12,1)).squeeze(-1)-b
        gradient=(a.transpose(-1,-2)@error[...,None]).reshape(n,4,3)+reg*f
        f=project(f-step[:,None,None]*gradient)
    # Scaling toward f=0 preserves each friction cone and guarantees all the
    # linear actuator constraints simultaneously, without asserting convergence.
    def feasible(candidate):
        candidate=project_friction(candidate,eligible)
        change=rotate(moment_map,candidate)
        bound=torch.where(change>=0,(gravity-torque_low)/change.clamp_min(1e-12),
                          (torque_high-gravity)/(-change).clamp_min(1e-12))
        bound=torch.where(change.abs()<1e-10,torch.ones_like(bound),bound)
        fraction=bound.min(-1).values.clamp(0.,1.)
        return candidate*fraction[...,None]*eligible[...,None],fraction
    f,fraction=feasible(f)
    best_cost=objective(f)
    # Normalize the torque split by a length so both ADMM residuals have units
    # of force. Its box remains the original actuator-feasible interval.
    length=config['wrench_length_m']
    transform=torch.zeros((n,12,12),device=a.device,dtype=a.dtype)
    for leg in range(4):
        transform[:,3*leg:3*leg+3,3*leg:3*leg+3]=moment_map[:,leg]*eligible[:,leg,None,None]/length
    t_low=torch.where(eligible[...,None],low/length,0.).reshape(n,12)
    t_high=torch.where(eligible[...,None],high/length,0.).reshape(n,12)
    at=a.transpose(-1,-2);tt=transform.transpose(-1,-2)
    factor=torch.linalg.cholesky(at@a+(reg+1.)*torch.eye(12,device=a.device,dtype=a.dtype)+tt@transform)
    constant=(at@b[...,None]).squeeze(-1)
    z=f.reshape(n,12).clone();w=(transform@z[...,None]).squeeze(-1).clamp(t_low,t_high)
    u=torch.zeros_like(z);v=torch.zeros_like(w)
    for _ in range(config['solver_iterations']):
        rhs=constant+z-u+(tt@(w-v)[...,None]).squeeze(-1)
        x=torch.cholesky_solve(rhs[...,None],factor).squeeze(-1)
        tx=(transform@x[...,None]).squeeze(-1)
        z=project_friction((x+u).reshape(n,4,3),eligible).reshape(n,12)
        w=(tx+v).clamp(t_low,t_high)
        u+=x-z;v+=tx-w
        candidate,candidate_fraction=feasible(z.reshape(n,4,3))
        cost=objective(candidate);better=cost<best_cost
        f=torch.where(better[:,None,None],candidate,f)
        fraction=torch.where(better[:,None],candidate_fraction,fraction)
        best_cost=torch.minimum(cost,best_cost)
    torque=gravity-rotate(moment_map,f)
    actual_wrench=(a@f.reshape(n,12,1)).squeeze(-1)
    actual_wrench[:,3:]*=config['wrench_length_m']
    return dict(forces=f,torque=torque,eligible=eligible,planned_wrench=actual_wrench,
                wrench_error=actual_wrench-desired_wrench,final_force_scale=fraction)


class CentroidalSupportV1:
    def __init__(self,count,device,config,asset_sha256):
        self.config=settings(config);self.model=LoadedCentroidalModel(device)
        if asset_sha256!=self.model.record['asset_sha256']:raise ValueError('Support asset mismatch')
        self.count=count
        self.hit=torch.zeros(count,device=device,dtype=torch.bool)
        self.contact=torch.zeros((count,4),device=device,dtype=torch.bool)
        self.onset=torch.zeros((count,4),device=device)
        self.previous=torch.zeros((count,4,3),device=device)
        self.last_diagnostic={}

    def reset(self,ids=None):
        if ids is None:ids=slice(None)
        for name in ('hit','contact','onset','previous'):getattr(self,name)[ids]=0
        self.last_diagnostic={}

    @torch.inference_mode()
    def action(self,base,public,default,limits):
        if public.shape!=(self.count,794) or base.shape!=(self.count,18):
            raise ValueError('Require public794D and original18-joint actions')
        self.hit|=public[:,773]>.5
        contacts=(public[:,57:61]>.5)&self.hit[:,None]
        onset=contacts&(~self.contact)
        self.onset.copy_(torch.where(onset,public[:,775,None],self.onset))
        self.contact.copy_(contacts);self.previous*=contacts[...,None]
        ids=torch.where(contacts.any(1))[0]
        self.last_diagnostic={}
        if not len(ids):return base
        obs=public[ids];feet_contact=contacts[ids];cfg=self.config
        forward=obs[:,763:766];up=-obs[:,6:9]
        rotation=torch.stack((forward,torch.linalg.cross(up,forward),up),1)
        q=default[None]+obs[:,9:27];qd=obs[:,27:45]
        k=self.model.dynamics(q,qd,obs[:,:3],obs[:,3:6],rotation)
        root=obs.new_tensor([1.1,0.,1.8])-rotate(rotation,obs[:,760:763])
        center=root+rotate(rotation,k['center']);cv=rotate(rotation,k['center_v'])
        feet=root[:,None]+rotate(rotation[:,None],k['feet'])
        goal=(feet*feet_contact[...,None]).sum(1)/feet_contact.sum(1)[:,None]
        # Upright composite COM offset at the current arm/leg configuration.
        goal[:,2]+=cfg['root_height_m']+k['center'][:,2]
        acceleration=cfg['horizontal_gain_s2']*(goal-center)-cfg['horizontal_damping_s']*cv
        acceleration[:,2]=cfg['vertical_gain_s2']*(goal[:,2]-center[:,2])-cfg['vertical_damping_s']*cv[:,2]
        force=self.model.mass.sum()*(acceleration+obs.new_tensor([0.,0.,9.81]))
        error=torch.linalg.cross(rotation[:,:,2],obs.new_tensor([0.,0.,1.]).expand(len(ids),-1))
        moment=rotate(k['inertia_world'],cfg['upright_gain_s2']*error)-cfg['momentum_damping_s']*k['momentum_world']
        coordinates=self.model.leg_coordinates
        g=k['joint_gravity_torque'][:,coordinates]
        low=(LEG_KP*(limits[:12,0]-q[:,:12])-LEG_KD*qd[:,:12]).clamp_min(-LEG_TORQUE_LIMIT)
        high=(LEG_KP*(limits[:12,1]-q[:,:12])-LEG_KD*qd[:,:12]).clamp_max(LEG_TORQUE_LIMIT)
        fit=support_forces(feet-center[:,None],k['jacobian'],rotation,g,low[:,coordinates],high[:,coordinates],
            feet_contact,torch.cat((force,moment),1),self.previous[ids],cfg)
        self.previous[ids]=fit['forces']
        target=q[:,:12].clone()
        target[:,coordinates]=q[:,coordinates]+(fit['torque']+LEG_KD*qd[:,coordinates])/LEG_KP
        target=target.clamp(limits[:12,0],limits[:12,1])
        desired=limits[:12,0]+.5*(base[ids,:12]+1)*(limits[:12,1]-limits[:12,0])
        blend=((obs[:,775,None]-self.onset[ids])/cfg['blend_s']).clamp(0.,1.)
        blend=blend.square()*(3-2*blend)*fit['eligible']
        desired[:,coordinates]=torch.lerp(desired[:,coordinates],target[:,coordinates],blend[...,None])
        result=base.clone()
        enabled=torch.zeros_like(desired,dtype=torch.bool)
        enabled[:,coordinates]=(blend>0)[...,None]
        normalized=2*(desired-limits[:12,0])/(limits[:12,1]-limits[:12,0])-1
        result[ids,:12]=torch.where(enabled,normalized,base[ids,:12])
        self.last_diagnostic=dict(env_ids=ids,measured_com_m=center,desired_com_m=goal,
            observed_contact=feet_contact,
            measured_momentum=k['momentum_world'],requested_wrench=torch.cat((force,moment),1),
            **fit,source='Estimated support forces through original position PD; no applied external force')
        return result


class SupportAssistedController:
    def __init__(self,source,count,device,config,asset_sha256):
        self.source=source
        self.support=CentroidalSupportV1(count,device,config,asset_sha256)

    @property
    def reference_controller(self):return self.source.reference_controller
    @property
    def logs(self):return self.source.logs
    @logs.setter
    def logs(self,value):self.source.logs=value

    def reset(self,ids=None):
        self.source.reset(ids);self.support.reset(ids)

    def action(self,observation,default,limits):
        base=self.source.action(observation,default,limits)
        return self.support.action(base,observation['policy'],default,limits)
