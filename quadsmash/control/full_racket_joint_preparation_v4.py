"""Correct commanded joint preparation using full physical tool support.

Commands are based on current public state and source-URDF kinematics. This
module never writes simulated state, applies forces, or replaces native rules.
"""
import math
import numpy as np
import torch
from quadsmash.physics.full_racket_v4 import load_racket,parts,string_strips


def validate_joint_preparation(config):
    if config is None:return None
    values=dict(maximum_x_m=float(config.get('maximum_x_m',1.38)),
                lookahead_s=float(config.get('lookahead_s',.06)),
                damping=float(config.get('damping',.04)),
                maximum_joint_delta_rad=float(config.get('maximum_joint_delta_rad',.4)),
                iterations=config.get('iterations',4))
    if not all(math.isfinite(values[k]) for k in values if k!='iterations'):
        raise ValueError('Finite joint preparation settings required')
    if not 1.1<=values['maximum_x_m']<1.5 or not 0<=values['lookahead_s']<=.15:
        raise ValueError('Invalid own-side command planning bounds')
    if not .01<=values['damping']<=.1 or not .05<=values['maximum_joint_delta_rad']<=.5:
        raise ValueError('Invalid bounded preparation solve')
    if type(values['iterations']) is not int or not 1<=values['iterations']<=8:
        raise ValueError('Require1..8preparation iterations')
    return values


class FullRacketJointPreparation(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=validate_joint_preparation(config)
        centers,axes,lengths,radii,kinds,halves=[],[],[],[],[],[]
        def add(center,axis,length,radius,kind,half):
            centers.append(center);axes.append(axis);lengths.append(length/2)
            radii.append(radius);kinds.append(kind);halves.append(half)
        racket=load_racket()
        for part in parts(racket):
            if not part['collision']:continue
            if part['kind']=='string_prism':
                for strip in string_strips(racket):
                    add(strip['center'],[0,0,1],0,0,2,np.asarray(strip['size'])/2)
            elif part['kind']=='box':
                add(part['center'],[0,0,1],0,0,2,np.asarray(part['size'])/2)
            else:
                add(part['center'],part['axis'],part['length'],part['radius'],
                    0 if part['kind']=='capsule' else 1,[0,0,0])
        for name,value in dict(centers=centers,axes=axes,half_lengths=lengths,
                               radii=radii,halves=halves).items():
            self.register_buffer(name,torch.tensor(np.asarray(value),dtype=torch.float64))
        self.register_buffer('kinds',torch.tensor(kinds,dtype=torch.long))
        self.register_buffer('eye',torch.eye(6,dtype=torch.float64)[None])

    def local_support(self,direction):
        """An actual maximizing point of the union, including the clamp."""
        dot=direction@self.axes.T
        axial=dot.sign()[:,:,None]*self.half_lengths[None,:,None]*self.axes[None]
        radial=direction[:,None]-dot[:,:,None]*self.axes[None]
        radial=radial/torch.linalg.vector_norm(radial,dim=2,keepdim=True).clamp_min(1e-12)
        radial=torch.where((self.kinds==0)[None,:,None],direction[:,None],radial)
        rounded=self.centers[None]+axial+self.radii[None,:,None]*radial
        boxes=self.centers[None]+direction[:,None].sign()*self.halves[None]
        points=torch.where((self.kinds==2)[None,:,None],boxes,rounded)
        indices=(points*direction[:,None]).sum(2).argmax(1)
        return points[torch.arange(len(direction),device=direction.device),indices]

    def support_state(self,joints,observation,kinematics):
        center,_,jp,jn,rotation,axes,origins=kinematics(joints,return_pose=True)
        forward=observation[:,763:766]
        local_direction=torch.einsum('nji,nj->ni',rotation,forward)
        point=center+torch.einsum('nij,nj->ni',rotation,self.local_support(local_direction))
        x=1.10+((point-observation[:,760:763])*forward).sum(1)
        point_jacobian=torch.linalg.cross(axes,point[:,None]-origins).transpose(1,2)
        gradient=torch.einsum('ni,nij->nj',forward,point_jacobian)
        return x,gradient,jp,jn,point

    def forward(self,arm,observation,limits,kinematics):
        if self.config is None:return arm
        lower=limits[:,0];upper=limits[:,1]
        original=arm.clamp(lower,upper);joints=original.clone()
        forward=observation[:,763:766]
        observed_hit=observation[:,773]>.5
        for _ in range(self.config['iterations']):
            x,g,_,_,point=self.support_state(joints,observation,kinematics)
            point_velocity=observation[:,:3]+torch.linalg.cross(observation[:,3:6],point)
            margin=(self.config['lookahead_s']*(point_velocity*forward).sum(1).clamp_min(0.)).clamp_max(.15)
            excess=(x-self.config['maximum_x_m']+margin).clamp_min(0.)
            excess=torch.where(observed_hit,torch.zeros_like(excess),excess)
            # Minimum joint change for the one active planning inequality.
            # Other pose coordinates remain free within the same physical
            # limits; trying to hold all five task coordinates can prevent
            # retracting a commanded tool that is already too far forward.
            delta=(-excess[:,None]*g/(g.square().sum(1,keepdim=True)+self.config['damping']**2)).clamp(-.15,.15)
            bound=self.config['maximum_joint_delta_rad']
            candidate=(joints+delta).clamp(torch.maximum(lower,original-bound),torch.minimum(upper,original+bound))
            joints=torch.where((excess>0)[:,None],candidate,joints)
        # Preserve the original followthrough command after an observed hit.
        return torch.where(observed_hit[:,None],arm,joints)
