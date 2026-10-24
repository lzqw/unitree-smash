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
                iterations=config.get('iterations',4),
                anticipate_measured_motion=config.get('anticipate_measured_motion',False),
                protect_post_hit_net_volume=config.get('protect_post_hit_net_volume',False),
                maximum_predictive_margin_m=float(config.get('maximum_predictive_margin_m',.15)),
                post_hit_height_margin_m=float(config.get('post_hit_height_margin_m',.04)))
    flags=['anticipate_measured_motion','protect_post_hit_net_volume']
    if any(type(values[k]) is not bool for k in flags):raise ValueError('Preparation switches must be boolean')
    if not all(math.isfinite(values[k]) for k in values if k not in ['iterations',*flags]):
        raise ValueError('Finite joint preparation settings required')
    if not 1.1<=values['maximum_x_m']<1.5 or not 0<=values['lookahead_s']<=.15:
        raise ValueError('Invalid own-side command planning bounds')
    if not .01<=values['damping']<=.1 or not .05<=values['maximum_joint_delta_rad']<=.5:
        raise ValueError('Invalid bounded preparation solve')
    if type(values['iterations']) is not int or not 1<=values['iterations']<=8:
        raise ValueError('Require1..8preparation iterations')
    if not .05<=values['maximum_predictive_margin_m']<=.30 or not .01<=values['post_hit_height_margin_m']<=.10:
        raise ValueError('Invalid predictive preparation margins')
    return values


class FullRacketJointPreparation(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=validate_joint_preparation(config)
        centers,axes,lengths,radii,kinds,halves,eligible=[],[],[],[],[],[],[]
        def add(center,axis,length,radius,kind,half,followthrough=True):
            centers.append(center);axes.append(axis);lengths.append(length/2)
            radii.append(radius);kinds.append(kind);halves.append(half);eligible.append(followthrough)
        racket=load_racket()
        for part in parts(racket):
            if not part['collision']:continue
            if part['kind']=='string_prism':
                for strip in string_strips(racket):
                    add(strip['center'],[0,0,1],0,0,2,np.asarray(strip['size'])/2)
            elif part['kind']=='box':
                add(part['center'],[0,0,1],0,0,2,np.asarray(part['size'])/2,part['name']!='Clamp')
            else:
                add(part['center'],part['axis'],part['length'],part['radius'],
                    0 if part['kind']=='capsule' else 1,[0,0,0])
        for name,value in dict(centers=centers,axes=axes,half_lengths=lengths,
                               radii=radii,halves=halves).items():
            self.register_buffer(name,torch.tensor(np.asarray(value),dtype=torch.float64))
        self.register_buffer('kinds',torch.tensor(kinds,dtype=torch.long))
        self.register_buffer('followthrough_eligible',torch.tensor(eligible,dtype=torch.bool))
        self.register_buffer('eye',torch.eye(6,dtype=torch.float64)[None])
        from quadsmash.physics.court_v4 import load_court
        court=load_court();self.net_top=max(court['net_center_top_m'],court['net_side_top_m'])

    def part_support(self,direction):
        """Actual support points of every declared physical primitive."""
        dot=direction@self.axes.T
        axial=dot.sign()[:,:,None]*self.half_lengths[None,:,None]*self.axes[None]
        radial=direction[:,None]-dot[:,:,None]*self.axes[None]
        radial=radial/torch.linalg.vector_norm(radial,dim=2,keepdim=True).clamp_min(1e-12)
        radial=torch.where((self.kinds==0)[None,:,None],direction[:,None],radial)
        rounded=self.centers[None]+axial+self.radii[None,:,None]*radial
        boxes=self.centers[None]+direction[:,None].sign()*self.halves[None]
        return torch.where((self.kinds==2)[None,:,None],boxes,rounded)

    def local_support(self,direction,allowed=None):
        """An actual maximizing point of the union, including the clamp."""
        points=self.part_support(direction)
        values=(points*direction[:,None]).sum(2)
        if allowed is not None:values=torch.where(allowed,values,torch.full_like(values,-float('inf')))
        indices=values.argmax(1)
        return points[torch.arange(len(direction),device=direction.device),indices]

    def support_state(self,joints,observation,kinematics,post_hit_mask=None,observed_center=False):
        center,_,jp,jn,rotation,axes,origins=kinematics(joints,return_pose=True)
        if observed_center:center=observation[:,45:48]
        forward=observation[:,763:766]
        local_direction=torch.einsum('nji,nj->ni',rotation,forward)
        allowed=None
        if post_hit_mask is not None:
            up=-observation[:,6:9]
            local_up=torch.einsum('nji,nj->ni',rotation,up)
            bottom=self.part_support(-local_up)
            center_z=1.80+((center-observation[:,760:763])*up).sum(1)
            minimum_z=center_z[:,None]+(bottom*local_up[:,None]).sum(2)
            # During a legal followthrough, physical racket components that
            # remain above the net may pass. The fixture always stays own-side.
            bottom_body=center[:,None]+torch.einsum('nij,nkj->nki',rotation,bottom)
            bottom_velocity=observation[:,None,:3]+torch.linalg.cross(observation[:,None,3:6].expand_as(bottom_body),bottom_body)
            bottom_velocity+=observation[:,None,51:54]+torch.linalg.cross(
                observation[:,None,54:57].expand_as(bottom_body),bottom_body-observation[:,None,45:48])
            descent=(self.config['lookahead_s']*(bottom_velocity*up[:,None]).sum(2).clamp_max(0.)).clamp_min(-.15)
            above=minimum_z+descent>self.net_top+self.config['post_hit_height_margin_m']
            exempt=post_hit_mask[:,None]&above&self.followthrough_eligible[None]
            allowed=~exempt
        point=center+torch.einsum('nij,nj->ni',rotation,self.local_support(local_direction,allowed))
        x=1.10+((point-observation[:,760:763])*forward).sum(1)
        point_jacobian=torch.linalg.cross(axes,point[:,None]-origins).transpose(1,2)
        gradient=torch.einsum('ni,nij->nj',forward,point_jacobian)
        return x,gradient,jp,jn,point

    def forward(self,arm,observation,limits,kinematics,actual_joints=None):
        if self.config is None:return arm
        lower=limits[:,0];upper=limits[:,1]
        original=arm.clamp(lower,upper);joints=original.clone()
        forward=observation[:,763:766]
        observed_hit=observation[:,773]>.5
        post_hit_mask=observed_hit if self.config['protect_post_hit_net_volume'] else None
        observed_x=observed_vx=original_x=None
        if self.config['anticipate_measured_motion']:
            if actual_joints is None:raise ValueError('Measured joint positions required for tool motion prediction')
            observed_x,_,_,_,observed_point=self.support_state(actual_joints,observation,kinematics,post_hit_mask,True)
            observed_velocity=observation[:,:3]+torch.linalg.cross(observation[:,3:6],observed_point)
            observed_velocity+=observation[:,51:54]+torch.linalg.cross(
                observation[:,54:57],observed_point-observation[:,45:48])
            observed_vx=(observed_velocity*forward).sum(1)
            original_x=self.support_state(original,observation,kinematics,post_hit_mask)[0]
        for _ in range(self.config['iterations']):
            x,g,_,_,point=self.support_state(joints,observation,kinematics,post_hit_mask)
            point_velocity=observation[:,:3]+torch.linalg.cross(observation[:,3:6],point)
            vx=(point_velocity*forward).sum(1)
            if observed_vx is not None:vx=torch.maximum(vx,observed_vx)
            margin=(self.config['lookahead_s']*vx.clamp_min(0.)).clamp_max(self.config['maximum_predictive_margin_m'])
            excess=(x-self.config['maximum_x_m']+margin).clamp_min(0.)
            if observed_x is not None:
                observed_margin=(self.config['lookahead_s']*observed_vx.clamp_min(0.)).clamp_max(self.config['maximum_predictive_margin_m'])
                # Retracting the command reduces this observed-state demand;
                # do not apply the same fixed correction once per iteration.
                actual_demand=(observed_x+observed_margin-self.config['maximum_x_m']+x-original_x).clamp_min(0.)
                excess=torch.maximum(excess,actual_demand)
            if not self.config['protect_post_hit_net_volume']:
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
        if not self.config['protect_post_hit_net_volume']:
            return torch.where(observed_hit[:,None],arm,joints)
        return joints
