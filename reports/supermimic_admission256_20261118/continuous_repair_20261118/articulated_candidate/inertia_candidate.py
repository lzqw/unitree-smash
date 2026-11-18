"""Differentiable authored-asset effective inertia prototype, not admitted.
World-aligned root linear/angular velocity followed by named18 joint velocity.
No contact force labels, future native state, controller or MPC are inputs.
Implicit PD, motor armature and active ground constraints are not modelled.
"""
import json
import torch
from torch import nn
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL,rotation,rotate

def skew(v):
 x,y,z=v.unbind(-1);o=torch.zeros_like(x)
 return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)

class ArticulatedInertiaCandidate(nn.Module):
 def __init__(self,asset_sha256,*,model_path=MODEL,dtype=torch.float64):
  super().__init__();self.geometry=FullBodyGeometryV1(asset_sha256,model_path=model_path,dtype=dtype)
  record=json.loads(model_path.read_text());inertias=[]
  for b in record['bodies']:
   R=rotation(torch.tensor(b['principal_axes_wxyz'],dtype=dtype))
   inertias.append(R@torch.diag(torch.tensor(b['diagonal_inertia_kg_m2'],dtype=dtype))@R.T)
  self.register_buffer('inertias',torch.stack(inertias))
  self.register_buffer('root_linear',torch.cat((torch.eye(3,dtype=dtype),torch.zeros(3,21,dtype=dtype)),1))
  self.register_buffer('root_angular',torch.cat((torch.zeros(3,3,dtype=dtype),torch.eye(3,dtype=dtype),torch.zeros(3,18,dtype=dtype)),1))
  self.register_buffer('joint_columns',torch.eye(24,dtype=dtype)[6:])
 def forward(self,joints,root_quaternion,point_local):
  g=self.geometry;n=len(joints);zero=joints.new_zeros(n,3)
  if point_local.shape!=(n,3):raise ValueError('Expected batch local racket contact point')
  bodies=g(joints,torch.zeros_like(joints),zero,root_quaternion,zero,zero)
  linear={g.root_id:self.root_linear.expand(n,-1,-1)};angular={g.root_id:self.root_angular.expand(n,-1,-1)}
  for i,(parent,child,joint) in enumerate(g.edges):
   Rparent=bodies['rotation'][:,parent];Rchild=bodies['rotation'][:,child]
   offset0=rotate(Rparent,g.parent_offset[i]);offset1=rotate(Rchild,g.child_offset[i])
   added=0. if joint is None else rotate(Rparent@g.parent_rotation[i],g.axis[i])[...,None]*self.joint_columns[joint]
   angular[child]=angular[parent]+added
   linear[child]=linear[parent]-skew(offset0)@angular[parent]+skew(offset1)@angular[child]
  Jv=torch.stack([linear[i] for i in range(len(g.names))],1);Jw=torch.stack([angular[i] for i in range(len(g.names))],1)
  R=bodies['rotation'];offset=rotate(R,g.local_com);Jcom=Jv-skew(offset)@Jw
  I=R@self.inertias@R.transpose(-1,-2)
  M=(g.mass[None,:,None,None]*(Jcom.transpose(-1,-2)@Jcom)+Jw.transpose(-1,-2)@I@Jw).sum(1)
  racket=g.ids['Racket'];arm=rotate(R[:,racket],point_local);J=Jv[:,racket]-skew(arm)@Jw[:,racket]
  A=J@torch.linalg.solve(M,J.transpose(-1,-2))
  return A,dict(mass_matrix=M,point_jacobian=J)


def sphere_impulse(relative,normal,racket_inverse_mass,*,mass,radius,restitution,friction):
 """One normal Newton restitution/Coulomb contact, no learned constants.
 Incoming relative velocity includes incoming sphere spin. Returns world impulse.
 Regime selection is discrete; within-regime derivatives remain available.
 """
 eye=torch.eye(3,device=relative.device,dtype=relative.dtype)
 nnT=normal[...,None]*normal[...,None,:]
 K=eye/mass+2.5/mass*(eye-nnT)+racket_inverse_mass
 vn=(relative*normal).sum(-1)
 stick=torch.linalg.solve(K,(-relative-restitution*vn.clamp_max(0)[...,None]*normal)[...,None]).squeeze(-1)
 pn=(stick*normal).sum(-1);pt=stick-pn[...,None]*normal
 tangent=relative-vn[...,None]*normal;speed=tangent.norm(dim=-1)
 direction=normal-friction*tangent/speed.clamp_min(1e-12)[...,None]
 denom=(normal*(K@direction[...,None]).squeeze(-1)).sum(-1)
 if not torch.isfinite(denom).all() or (denom<=0).any():raise ValueError('Invalid articulated impact effective mass')
 slide=-(1+restitution)*vn[...,None]/denom[...,None]*direction
 sticking=(pn>=0)&(pt.norm(dim=-1)<=friction*pn)
 impulse=torch.where(sticking[...,None],stick,slide)
 impulse=torch.where((vn<0)[...,None],impulse,torch.zeros_like(impulse))
 return impulse,dict(sticking=sticking,approaching=vn<0,relative_inverse_mass=K)
