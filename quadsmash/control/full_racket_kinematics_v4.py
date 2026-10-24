"""Batched source-URDF tool pose/Jacobian; no simulator or future-state input."""
import numpy as np
import torch
from .full_racket_geometry_v4 import FullRacketGeometryV4


class FullRacketKinematicsV4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model=FullRacketGeometryV4()
        for name,value in dict(arm_origin=model.arm_origin,
            offsets=np.stack([p for p,_ in model.joints]),
            rotations=np.stack([r for _,r in model.joints]),
            tool_offset=model.tool_offset,tool_rotation=model.tool_rotation).items():
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))

    def forward(self,joints,local_offset=None):
        count=len(joints)
        rotation=torch.eye(3,device=joints.device,dtype=joints.dtype)[None].expand(count,-1,-1)
        position=self.arm_origin[None].expand(count,-1)
        origins,axes=[],[]
        for i in range(6):
            position=position+torch.einsum('nij,j->ni',rotation,self.offsets[i])
            rotation=rotation@self.rotations[i]
            origins.append(position);axes.append(rotation[:,:,2])
            cosine,sine=joints[:,i].cos(),joints[:,i].sin()
            zero,one=torch.zeros_like(cosine),torch.ones_like(cosine)
            turn=torch.stack((cosine,-sine,zero,sine,cosine,zero,zero,zero,one),1).reshape(count,3,3)
            rotation=rotation@turn
        offset=self.tool_offset if local_offset is None else self.tool_offset+self.tool_rotation@local_offset
        center=position+torch.einsum('nij,j->ni',rotation,offset)
        normal=(rotation@self.tool_rotation)[:,:,0]
        axes=torch.stack(axes,1);origins=torch.stack(origins,1)
        jacobian=torch.linalg.cross(axes,center[:,None]-origins).transpose(1,2)
        normal_jacobian=torch.linalg.cross(axes,normal[:,None].expand(-1,6,-1)).transpose(1,2)
        return center,normal,jacobian,normal_jacobian
