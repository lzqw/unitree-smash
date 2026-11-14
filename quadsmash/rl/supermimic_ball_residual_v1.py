"""Geometry-conditioned non-free-flight residual, separate from robot motion.

Its gate predicts a measurable deviation from the free-flight prior, not a
certified PhysX contact. Task event rewards must use separate native labels.
"""
import torch
from torch import nn

from .supermimic_physical_dataset_v1 import decode_physical
from .supermimic_v1 import MLP


def ball_geometry_features(state,geometry):
    p=decode_physical(state)
    body=geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],
        p['root_quaternion_w'],p['root_link_velocity_w'],p['root_angular_velocity_w'])
    i=geometry.ids['Racket'];r=body['rotation'][:,i];center=body['position'][:,i]
    delta=p['ball_position_w']-center
    local=(r.transpose(-1,-2)@delta.unsqueeze(-1)).squeeze(-1)
    surface=body['velocity'][:,i]+torch.linalg.cross(body['angular_velocity'][:,i],delta)
    relative=(r.transpose(-1,-2)@(p['ball_velocity_w']-surface).unsqueeze(-1)).squeeze(-1)
    net=p['ball_position_w']-state.new_tensor([1.5,0.,1.16])
    return torch.cat((p['ball_position_w'],p['ball_velocity_w'],local,relative,r[:,:,0],
        body['velocity'][:,i],body['angular_velocity'][:,i],p['ball_position_w'][:,2:3]-.025,net),-1)


class BallResidualV1(nn.Module):
    schema='geometry-conditioned-ball-residual-v1'

    def __init__(self,gate_hidden=(128,128),response_hidden=(512,512)):
        super().__init__()
        self.gate=MLP(25,gate_hidden,1);self.response=MLP(25,response_hidden,6)
        self.register_buffer('mean',torch.zeros(25));self.register_buffer('scale',torch.ones(25))
        self.register_buffer('residual_scale',torch.ones(6))

    @torch.no_grad()
    def normalize_from_training(self,features,residual):
        self.mean.copy_(features.mean(0));self.scale.copy_(features.std(0,unbiased=False).clamp_min(.01))
        self.residual_scale.copy_(residual.std(0,unbiased=False).clamp_min(.001))

    def normalized(self,features):return (features-self.mean)/self.scale

    def forward(self,features,present):
        x=self.normalized(features)
        probability=self.gate(x).sigmoid()
        delta=self.response(x)*self.residual_scale
        return present[:,None].to(x.dtype)*probability*delta,probability[:,0]
