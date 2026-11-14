"""Physical transition candidate; not yet an end-to-end policy learner.

Predicts 55D robot/ball state and foot contact under the recorded FINAL joint
targets. It does not invent history or predict a free scalar reward. Launch
parameters describe the known environment input, not future contact labels;
they must never be appended to the deployed actor observation.
"""
import torch
from torch import nn

from .supermimic_physical_dataset_v1 import encode_physical, decode_physical
from .supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1
from .supermimic_v1 import MLP


class PhysicalWorldV1(nn.Module):
    schema = 'physical55-final-target-dynamics-v1'

    def __init__(self, hidden=(1024,1024,1024), control_dt_s=.01, launch_s=2.):
        super().__init__()
        self.dt, self.launch_s = float(control_dt_s), float(launch_s)
        self.flight = ShuttleFreeFlightV1(control_dt_s=control_dt_s)
        self.body = MLP(103,hidden,59)
        self.register_buffer('input_mean',torch.zeros(103))
        self.register_buffer('input_scale',torch.ones(103))
        self.register_buffer('delta_mean',torch.zeros(55))
        self.register_buffer('delta_scale',torch.ones(55))
        self.register_buffer('normalization_ready',torch.tensor(False))

    def features(self,state,targets,previous_action,feet,present,elapsed,launch):
        return torch.cat((state,targets,previous_action,feet.to(state.dtype),
                          present.reshape(-1,1).to(state.dtype),elapsed.reshape(-1,1),launch),-1)

    def prior(self,state,present,elapsed,launch):
        p=decode_physical(state)
        # Constant-rate articulated motion is a prior, not a no-contact
        # acceleration certificate. Native contact effects remain supervised.
        root=p['root_position_w']+self.dt*p['root_link_velocity_w']
        q=p['root_quaternion_w'];w=p['root_angular_velocity_w']
        dq=torch.cat((-(w*q[:,1:]).sum(-1,keepdim=True),
                      q[:,:1]*w+torch.linalg.cross(w,q[:,1:])), -1)
        orientation=nn.functional.normalize(q+.5*self.dt*dq,dim=-1)
        launch_now=(~present.bool()) & (elapsed.flatten() >= self.launch_s-1e-6)
        active=present.bool() | launch_now
        ball_p=torch.where(launch_now[:,None],launch[:,:3],p['ball_position_w'])
        ball_v=torch.where(launch_now[:,None],launch[:,3:],p['ball_velocity_w'])
        flown_p,flown_v=self.flight(ball_p,ball_v)
        return encode_physical(dict(p,root_position_w=root,root_quaternion_w=orientation,
             joint_position=p['joint_position']+self.dt*p['joint_velocity'],
             ball_position_w=torch.where(active[:,None],flown_p,ball_p),
             ball_velocity_w=torch.where(active[:,None],flown_v,ball_v))),active

    @torch.no_grad()
    def set_normalization(self,features,prior,next_state):
        if features.shape[-1]!=103 or prior.shape!=next_state.shape or prior.shape[-1]!=55:
            raise ValueError('Invalid physical normalization arrays')
        self.input_mean.copy_(features.mean(0))
        self.input_scale.copy_(features.std(0,unbiased=False).clamp_min(.01))
        same=(prior[:,3:7]*next_state[:,3:7]).sum(-1,keepdim=True)>=0
        aligned=torch.cat((next_state[:,:3],torch.where(same,next_state[:,3:7],-next_state[:,3:7]),next_state[:,7:]),-1)
        delta=aligned-prior
        self.delta_mean.copy_(delta.mean(0))
        self.delta_scale.copy_(delta.std(0,unbiased=False).clamp_min(1e-3))
        self.normalization_ready.fill_(True)

    def forward(self,state,targets,previous_action,feet,present,elapsed,launch):
        features=self.features(state,targets,previous_action,feet,present,elapsed,launch)
        out=self.body((features-self.input_mean)/self.input_scale)
        prior,active=self.prior(state,present,elapsed,launch)
        predicted=prior+self.delta_mean+self.delta_scale*out[:,:55]
        p=decode_physical(predicted)
        p['root_quaternion_w']=nn.functional.normalize(p['root_quaternion_w'],dim=-1)
        # A parked ball is still an external schedule state, not something
        # the actor may move via residual control before the launch.
        old=decode_physical(state)
        p['ball_position_w']=torch.where(active[:,None],p['ball_position_w'],old['ball_position_w'])
        p['ball_velocity_w']=torch.where(active[:,None],p['ball_velocity_w'],old['ball_velocity_w'])
        return encode_physical(p),out[:,55:],active

    def state_loss(self,predicted,target):
        # Quaternion double cover must not create a spurious 2-unit penalty.
        same=(predicted[:,3:7]*target[:,3:7]).sum(-1,keepdim=True)>=0
        aligned=torch.cat((target[:,:3],torch.where(same,target[:,3:7],-target[:,3:7]),target[:,7:]),-1)
        return ((predicted-aligned)/self.delta_scale).square().mean()
