"""Isolated frozen-base lookahead driven only by predicted full-body states.

Never consumes recorded future actions. Each candidate gets its own controller
history; returned absolute targets are retained for identical prefix execution.
"""
from copy import deepcopy
import numpy as np
import torch
from quadsmash.control.initial_intercept import observed_initial_ball,propagate_ball_state


class CausalFrozenLegReferenceFactory:
    def __init__(self, teacher, parameters, observation, default, limits, event_time,
                 target_offset, lane, physics, dt_s=.01, inference_counts=None):
        self.inference_counts=inference_counts
        self.teacher=deepcopy(teacher)
        self.parameters=parameters.detach().clone()
        self.public=observation.detach().clone()
        self.default=default.detach().clone();self.limits=limits.detach().clone()
        self.event_time=event_time.detach().clone();self.target_offset=target_offset.detach().clone()
        self.lane=int(lane);self.physics=deepcopy(physics);self.dt_s=dt_s
        self.ball_position,self.ball_velocity=observed_initial_ball(
            self.public[self.lane:self.lane+1].cpu().numpy(),[1.1,0.,1.8],[1.,0.,0.])

    def __call__(self):
        teacher=deepcopy(self.teacher);public=self.public.clone();lane=self.lane
        from scripts.audit_teacher_action_sources_v1 import FrozenBaseInferenceAudit
        audit=FrozenBaseInferenceAudit(teacher)
        start=float(public[lane,775]);last_node=-1
        airborne_seen=bool(public[lane,772:775].sum()>.5)
        goal_time=float(public[lane,759]+public[lane,775])
        hit_seen=bool(public[lane,773]>.5)
        factory=self
        from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
        from scripts.predicted_observation_frame_v1 import predicted_frame, encode_final_command
        model=LoadedPinocchioModelV1()
        committed_node=-1
        def reference(state,node):
            nonlocal last_node,airborne_seen
            if node and committed_node != node-1:
                raise ValueError("Previous final18 command was not committed")
            if node!=last_node+1:raise ValueError('Causal leg reference requires sequential predicted states')
            last_node=node
            old=public[lane].clone()
            if node:public[lane,69:759]=old[:690]
            def put(sl,value):public[lane,sl]=torch.as_tensor(value,device=public.device,dtype=public.dtype)
            rotation=state.root_rotation_world
            ball,velocity=propagate_ball_state(factory.ball_position[0],factory.ball_velocity[0],node*factory.dt_s,factory.physics)
            put(slice(0,69),predicted_frame(model,state,self.default.cpu().numpy(),ball,velocity,
                bool(old[67]>.5)))
            target=rotation.T@(np.array([1.1,0.,1.8])-state.root_position_world)
            put(slice(760,763),target);put(slice(763,766),rotation.T@np.array([1.,0.,0.]))
            put(slice(766,769),-state.v[:3]-np.cross(state.v[3:6],target))
            public[lane,775]=start+node*factory.dt_s
            public[lane,759]=goal_time-public[lane,775]
            airborne=not state.contacts.any();airborne_seen|=airborne
            phase=(3 if hit_seen else 2) if airborne else (4 if airborne_seen else (1 if (rotation@state.v[:3])[2]>.05 else 0))
            public[lane,770:775]=0.;public[lane,770+phase]=1.
            before=dict(audit.counts)
            with torch.inference_mode():
                action=teacher.action({'policy':public},factory.parameters,factory.default,
                    factory.limits,event_time=factory.event_time,target_offset_world=factory.target_offset)
            if factory.inference_counts is not None:
                for name,count in audit.counts.items():
                    factory.inference_counts[name]=factory.inference_counts.get(name,0)+count-before[name]
            target=factory.limits[:,0]+.5*(action[lane].clamp(-1,1)+1)*(factory.limits[:,1]-factory.limits[:,0])
            reference.last_proposal_target=target.detach().cpu().numpy().copy()
            return target[:12].detach().cpu().numpy().copy()
        def commit(target,node):
            nonlocal committed_node
            if node!=last_node or committed_node==node:
                raise ValueError('Final command commit must follow one prediction')
            public[lane,776:794]=public.new_tensor(encode_final_command(target,factory.limits.cpu().numpy()))
            committed_node=node
        reference.commit_final_command=commit
        return reference
