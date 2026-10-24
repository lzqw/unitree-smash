"""Ball-aware execution of a learned reach controller, with causal prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from .full_racket_teacher_v4 import FullRacketTeacherV4
from .full_racket_geometry_v4 import FullRacketGeometryV4
from .initial_intercept import predict_initial_intercept

ROOT=Path(__file__).resolve().parents[2]


class FullRacketRallyControllerV4:
    def __init__(self,checkpoint,count,device):
        self.checkpoint=checkpoint
        self.teacher=FullRacketTeacherV4(checkpoint['backbone'],checkpoint['seed'],
            ROOT/'configs/g07_go2_walk_transfer.json',count,device)
        self.base=checkpoint['parameters'].to(device)
        self.parameters=self.base[None].repeat(count,1)
        self.event_time=torch.full((count,),2.75,device=device)
        self.planned=torch.zeros(count,dtype=torch.bool,device=device)
        self.physics=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
        model=FullRacketGeometryV4();q=np.array(checkpoint['seed']['joint_position_rad'])
        self.nominal_target=np.array(checkpoint['seed'].get('desired_ball_center_m',[1.10,0.,1.80]))
        self.jacobian=model.contact_jacobian(q,uv=checkpoint['seed'].get('string_uv_m',[0.,0.]),face=checkpoint['seed'].get('face_sign',1),base_position=(0,0,0))
        self.inverse=self.jacobian.T@np.linalg.inv(self.jacobian@self.jacobian.T+.04**2*np.eye(3))
        self.logs=[]
        self.target_offset_world=torch.zeros((count,3),device=device)

    def reset(self):
        self.teacher.reset();self.parameters.copy_(self.base[None])
        self.event_time.fill_(2.75);self.planned.zero_();self.logs=[]
        self.target_offset_world.zero_()

    @torch.inference_mode()
    def action(self,observation,default,limits):
        obs=observation['policy']
        ids=torch.where((~self.planned)&(obs[:,67]>.5)&(obs[:,775]>=2.0))[0]
        if len(ids):
            public=obs[ids].cpu().numpy()
            # The extra learned parameters choose a legal interception plane
            # and a timing residual. Both are fixed policy parameters, never
            # case labels or future simulator states.
            intercept=self.parameters[ids,24].cpu().numpy() if self.parameters.shape[1]>=25 else self.nominal_target[0]
            plan=predict_initial_intercept(public,[1.10,0,1.80],[1.,0,0],intercept,self.physics,
                                           dt=.001,max_seconds=1.5)
            arrival=public[:,775]+plan['arrival_time_s']
            # Correct the measured no-ball event offset, learned before this
            # episode, using only the current observation's ball prediction.
            calibration=self.checkpoint.get('event_calibration_s',2.75-self.checkpoint['result']['event_time_s'])
            residual=self.parameters[ids,25].cpu().numpy() if self.parameters.shape[1]>=26 else 0.
            calibrated=arrival+calibration+residual
            valid=plan['valid']&np.isfinite(calibrated)&(calibrated>=2.3)&(calibrated<=3.3)
            self.event_time[ids]=torch.as_tensor(np.where(valid,calibrated,2.75),device=obs.device,dtype=obs.dtype)
            delta=plan['position_m']-np.array(self.checkpoint['result']['virtual_ball_center_m'])
            correction=np.clip(np.nan_to_num(delta)@self.inverse.T,-.30,.30)
            correction[~valid]=0.
            if self.parameters.shape[1]<20:
                self.parameters[ids,8:14]+=torch.as_tensor(correction,device=obs.device,dtype=obs.dtype)
            else:
                target_delta=plan['position_m']-self.nominal_target
                target_delta[~valid]=0.
                self.target_offset_world[ids]=torch.as_tensor(target_delta,device=obs.device,dtype=obs.dtype)
            self.planned[ids]=True
            self.logs.append(dict(input='Current public794D observation, fixed known command and learned no-ball calibration only',
                env_ids=ids.cpu().tolist(),observation_clock_s=public[:,775].tolist(),
                used_public_observations=public.tolist(),calibrated_event_time_s=calibrated.tolist(),
                arm_joint_correction_rad=correction.tolist(),**{k:v.tolist() for k,v in plan.items()}))
        return self.teacher.action(observation,self.parameters,default,limits,event_time=self.event_time,
                                   target_offset_world=self.target_offset_world)
