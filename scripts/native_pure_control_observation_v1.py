"""Independent robot-body/ground normal forces for the pure-control trial.

No collision, task score or original policy observation is changed. 1 ms waveforms plus every-physics-step peak, impulse and contact duration are
recorded; no continuous-time guarantee between physical steps is claimed.
"""
import json
import hashlib
from pathlib import Path
import re
import numpy as np
import torch


class NativePureControlObservationV1:
    def __init__(self,env,output):
        from pxr import Usd,UsdPhysics
        from omni.usd import get_context
        self.env=env;self.output=Path(output);self.names=list(env.robot.body_names)
        if env.num_envs!=128 or abs(env.physics_dt-.0000625)>1e-12:raise ValueError('Original panel/clock required')
        stage=get_context().get_stage();root=stage.GetPrimAtPath('/World/envs/env_0/Robot')
        paths={}
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() in paths:raise ValueError('Ambiguous rigid body name')
                paths[prim.GetName()]=str(prim.GetPath())
        if not set(self.names).issubset(paths):raise ValueError('Incomplete original rigid body view')
        patterns=[paths[name].replace('/env_0/','/env_*/') for name in self.names]
        self.view=env.feet_contact._physics_sim_view.create_rigid_contact_view(patterns,
            filter_patterns=[['/World/Ground/geometry/mesh'] for _ in patterns],max_contact_data_count=131072)
        mapping={}
        for i,path in enumerate(self.view.sensor_paths):
            lane=int(re.search(r'/env_(\d+)/',path)[1]);name=path.rsplit('/',1)[1]
            key=(lane,name)
            if key in mapping:raise ValueError('Repeated body view row')
            mapping[key]=i
        expected=[(lane,name) for lane in range(128) for name in self.names]
        if set(mapping)!=set(expected) or self.view.filter_count!=1:raise ValueError('Incomplete ground pair view')
        self.order=torch.tensor([mapping[key] for key in expected],device=env.device)
        shape=(800,128,len(self.names))
        self.peaks=torch.zeros(shape,device=env.device)
        self.impulses=torch.zeros(shape+(3,),device=env.device)
        self.over1=torch.zeros(shape,dtype=torch.int32,device=env.device)
        self.first=torch.full(shape,-1,dtype=torch.int32,device=env.device)
        self.capacity=8000;self.count=0;self.seen=-1;self.frames=np.empty(8000,np.int64)
        self.forces=torch.empty((8000,128,len(self.names),3),device=env.device)
        self.episodes=torch.empty((8000,128),dtype=torch.int64,device=env.device)
        self.current=None;self.original=env._observe_physics_frame
        def observed(frame):
            result=self.original(frame)
            if frame>self.seen:
                self.seen=frame
                if 0<frame<=128000:
                    self.current=self.read()
                    index=(frame-1)//160;magnitude=self.current.norm(dim=-1)
                    self.peaks[index]=torch.maximum(self.peaks[index],magnitude)
                    self.impulses[index]+=self.current*env.physics_dt
                    hit=magnitude>1.
                    self.over1[index]+=hit.to(torch.int32)
                    self.first[index]=torch.where(hit&(self.first[index]<0),frame,self.first[index])
                if 0<frame<=128000 and frame%16==0:
                    if self.count>=self.capacity:raise ValueError('Contact storage bound')
                    self.forces[self.count].copy_(self.current)
                    self.episodes[self.count].copy_(env.episode_index)
                    self.frames[self.count]=frame;self.count+=1
            return result
        self.wrapper=observed;env._observe_physics_frame=observed

    def read(self):
        force=self.view.get_contact_force_matrix(self.env.physics_dt)
        if tuple(force.shape)!=(128*len(self.names),1,3):raise ValueError('Unexpected body ground matrix')
        return force[self.order,0].reshape(128,len(self.names),3)

    def save(self):
        path=self.output/'pure_control_ground_forces.npz';tmp=path.with_suffix('.tmp')
        data=self.forces[:self.count].cpu().numpy()
        if not np.isfinite(data).all():raise ValueError('Nonfinite ground telemetry')
        with tmp.open('wb') as f:np.savez_compressed(f,physics_frame=self.frames[:self.count],
            ground_normal_force_world_n=data,episode_index=self.episodes[:self.count].cpu().numpy(),
            interval_peak_normal_n=self.peaks.cpu().numpy(),interval_normal_impulse_ns=self.impulses.cpu().numpy(),
            interval_frames_over1n=self.over1.cpu().numpy(),interval_first_frame_over1n=self.first.cpu().numpy())
        tmp.replace(path)
        meta=dict(body_names=self.names,samples=self.count,sampling_period_s=.001,
            complete=bool(np.array_equal(self.frames[:self.count],np.arange(16,128001,16))),
            ground_filter='/World/Ground/geometry/mesh',normal_force_only=True,
            geometry_changed=False,historical_scoring_changed=False,
            physics_rate_aggregation_s=self.env.physics_dt,
            interval_aggregates_complete=self.seen==128000,
            absence_between_physics_steps_not_certified=True,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (self.output/'pure_control_ground_forces.json').write_text(json.dumps(meta,indent=2)+'\n')
        return meta

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:raise ValueError('Unexpected observer owner')
        self.env._observe_physics_frame=self.original
