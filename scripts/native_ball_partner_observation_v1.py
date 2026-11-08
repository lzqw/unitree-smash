"""Passive partner-force observation around measured lane divergence events.

Includes shuttles omitted by the earlier robot-only partner view. It neither
changes collision filtering nor controls the robot. Passivity needs replay.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeBallPartnerObservationV1:
    def __init__(self, env, output):
        if env.num_envs != 128 or abs(env.physics_dt-.0000625) > 1e-12:
            raise ValueError('Original 128-lane physics clock required')
        self.env=env;self.output=Path(output);self.lanes=[46,82,87,120,125]
        if any(self.output.glob('ball_partner_observation*')) or hasattr(env,'_ball_partner_observer'):
            raise ValueError('Fresh passive observer required')
        links=env.robot.root_physx_view.link_paths
        if len(links)!=128 or any(len(row)!=28 for row in links):
            raise ValueError('Expected robot mapping')
        self.filters=[]
        for lane,row in enumerate(links):
            if any(not str(p).startswith(f'/World/envs/env_{lane}/') for p in row):
                raise ValueError('Unexpected robot lane mapping')
            self.filters.extend(map(str,row))
            self.filters.append(f'/World/envs/env_{lane}/Shuttle')
        self.filters.append('/World/Ground/geometry/mesh')
        self.paths=[f'/World/envs/env_{lane}/Shuttle' for lane in self.lanes]+list(map(str,links[87]))
        self.sensor_lanes=self.lanes+[87]*28
        self.view=env.feet_contact._physics_sim_view.create_rigid_contact_view(self.paths,
            filter_patterns=[self.filters.copy() for _ in self.paths],max_contact_data_count=65536)
        sensors=list(self.view.sensor_paths);backend=self.view.filter_paths
        if set(sensors)!=set(self.paths) or len(sensors)!=33 or self.view.filter_count!=3713:
            raise ValueError('Incomplete shuttle/robot contact view')
        if backend==self.filters:self.backend_rows=1
        elif len(backend)==33 and all(row==self.filters for row in backend):self.backend_rows=33
        else:raise ValueError('Contact partner ordering changed')
        self.order=torch.tensor([sensors.index(p) for p in self.paths],device=env.device)
        self.windows=[[44960,45440],[77120,77600],[79200,79840]]
        self.expected=np.concatenate([np.arange(lo,hi+1) for lo,hi in self.windows])
        self.allowed=set(self.expected.tolist());self.frames=[];self.buffers={};self.seen=-1
        self.original=env._observe_physics_frame
        def observed(frame):
            result=self.original(frame)
            if frame>self.seen:
                self.seen=frame
                if frame in self.allowed:self.capture(frame)
            return result
        self.wrapper=observed;env._observe_physics_frame=observed;env._ball_partner_observer=self

    def capture(self,frame):
        e=self.env
        if len(self.frames)>=len(self.expected) or e._observation_physics_frame!=frame:
            raise ValueError('Wrong physical sample clock')
        force=self.view.get_contact_force_matrix(e.physics_dt)[self.order]
        if tuple(force.shape)!=(33,3713,3):raise ValueError('Unexpected force matrix')
        pairs=force[:,:3712].reshape(33,128,29,3)
        values,partners=torch.linalg.vector_norm(pairs,dim=-1).max(dim=-1)
        arrays=dict(maximum_individual_pair_normal_force_n=values,
            maximum_partner_body_index=partners.to(torch.uint8),ground_force_world_n=force[:,-1],
            shuttle_state=e.shuttle.data.root_state_w[self.lanes],
            joint_position=e.robot.data.joint_pos[self.lanes][:,e.joint_ids],
            joint_velocity=e.robot.data.joint_vel[self.lanes][:,e.joint_ids],
            submitted_position_target=e.robot._joint_pos_target_sim[self.lanes][:,e.joint_ids],
            episode_index=e.episode_index[self.lanes])
        if not self.buffers:
            required=sum(len(self.expected)*x.numel()*x.element_size() for x in arrays.values())
            if required>64*1024**2:raise ValueError('Telemetry exceeds declared 64 MiB bound')
            self.buffer_bytes=required
            self.buffers={k:torch.empty((len(self.expected),)+tuple(x.shape),dtype=x.dtype,device=x.device)
                          for k,x in arrays.items()}
        for key,value in arrays.items():self.buffers[key][len(self.frames)].copy_(value.detach())
        self.frames.append(frame)

    def save(self):
        arrays={k:x[:len(self.frames)].detach().cpu().numpy() for k,x in self.buffers.items()}
        if any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite force observation')
        path=self.output/'ball_partner_observation.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,physics_frame=self.frames,**arrays)
        tmp.replace(path)
        result=dict(scope=__doc__,lanes=self.lanes,sensor_paths=self.paths,sensor_lanes=self.sensor_lanes,
            partner_paths=self.filters,bodies_per_environment=29,backend_filter_paths_exact=True,
            backend_identical_filter_rows=self.backend_rows,windows_physics_frames=self.windows,
            samples=len(self.frames),complete_window=bool(np.array_equal(self.frames,self.expected)),
            physics_dt_s=self.env.physics_dt,buffer_bytes=getattr(self,'buffer_bytes',0),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            normal_forces_only=True,collision_filters_changed=False,observer_passivity_pending=True,
            valid_expert_labels=False)
        path=self.output/'ball_partner_observation.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:raise ValueError('Observer lifecycle changed')
        self.env._observe_physics_frame=self.original;del self.env._ball_partner_observer
