"""Passive force-partner coverage for the three implicated native lanes.

Retains the maximum individual body-pair normal-force norm per partner
environment, avoiding cancellation of opposing forces. Observations require
an exact controller replay audit before interpretation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeAllPartnerObservationV1:
    def __init__(self,env,output):
        if env.num_envs!=128 or abs(env.physics_dt-.0000625)>1e-12:
            raise ValueError('Original panel and clock required')
        self.env=env;self.output=Path(output);self.lanes=[3,46,125]
        if any(self.output.glob('all_partner_observation*')) or hasattr(env,'_all_partner_observer'):
            raise ValueError('Fresh observer required')
        links=env.robot.root_physx_view.link_paths
        if len(links)!=128 or any(len(row)!=28 for row in links):raise ValueError('Expected full 128x28 link mapping')
        self.filters=[]
        for lane,row in enumerate(links):
            if any(not str(p).startswith(f'/World/envs/env_{lane}/') for p in row):
                raise ValueError('Unexpected environment mapping')
            self.filters.extend(map(str,row))
        if len(set(self.filters))!=3584:raise ValueError('Nonunique partner mapping')
        self.filters.append('/World/Ground/geometry/mesh')
        self.paths=[str(p) for lane in self.lanes for p in links[lane]]
        self.view=env.feet_contact._physics_sim_view.create_rigid_contact_view(self.paths,
            filter_patterns=[self.filters.copy() for _ in self.paths],max_contact_data_count=8192)
        sensors=list(self.view.sensor_paths);backend=self.view.filter_paths
        if set(sensors)!=set(self.paths) or len(sensors)!=84 or self.view.filter_count!=3585:
            raise ValueError('Incomplete contact view')
        if backend==self.filters:self.backend_rows=1
        elif len(backend)==84 and all(row==self.filters for row in backend):self.backend_rows=84
        else:raise ValueError('Backend partner order differs from requested exact paths')
        self.order=torch.tensor([sensors.index(p) for p in self.paths],device=env.device)
        self.expected=np.r_[np.arange(77120,77601),np.arange(90400,90881)]
        self.allowed=set(self.expected.tolist());self.frames=[];self.buffers={};self.seen=-1
        self.original=env._observe_physics_frame
        def observed(frame):
            result=self.original(frame)
            if frame>self.seen:
                self.seen=frame
                if frame in self.allowed:self.capture(frame)
            return result
        self.wrapper=observed;env._observe_physics_frame=observed;env._all_partner_observer=self

    def capture(self,frame):
        e=self.env
        if len(self.frames)>=len(self.expected) or e._observation_physics_frame!=frame:
            raise ValueError('Wrong physical observation clock')
        force=self.view.get_contact_force_matrix(e.physics_dt)[self.order]
        if tuple(force.shape)!=(84,3585,3):raise ValueError('Unexpected force matrix')
        body_force=force[:,:3584].reshape(84,128,28,3)
        values,partner=torch.linalg.vector_norm(body_force,dim=-1).max(dim=-1)
        arrays=dict(maximum_individual_pair_normal_force_n=values,
            maximum_partner_body_index=partner.to(torch.uint8),ground_force_world_n=force[:,-1],
            joint_position=e.robot.data.joint_pos[self.lanes][:,e.joint_ids],
            joint_velocity=e.robot.data.joint_vel[self.lanes][:,e.joint_ids],
            submitted_position_target=e.robot._joint_pos_target_sim[self.lanes][:,e.joint_ids],
            episode_index=e.episode_index[self.lanes])
        if not self.buffers:
            required=sum(len(self.expected)*x.numel()*x.element_size() for x in arrays.values())
            if required>64*1024**2:raise ValueError('64 MiB telemetry bound exceeded')
            self.buffer_bytes=required
            self.buffers={k:torch.empty((len(self.expected),)+tuple(x.shape),dtype=x.dtype,device=x.device) for k,x in arrays.items()}
        for k,x in arrays.items():self.buffers[k][len(self.frames)].copy_(x.detach())
        self.frames.append(frame)

    def save(self):
        arrays={k:x[:len(self.frames)].detach().cpu().numpy() for k,x in self.buffers.items()}
        if any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite contact observation')
        path=self.output/'all_partner_observation.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,physics_frame=self.frames,**arrays)
        tmp.replace(path)
        report=dict(scope=__doc__,lanes=self.lanes,sensor_paths=self.paths,partner_paths=self.filters,
            backend_filter_paths_exact=True,backend_identical_filter_rows=self.backend_rows,
            samples=len(self.frames),complete_window=bool(np.array_equal(self.frames,self.expected)),
            windows_s=[[4.82,4.85],[5.65,5.68]],physics_dt_s=self.env.physics_dt,
            sampling_interval_s=self.env.physics_dt,buffer_bytes=getattr(self,'buffer_bytes',0),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            unchanged_collision_filters=True,normal_forces_only=True,observer_passivity_pending=True,
            valid_expert_labels=False)
        path=self.output/'all_partner_observation.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:raise ValueError('Observer lifecycle mismatch')
        self.env._observe_physics_frame=self.original;del self.env._all_partner_observer
