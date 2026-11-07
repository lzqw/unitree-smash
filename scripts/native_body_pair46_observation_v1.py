"""Passive body-pair normal-force recording for the original TRAIN46 divergence experiment.

No collision, filtering, command or task-scoring property is changed. A paired
repeat audit must establish whether creating this view preserved execution.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeBodyPair46ObservationV1:
    def __init__(self, env, output):
        if env.num_envs!=128 or abs(env.physics_dt-.0000625)>1e-12:
            raise ValueError('Original native cohort and physics clock required')
        self.output=Path(output)
        if any(self.output.glob('body_pair_observation*')) or hasattr(env,'_body_pair_observer'):
            raise ValueError('Fresh observer required')
        self.env=env;self.lanes=[3,46]
        links=env.robot.root_physx_view.link_paths
        self.paths=[str(p) for lane in self.lanes for p in links[lane]]
        if len(self.paths)!=56 or len(set(self.paths))!=56:
            raise ValueError('Expected two complete 28-link articulations')
        for lane in self.lanes:
            if any(not str(p).startswith(f'/World/envs/env_{lane}/') for p in links[lane]):
                raise ValueError('Articulation path/lane mapping mismatch')
        self.filters=self.paths+['/World/Ground/geometry/mesh']
        self.view=env.feet_contact._physics_sim_view.create_rigid_contact_view(
            self.paths,filter_patterns=[self.filters.copy() for _ in self.paths],max_contact_data_count=8192)
        sensors=list(self.view.sensor_paths)
        if len(sensors)!=56 or set(sensors)!=set(self.paths) or self.view.filter_count!=57:
            raise ValueError('Incomplete pair-force view')
        self.order=torch.tensor([sensors.index(p) for p in self.paths],device=env.device)
        # Exact paths avoid wildcard partner ambiguity. Retain backend filter
        # paths for independent interpretation, not just requested patterns.
        self.actual_filters=self.view.filter_paths
        self.expected_frames=np.concatenate([np.arange(56800,61601,4),np.arange(75840,79041,4)])
        self.allowed=set(self.expected_frames.tolist());self.count=0;self.seen=-1
        self.frames=np.empty(len(self.expected_frames),dtype=np.int64);self.buffers={}
        self.original=env._observe_physics_frame
        def observed(frame):
            result=self.original(frame)
            if frame>self.seen:
                self.seen=frame
                if frame in self.allowed:self.capture(frame)
            return result
        self.wrapper=observed;env._observe_physics_frame=observed;env._body_pair_observer=self

    def capture(self,frame):
        e=self.env
        if self.count>=len(self.frames) or e._observation_physics_frame!=frame:
            raise ValueError('Invalid observation clock')
        forces=self.view.get_contact_force_matrix(e.physics_dt)
        if tuple(forces.shape)!=(56,57,3):raise ValueError('Unexpected pair-force tensor shape')
        values=dict(normal_force_world_n=forces[self.order],
            joint_position=e.robot.data.joint_pos[self.lanes][:,e.joint_ids],
            joint_velocity=e.robot.data.joint_vel[self.lanes][:,e.joint_ids],
            submitted_position_target=e.robot._joint_pos_target_sim[self.lanes][:,e.joint_ids],
            body_position_world=e.robot.data.body_link_pos_w[self.lanes],
            episode_index=e.episode_index[self.lanes])
        if not self.buffers:
            self.buffers={k:torch.empty((len(self.frames),)+tuple(v.shape),dtype=v.dtype,device=v.device) for k,v in values.items()}
        for k,v in values.items():self.buffers[k][self.count].copy_(v.detach())
        self.frames[self.count]=frame;self.count+=1

    def save(self):
        values={k:v[:self.count].detach().cpu().numpy() for k,v in self.buffers.items()}
        if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite telemetry')
        path=self.output/'body_pair_observation.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,physics_frame=self.frames[:self.count],**values)
        tmp.replace(path)
        result=dict(lanes=self.lanes,sensor_paths=self.paths,requested_filter_paths=self.filters,
            backend_filter_paths=self.actual_filters,body_names=list(self.env.robot.body_names),
            physics_dt_s=self.env.physics_dt,sampling_interval_s=.00025,samples=self.count,
            complete_window=bool(np.array_equal(self.frames[:self.count],self.expected_frames)),
            force_semantics='Instantaneous pair-filtered normal force, not tangential or time-averaged force.',
            view_creation_passivity_requires_paired_native_audit=True,
            action_changed=False,physical_parameters_changed=False,scoring_changed=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid_expert_labels=False)
        out=self.output/'body_pair_observation.json';tmp=out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
        return result

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:raise ValueError('Observer lifecycle mismatch')
        self.env._observe_physics_frame=self.original;del self.env._body_pair_observer
