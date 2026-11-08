"""Passive physics-step states around two declared foot-touchdown errors.

No new collision view, control, material, or task scoring. These are records,
not reset snapshots: hidden PhysX contact and solver caches are absent.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeFootImpactObserverV1:
    def __init__(self, env, output):
        if env.num_envs != 128 or abs(env.physics_dt-.0000625) > 1e-12:
            raise ValueError('Original cohort and physics clock required')
        self.env, self.output = env, Path(output)
        if hasattr(env, '_foot_impact_observer') or any(self.output.glob('foot_impact_observation*')):
            raise ValueError('Fresh observer required')
        if list(env.feet_contact.body_names) != ['FL_foot','FR_foot','RL_foot','RR_foot']:
            raise ValueError('Explicit foot ordering required')
        self.lanes = [85, 87]
        self.indices = torch.tensor(self.lanes, device=env.device)
        # Include the whole 298->299 and 303->304 control intervals and neighbors.
        self.first, self.last = 297*160, 306*160
        self.capacity = self.last-self.first+1
        self.count, self.seen = 0, -1
        self.buffers = {}
        self.frames = np.empty(self.capacity, dtype=np.int64)
        self.original = env._observe_physics_frame
        self.origins = env.scene.env_origins[self.indices].detach().cpu().numpy().copy()
        self.joint_names = [env.robot.joint_names[i] for i in env.joint_ids]

        def observe(frame):
            result = self.original(frame)
            if frame > self.seen:
                self.seen = frame
                if self.first <= frame <= self.last:
                    if env._observation_physics_frame != frame:
                        raise ValueError('Stale physics state')
                    self.capture(frame)
            return result
        self.wrapper = observe
        env._observe_physics_frame = observe
        env._foot_impact_observer = self

    def capture(self, frame):
        if frame != self.first+self.count or self.count >= self.capacity:
            raise ValueError('Missing or duplicated physics frame')
        e, d, ids = self.env, self.env.robot.data, self.indices
        values = dict(root_position_world=d.root_link_pos_w[ids],
            root_quaternion_wxyz=d.root_link_quat_w[ids],
            root_com_linear_velocity_world=d.root_com_lin_vel_w[ids],
            root_com_angular_velocity_world=d.root_com_ang_vel_w[ids],
            root_com_offset_body=d.body_com_pos_b[ids,0],
            joint_position=d.joint_pos[ids][:,e.joint_ids],
            joint_velocity=d.joint_vel[ids][:,e.joint_ids],
            previous_target=e.targets[ids][:,e.joint_ids],
            foot_force_world=e.feet_contact.data.net_forces_w[ids],
            observed_contacts=e.observed_foot_contact[ids],
            episode_index=e.episode_index[ids])
        if not self.buffers:
            self.bytes = self.capacity*sum(v.numel()*v.element_size() for v in values.values())
            if self.bytes > 4*1024**2:
                raise ValueError('Declared telemetry memory cap exceeded')
            self.buffers = {k: torch.empty((self.capacity,)+tuple(v.shape),
                device=v.device,dtype=v.dtype) for k,v in values.items()}
        for k,v in values.items():
            self.buffers[k][self.count].copy_(v.detach())
        self.frames[self.count] = frame
        self.count += 1

    def save(self):
        values = {k:v[:self.count].detach().cpu().numpy() for k,v in self.buffers.items()}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite physics observations')
        path = self.output/'foot_impact_observation.npz'
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,physics_frame=self.frames[:self.count],**values)
        temp.replace(path)
        result = dict(samples=self.count,complete_window=self.count==self.capacity,
            lanes=self.lanes,first_physics_frame=self.first,last_physics_frame=self.last,
            physics_dt_s=self.env.physics_dt,joint_names=self.joint_names,
            origins_world=self.origins.tolist(),allocated_bytes=getattr(self,'bytes',0),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            capture='After existing observation update, before the next physics integration.',
            force_semantics='Existing contact sensor net force; not assumed to include full tangential impulse.',
            target_semantics='Current target buffer; boundary alignment must be checked against the control bank.',
            physical_parameters_changed=False,controls_changed=False,scoring_changed=False,
            passive_replay_verified=False,native_reset_capable=False,valid_expert_labels=False)
        path = self.output/'foot_impact_observation.json'
        temp = path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
        return result

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:
            raise ValueError('Observer lifecycle changed')
        self.env._observe_physics_frame = self.original
        del self.env._foot_impact_observer
