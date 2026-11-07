"""Passive selected-lane physics-frame telemetry for contact timing diagnosis.

Attach before rollout, after environment construction. Calls the original
observation hook exactly once and never changes actions, contact flags or
physics settings. This recorder has no controller or expert-label authority.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeContactMicrotraceV1:
    def __init__(self, env, output, *, lane, first_frame, last_frame):
        if (type(lane) is not int or not 0 <= lane < env.num_envs
                or type(first_frame) is not int or type(last_frame) is not int
                or not 0 < first_frame <= last_frame or last_frame-first_frame > 16000):
            raise ValueError('Bounded lane and physics-frame window required')
        if hasattr(env, '_dense_contact_observer'):
            raise ValueError('Recorder already attached')
        self.env = env
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob('contact_microtrace*')):
            raise ValueError('Fresh output required')
        self.lane = lane
        self.first = first_frame
        self.last = last_frame
        self.capacity = last_frame-first_frame+1
        self.frames = np.empty(self.capacity, dtype=np.int64)
        self.buffers = {}
        self.count = 0
        self.seen_frame = -1
        self.original = env._observe_physics_frame
        self.joint_ids = list(env.joint_ids)
        self.foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.foot_ids = [env.robot.body_names.index(name) for name in self.foot_names]
        # Do not silently reorder a flag whose order differs from its sensor.
        if list(env.feet_contact.body_names) != self.foot_names:
            raise ValueError('Contact flag/sensor foot order needs explicit review')
        self.metadata = dict(
            lane=lane, first_frame=first_frame, last_frame=last_frame,
            physics_dt_s=float(env.physics_dt), foot_names=self.foot_names,
            joint_names=[env.robot.joint_names[i] for i in self.joint_ids],
            env_origin_world=env.scene.env_origins[lane].detach().cpu().tolist(),
            force_semantics='Installed-version net_forces_w; normal force only. No measured tangential force.',
            sample_semantics='State after original _observe_physics_frame; target buffer at observation time, not proof of realized torque.',
            action_changed=False, physics_changed=False, valid_expert_labels=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

        def observed(frame):
            value = self.original(frame)
            if frame > self.seen_frame:
                self.seen_frame = frame
                if self.first <= frame <= self.last:
                    if env._observation_physics_frame != frame:
                        raise ValueError('Observation hook did not advance to sampled frame')
                    self.capture(frame)
            return value

        self.wrapper = observed
        env._observe_physics_frame = observed
        env._dense_contact_observer = self

    def capture(self, frame):
        if self.count >= self.capacity:
            raise ValueError('Capture exceeded declared window')
        e = self.env
        d = e.robot.data
        lane = self.lane
        samples = dict(
            joint_position_rad=d.joint_pos[lane, self.joint_ids],
            joint_velocity_rad_s=d.joint_vel[lane, self.joint_ids],
            joint_target_buffer_rad=e.targets[lane, self.joint_ids],
            root_link_position_world=d.root_link_pos_w[lane],
            root_link_quaternion_wxyz=d.root_link_quat_w[lane],
            root_com_linear_velocity_world=d.root_com_lin_vel_w[lane],
            root_com_angular_velocity_world=d.root_com_ang_vel_w[lane],
            root_com_position_body=d.body_com_pos_b[lane, 0],
            foot_position_world=d.body_pos_w[lane, self.foot_ids],
            foot_quaternion_wxyz=d.body_quat_w[lane, self.foot_ids],
            foot_normal_force_world_n=e.feet_contact.data.net_forces_w[lane],
            observed_foot_contact=e.observed_foot_contact[lane],
            episode_index=e.episode_index[lane])
        if not self.buffers:
            self.buffers = {name: torch.empty((self.capacity,)+tuple(value.shape),
                           dtype=value.dtype, device=value.device) for name,value in samples.items()}
        for name, value in samples.items():
            self.buffers[name][self.count].copy_(value.detach())
        self.frames[self.count] = frame
        self.count += 1
        # Persist the bounded window before later MPC calls can hit a process
        # timeout. This one-time host transfer changes no simulation time.
        if frame == self.last:
            self.save()

    def save(self):
        """May be called from finally; incomplete windows remain explicit."""
        values = {name: value[:self.count].detach().cpu().numpy()
                  for name, value in self.buffers.items()}
        frames = self.frames[:self.count].copy()
        values['physics_frame'] = frames
        path = self.output/'contact_microtrace.npz'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **values)
        tmp.replace(path)
        complete = np.array_equal(frames, np.arange(self.first, self.last+1))
        result = dict(self.metadata, samples=self.count, complete_window=bool(complete),
                      native_execution_not_inferred_from_recorder=True,
                      data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        out = self.output/'contact_microtrace.json'
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.replace(out)
        return result

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:
            raise ValueError('Another hook replaced this recorder')
        self.env._observe_physics_frame = self.original
        del self.env._dense_contact_observer
