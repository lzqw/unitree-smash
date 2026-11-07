"""Passive paired-lane telemetry at the actual physics-frame boundary.

Command buffers are recorded separately from physical state. The simulator
target buffer is a submitted target, never a measurement of realized torque.
This instrument neither changes control nor supplies expert labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class NativeIsolationMicrotraceV1:
    def __init__(self, env, output, *, lanes=(3, 125), first_frame=89600, last_frame=92000):
        if (not 1 <= len(lanes) <= 4 or len(set(lanes)) != len(lanes)
                or any(type(i) is not int or not 0 <= i < env.num_envs for i in lanes)
                or not 0 < first_frame <= last_frame or last_frame-first_frame > 16000):
            raise ValueError('Require bounded unique lanes and physics window')
        if hasattr(env, '_isolation_micro_observer'):
            raise ValueError('Isolation recorder already attached')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob('isolation_microtrace*')):
            raise ValueError('Fresh recorder output required')
        self.env, self.lanes = env, list(lanes)
        self.first, self.last = first_frame, last_frame
        self.capacity = last_frame-first_frame+1
        self.frames = np.empty(self.capacity, dtype=np.int64)
        self.buffers, self.count, self.seen = {}, 0, -1
        self.joints = list(env.joint_ids)
        self.original = env._observe_physics_frame
        self.metadata = dict(lanes=self.lanes, first_frame=first_frame, last_frame=last_frame,
            physics_dt_s=float(env.physics_dt),
            joint_names=[env.robot.joint_names[i] for i in self.joints],
            foot_names=list(env.feet_contact.body_names),
            origins_world=env.scene.env_origins[self.lanes].detach().cpu().tolist(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            force_semantics='Contact normal force only; no tangential force or collision-partner attribution.',
            command_semantics='Environment request, articulation request, and PhysX submission buffers are distinct; none is realized torque.',
            action_changed=False, physics_changed=False, valid_expert_labels=False)

        def observed(frame):
            value = self.original(frame)
            if frame > self.seen:
                self.seen = frame
                if self.first <= frame <= self.last:
                    if env._observation_physics_frame != frame:
                        raise ValueError('Original observation did not reach this frame')
                    self.capture(frame)
            return value

        self.wrapper = observed
        env._observe_physics_frame = observed
        env._isolation_micro_observer = self

    def capture(self, frame):
        if self.count >= self.capacity:
            raise ValueError('Exceeded bounded telemetry window')
        e, ids, joints = self.env, self.lanes, self.joints
        robot, ball = e.robot, e.shuttle
        d = robot.data
        values = dict(
            joint_position=d.joint_pos[ids][:, joints],
            joint_velocity=d.joint_vel[ids][:, joints],
            env_position_request=e.targets[ids][:, joints],
            articulation_position_request=d.joint_pos_target[ids][:, joints],
            submitted_position_target=robot._joint_pos_target_sim[ids][:, joints],
            submitted_velocity_target=robot._joint_vel_target_sim[ids][:, joints],
            submitted_effort_target=robot._joint_effort_target_sim[ids][:, joints],
            external_robot_force_buffer=robot._external_force_b[ids],
            external_robot_torque_buffer=robot._external_torque_b[ids],
            external_robot_wrench_enabled=torch.as_tensor(robot.has_external_wrench),
            external_robot_wrench_global=torch.as_tensor(robot._use_global_wrench_frame),
            external_robot_wrench_positions_enabled=torch.as_tensor(robot.uses_external_wrench_positions),
            external_robot_wrench_positions=robot._external_wrench_positions_b[ids],
            root_link_position=d.root_link_pos_w[ids],
            root_link_quaternion=d.root_link_quat_w[ids],
            root_com_linear_velocity=d.root_com_lin_vel_w[ids],
            root_com_angular_velocity=d.root_com_ang_vel_w[ids],
            foot_normal_force=e.feet_contact.data.net_forces_w[ids],
            observed_foot_contact=e.observed_foot_contact[ids],
            ball_world_position=ball.data.root_pos_w[ids],
            ball_world_velocity=ball.data.root_lin_vel_w[ids],
            episode_index=e.episode_index[ids])
        if not self.buffers:
            self.buffers = {key: torch.empty((self.capacity,)+tuple(value.shape),
                dtype=value.dtype, device=value.device) for key, value in values.items()}
        for key, value in values.items():
            self.buffers[key][self.count].copy_(value.detach())
        self.frames[self.count] = frame
        self.count += 1
        if frame == self.last:
            self.save()

    def save(self):
        frames = self.frames[:self.count].copy()
        values = {key: value[:self.count].detach().cpu().numpy() for key,value in self.buffers.items()}
        path = self.output/'isolation_microtrace.npz'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, physics_frame=frames, **values)
        tmp.replace(path)
        result = dict(self.metadata, samples=self.count,
            complete_window=bool(np.array_equal(frames, np.arange(self.first, self.last+1))),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native_execution_not_inferred_from_recorder=True)
        out = self.output/'isolation_microtrace.json'
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(out)
        return result

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:
            raise ValueError('Another observer replaced this hook')
        self.env._observe_physics_frame = self.original
        del self.env._isolation_micro_observer
