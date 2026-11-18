"""Bounded read-only native substep trace for one diagnostic control tick.

Records already-consumed PhysX frames, never advances simulation/task/history.
GPU buffers are copied to CPU only after the control tick. This is measured
native data, not interpolation or a contact-cache restoration mechanism.
"""
import torch


class NativeSubstepTraceV1:
    schema = 'native-one-control-tick-substeps-v1'

    def __init__(self, env):
        self.env = env
        self.original = env._consume_physics_frame
        self.had_instance_method = '_consume_physics_frame' in vars(env)
        self.saved_instance_method = vars(env).get('_consume_physics_frame')
        self.capacity = int(env.cfg.decimation) + 1
        self.count = 0
        self.buffers = {}
        self.active = True
        self.closed = False
        self.start_frame = int(env._sim_step_counter)
        self.last_recorded_frame = self.start_frame
        self._record(self.start_frame)

        def wrapped(frame):
            before = int(env._last_physics_frame)
            result = self.original(frame)
            after = int(env._last_physics_frame)
            if self.active and after > before and after > self.last_recorded_frame:
                if after != int(frame):
                    raise RuntimeError('Native consumed-frame identity mismatch')
                self._record(after)
                self.last_recorded_frame = after
            return result
        self.wrapper = wrapped
        env._consume_physics_frame = wrapped

    @torch.no_grad()
    def _record(self, frame):
        if self.count >= self.capacity:
            raise RuntimeError('Native diagnostic substep capacity exceeded')
        env = self.env
        data = env.robot.data
        rid = env.racket_body_id
        contact = env.shuttle_contact.data
        values = dict(
            physics_frame=torch.tensor(frame, device=env.device, dtype=torch.long),
            ball_clock=env.ball_clock,
            shuttle_state_w=env.shuttle.data.root_state_w,
            racket_position_w=data.body_link_pos_w[:, rid],
            racket_quaternion_w=data.body_link_quat_w[:, rid],
            racket_linear_velocity_w=data.body_link_lin_vel_w[:, rid],
            racket_angular_velocity_w=data.body_link_ang_vel_w[:, rid],
            joint_position=data.joint_pos[:, env.joint_ids],
            joint_velocity=data.joint_vel[:, env.joint_ids],
            applied_torque=data.applied_torque[:, env.joint_ids],
            racket_force_w=contact.force_matrix_w[:, 0, 0],
            racket_contact_position_w=contact.contact_pos_w[:, 0, 0])
        for key, value in values.items():
            if key not in self.buffers:
                self.buffers[key] = torch.empty((self.capacity, *value.shape), device=value.device, dtype=value.dtype)
            self.buffers[key][self.count].copy_(value.detach())
        self.count += 1

    def close(self):
        if self.closed:
            return
        if self.env._consume_physics_frame is not self.wrapper:
            raise RuntimeError('Native trace hook replaced unexpectedly')
        if self.had_instance_method:
            self.env._consume_physics_frame = self.saved_instance_method
        else:
            delattr(self.env, '_consume_physics_frame')
        self.active = False
        self.closed = True

    def finish(self):
        self.close()
        if self.count != self.capacity:
            raise RuntimeError(f'Incomplete native substep tick: {self.count}/{self.capacity}')
        frames = self.buffers['physics_frame'][:self.count]
        if not torch.equal(frames, torch.arange(self.start_frame, self.start_frame+self.capacity, device=frames.device)):
            raise RuntimeError('Missing, duplicated or nonconsecutive native substeps')
        tensors = {k: v[:self.count].clone() for k, v in self.buffers.items()}
        # Undefined contact positions are retained; zero is never substituted.
        tensors['racket_contact_position_valid'] = torch.isfinite(tensors['racket_contact_position_w']).all(-1)
        return dict(schema=self.schema, count=self.count, physics_dt_s=float(self.env.physics_dt),
                    start_frame=self.start_frame, end_frame=int(frames[-1]), tensors=tensors,
                    scope='observed consumed native frames; no object-tree snapshots or extra event updates')
