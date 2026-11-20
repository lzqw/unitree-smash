"""Bounded read-only native substep trace for one diagnostic control tick.

Records already-consumed PhysX frames, never advances simulation/task/history.
GPU buffers are copied to CPU only after the control tick. This is measured
native data, not interpolation or a contact-cache restoration mechanism.
"""
import torch


def packed_point_indices(counts, starts):
    counts = counts.long()
    pairs = torch.arange(counts.numel(), device=counts.device).repeat_interleave(counts.flatten())
    offsets = counts.flatten().cumsum(0) - counts.flatten()
    point_index = torch.arange(pairs.numel(), device=pairs.device) - offsets[pairs]
    indices = starts.flatten()[pairs].long() + point_index
    return pairs // counts.shape[1], pairs % counts.shape[1], indices


def pack_manifold_frame(contact_data):
    """Copy populated entries only, retaining the body/filter pair for every point."""
    forces, points, normals, separation, counts, starts = contact_data
    lane, filters, indices = packed_point_indices(counts, starts)
    return dict(lane=lane, filter_index=filters,
                normal_force_n=forces[indices].flatten(), point_w=points[indices],
                normal_w=normals[indices], separation_m=separation[indices].flatten())


class NativeSubstepTraceV1:
    schema = 'native-one-control-tick-substeps-v1'

    def __init__(self, env, *, start_frame=None):
        self.env = env
        self.original = env._consume_physics_frame
        self.had_instance_method = '_consume_physics_frame' in vars(env)
        self.saved_instance_method = vars(env).get('_consume_physics_frame')
        self.capacity = int(env.cfg.decimation) + 1
        self.count = 0
        self.buffers = {}
        self.manifold_frames = []
        self.active = True
        self.closed = False
        self.start_frame = int(env._sim_step_counter if start_frame is None else start_frame)
        self.last_recorded_frame = self.start_frame
        self.manual = start_frame is not None
        self._record(self.start_frame)
        if self.manual:
            return

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

    def observe_frame(self, frame):
        """Append one already advanced native frame in a controlled contact fixture."""
        if not self.manual or self.closed or frame != self.last_recorded_frame + 1:
            raise RuntimeError('Expected the next native fixture physics frame')
        self._record(frame)
        self.last_recorded_frame = frame

    @torch.no_grad()
    def _record(self, frame):
        if self.count >= self.capacity:
            raise RuntimeError('Native diagnostic substep capacity exceeded')
        env = self.env
        data = env.robot.data
        rid = env.racket_body_id
        contact = env.shuttle_contact.data
        force_matrix = contact.force_matrix_w
        contact_positions = contact.contact_pos_w
        view = getattr(env.shuttle_contact, "contact_physx_view", None)
        if view is not None:
            if force_matrix.shape[1] != 1:
                raise ValueError("Expected one shuttle sensor body per environment")
            self.manifold_frames.append(pack_manifold_frame(view.get_contact_data(dt=env.physics_dt)))
        sensors = getattr(getattr(env, 'scene', None), 'sensors', None)
        net_sensor = sensors.get('inverse_net') if hasattr(sensors, 'get') else None
        net_force_matrix = (None if net_sensor is None else
                            getattr(getattr(net_sensor, 'data', None), 'force_matrix_w', None))
        values = dict(
            physics_frame=torch.tensor(frame, device=env.device, dtype=torch.long),
            ball_clock=(torch.full_like(env.ball_clock, (frame - self.start_frame) * env.physics_dt)
                        if self.manual else env.ball_clock),
            shuttle_state_w=env.shuttle.data.root_state_w,
            racket_position_w=data.body_link_pos_w[:, rid],
            racket_quaternion_w=data.body_link_quat_w[:, rid],
            racket_linear_velocity_w=data.body_link_lin_vel_w[:, rid],
            racket_angular_velocity_w=data.body_link_ang_vel_w[:, rid],
            joint_position=data.joint_pos[:, env.joint_ids],
            joint_velocity=data.joint_vel[:, env.joint_ids],
            applied_torque=data.applied_torque[:, env.joint_ids],
            # Keep the raw pair slots so the pilot can distinguish racket,
            # ground and later-added filters without reconstructing a contact
            # cache.  The first pair remains the legacy racket fields used by
            # existing consumers.
            contact_force_matrix_w=force_matrix,
            contact_position_matrix_w=contact_positions,
            racket_force_w=force_matrix[:, 0, 0],
            ground_force_w=(force_matrix[:, 0, 1]
                            if force_matrix.shape[2] > 1 else torch.zeros_like(force_matrix[:, 0, 0])),
            net_force_matrix_w=(torch.zeros_like(force_matrix)
                                if net_force_matrix is None else net_force_matrix),
            racket_contact_position_w=contact_positions[:, 0, 0])
        if self.manual:
            if hasattr(contact, "net_forces_w"):
                values["shuttle_total_normal_force_w"] = contact.net_forces_w[:, 0]
            for field in ("joint_position", "joint_velocity", "applied_torque"):
                del values[field]
            if net_force_matrix is not None:
                values["net_force_matrix_w"] = net_force_matrix[:, :, -1:]
        if view is not None:
            friction, _, counts, starts = view.get_friction_data(dt=env.physics_dt)
            lane, _, indices = packed_point_indices(counts[:, :1], starts[:, :1])
            friction_on_racket_pair = torch.zeros_like(values["racket_force_w"])
            friction_on_racket_pair.index_add_(0, lane, friction[indices])
            values["racket_friction_force_w"] = friction_on_racket_pair
        for key, value in values.items():
            if key not in self.buffers:
                self.buffers[key] = torch.empty((self.capacity, *value.shape), device=value.device, dtype=value.dtype)
            self.buffers[key][self.count].copy_(value.detach())
        self.count += 1

    def close(self):
        if self.closed:
            return
        if self.manual:
            self.active = False
            self.closed = True
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
        body_names = getattr(getattr(self.env.robot, 'data', None), 'body_names', None)
        if body_names is None:
            body_names = getattr(self.env.robot, 'body_names', None)
        racket_body_name = None
        if body_names is not None and self.env.racket_body_id < len(body_names):
            racket_body_name = str(body_names[self.env.racket_body_id])
        filter_paths = getattr(getattr(getattr(self.env, 'shuttle_contact', None), 'cfg', None),
                               'filter_prim_paths_expr', None)
        contact_metadata = {
            'racket_body_id': int(self.env.racket_body_id),
            'racket_body_name': racket_body_name,
            'filter_prim_paths': None if filter_paths is None else [str(path) for path in filter_paths],
            'racket_filter_index': 0,
            'ground_filter_index': 1,
            'normal_source': 'physx_contact_data' if self.manifold_frames else 'unavailable',
            'friction_source': ('physx_filtered_racket_pair' if 'racket_friction_force_w' in tensors
                                else 'unavailable'),
        }
        return dict(schema=self.schema, count=self.count, physics_dt_s=float(self.env.physics_dt),
                    start_frame=self.start_frame, end_frame=int(frames[-1]), tensors=tensors,
                    contact_metadata=contact_metadata, manifold_frames=self.manifold_frames,
                    scope='observed consumed native frames; no object-tree snapshots or extra event updates')
