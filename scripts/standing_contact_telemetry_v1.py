"""Read-only native contact telemetry, preserving the original outcome metric."""
import json
from pathlib import Path
import re

import numpy as np
import torch


def host(value):
    return value.detach().cpu().numpy().copy()


def path_identity(path):
    match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/(?:Go2/|Piper/)?([^/]+)', path)
    if match is None:
        raise ValueError('Unrecognized native sensor body path: ' + path)
    return int(match[1]), match[2]


def unpack_contacts(view, dt, selected_envs):
    """Keep native row identities; do not assume lexical/numeric env ordering."""
    buffers = [host(x) for x in view.get_contact_data(dt)]
    if len(buffers) != 6:
        raise ValueError('Installed native contact-data API must return six buffers')
    force, point, normal, separation, count, start = buffers
    rows = []
    for row, path in enumerate(view.sensor_paths):
        env_id, body = path_identity(path)
        if env_id not in selected_envs:
            continue
        for column in range(view.filter_count):
            n, offset = int(count[row, column]), int(start[row, column])
            if not n:
                continue
            if offset < 0 or offset+n > len(force):
                raise ValueError('Native contact buffer range is invalid')
            subset = slice(offset, offset+n)
            f = force[subset].reshape(n, 1)
            rows.append(dict(env_id=env_id, body=body, sensor_path=path,
                sensor_row=row, filter_column=column, count=n,
                normal_force_n=f[:, 0].tolist(), point_world_m=point[subset].tolist(),
                normal_world=normal[subset].tolist(), separation_m=separation[subset].reshape(n).tolist(),
                summed_normal_force_world_n=(f*normal[subset]).sum(0).tolist()))
    return rows


class StandingContactTelemetry:
    """Observe existing sensor buffers only; never alter commands or state."""
    def __init__(self, env, output):
        self.output = Path(output)
        self.frames, self.clock, self.body_forces, self.foot_forces = [], [], [], []
        self.events = []
        self.seen = np.zeros(env.num_envs, dtype=bool)
        self.sensor_names = ['body_ground_contact', 'arm_ground_contact', 'tool_ground_contact']
        self.metadata = dict(scope='Additional native reads only; original stability scoring unchanged',
            physics_dt_s=env.physics_dt, robot_body_names=list(env.robot.body_names),
            articulation_paths=list(env.robot.root_physx_view.prim_paths),
            original_metric='Norm per filtered body, summed over original nonfoot groups',
            sensors=[])
        for name, sensor in zip(self.sensor_names, env.ground_sensors):
            view = sensor.contact_physx_view
            paths = list(view.sensor_paths)
            body_paths = list(sensor.body_physx_view.prim_paths)
            self.metadata['sensors'].append(dict(name=name, sensor_paths=paths,
                body_view_paths=body_paths, filter_paths=list(view.filter_paths),
                body_names=list(sensor.body_names), sensor_count=view.sensor_count,
                filter_count=view.filter_count, maximum_contact_data_count=view.max_contact_data_count,
                actual_identities=[path_identity(p) for p in paths],
                reported_force_matrix_shape=list(sensor.data.force_matrix_w.shape)))
        self.metadata['foot_sensor_paths'] = list(env.feet_contact.contact_physx_view.sensor_paths)
        self._atomic('contact_telemetry_metadata.json', self.metadata)

    def _atomic(self, name, value):
        destination = self.output/name
        temporary = destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(destination)

    def on_physics(self, env):
        forces = [host(sensor.data.force_matrix_w) for sensor in env.ground_sensors]
        if any(x.ndim != 4 or x.shape[2:] != (1, 3) for x in forces):
            raise ValueError('Expected the original single global-ground filter')
        combined = np.concatenate([x[:, :, 0] for x in forces], axis=1)
        self.frames.append(env._jump_physics_frame)
        self.clock.append(host(env.jump_clock))
        self.body_forces.append(combined)
        self.foot_forces.append(host(env.feet_contact.data.net_forces_w))
        live = host(env.episode_index) == 0
        hit = live & (~self.seen) & (np.linalg.norm(combined, axis=-1).sum(1) > 1.)
        if not hit.any():
            return
        selected = set(np.flatnonzero(hit).tolist())
        self.seen |= hit
        data = env.robot.data
        event = dict(physics_frame=env._jump_physics_frame,
            first_event_env_ids=sorted(selected),
            body_link_pose_world_all_lanes=host(torch.cat((data.body_link_pos_w,
                data.body_link_quat_w), dim=-1)).tolist(),
            root_link_twist_world_all_lanes=host(torch.cat((data.root_link_lin_vel_w,
                data.root_ang_vel_w), dim=-1)).tolist(),
            joint_position_rad_all_lanes=host(data.joint_pos[:, env.joint_ids]).tolist(),
            joint_velocity_rad_s_all_lanes=host(data.joint_vel[:, env.joint_ids]).tolist(),
            sensors=[])
        for name, sensor in zip(self.sensor_names, env.ground_sensors):
            event['sensors'].append(dict(name=name,
                force_matrix_world_n=host(sensor.data.force_matrix_w).tolist(),
                net_force_world_n=host(sensor.data.net_forces_w).tolist(),
                native_patch_contacts=unpack_contacts(sensor.contact_physx_view, env.physics_dt,
                                                     set(range(env.num_envs)))))
        self.events.append(event)
        self._atomic('first_nonfoot_contact_details.json', self.events)

    def save(self):
        np.savez_compressed(self.output/'native_contact_telemetry.npz',
            physics_frame=np.array(self.frames), clock_s=np.stack(self.clock),
            nonfoot_ground_force_world_n=np.stack(self.body_forces),
            foot_net_force_world_n=np.stack(self.foot_forces))
        self._atomic('first_nonfoot_contact_details.json', self.events)
