"""Passive ground-filtered forces and foot-pair friction telemetry.

Existing control/scoring still consumes its original observations. Filtered
data is saved separately and must pass the parent trajectory replay audit.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from scripts.native_timed_impact_observer_v1 import NativeTimedImpactObserverV1


def aggregate_friction(forces, points, counts, starts, origin):
    """Sum valid pair entries only; uninitialized padding is never consumed."""
    counts, starts = counts.reshape(-1).to(torch.int64), starts.reshape(-1).to(torch.int64)
    slots = torch.arange(forces.shape[0], device=forces.device)
    mask = (slots[None] >= starts[:, None]) & (slots[None] < (starts + counts)[:, None])
    selected = torch.where(mask[..., None], forces[None], 0.)
    moment = torch.cross(points - origin, forces, dim=-1)
    moment = torch.where(mask[..., None], moment[None], 0.)
    invalid = ((counts < 0) | ((counts > 0) & ((starts < 0) | (starts + counts > len(forces))))).any()
    return selected.sum(1), moment.sum(1), invalid


class NativeGroundPairObserverV1(NativeTimedImpactObserverV1):
    def __init__(self, env, output):
        super().__init__(env, output)
        self.names = list(env.feet_contact.body_names)
        self.ground_path = '/World/Ground/geometry/mesh'
        self.ground_views, self.ground_orders = [], []
        self.pair_views, self.pair_filters = [], []
        self.body_ids = [env.robot.body_names.index(n) for n in self.names]
        sim = env.feet_contact._physics_sim_view
        for name in self.names:
            view = sim.create_rigid_contact_view('/World/envs/env_*/Robot/Go2/' + name,
                filter_patterns=[self.ground_path], max_contact_data_count=1024)
            mapping = {}
            for row, path in enumerate(view.sensor_paths):
                match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/Go2/' + name, path)
                if match is None or int(match[1]) in mapping:
                    raise ValueError('Unexpected or duplicate ground-view row')
                mapping[int(match[1])] = row
            if set(mapping) != set(range(128)) or view.filter_count != 1:
                raise ValueError('Incomplete single-foot ground view')
            self.ground_views.append(view)
            self.ground_orders.append(torch.tensor([mapping[i] for i in range(128)], device=env.device))
        for case in self.lanes:
            prefix = f'/World/envs/env_{case}/Robot/Go2/'
            for name in self.names:
                filters = [self.ground_path] + [prefix + other for other in self.names if other != name]
                view = sim.create_rigid_contact_view(prefix + name, filter_patterns=filters,
                    max_contact_data_count=64)
                if list(view.sensor_paths) != [prefix + name] or view.filter_count != 4:
                    raise ValueError('Wrong single-body pair view')
                self.pair_views.append(view)
                self.pair_filters.append(filters)
        self.filtered_buffers = {}
        self.ground_buffers = {}
        self.ground_frames = np.empty(800, np.int64)
        self.ground_count = 0
        self.ground_seen = -1
        previous = self.wrapper

        def observe(frame):
            result = previous(frame)
            if frame > self.ground_seen:
                self.ground_seen = frame
                if 0 < frame <= 128000 and frame % 160 == 0:
                    self.capture_ground(frame)
            return result
        self.wrapper = observe
        env._observe_physics_frame = observe

    @staticmethod
    def store(buffers, capacity, index, values):
        if not buffers:
            size = capacity * sum(v.numel() * v.element_size() for v in values.values())
            if size > 32 * 1024**2:
                raise ValueError('Filtered telemetry exceeds declared memory bound')
            buffers.update({k: torch.empty((capacity,) + tuple(v.shape), device=v.device, dtype=v.dtype)
                for k, v in values.items()})
        for k, v in values.items():
            buffers[k][index].copy_(v.detach())

    def capture(self, frame):
        super().capture(frame)
        normal, friction, moment, counts, invalid = [], [], [], [], []
        e = self.env
        for i, view in enumerate(self.pair_views):
            matrix = view.get_contact_force_matrix(e.physics_dt)
            if tuple(matrix.shape) != (1, 4, 3):
                raise ValueError('Wrong filtered force matrix shape')
            forces, points, count, start = view.get_friction_data(e.physics_dt)
            origin = e.robot.data.body_link_pos_w[self.lanes[i // 4], self.body_ids[i % 4]]
            force, torque, bad = aggregate_friction(forces, points, count, start, origin)
            normal.append(matrix[0].clone()); friction.append(force); moment.append(torque)
            counts.append(count.reshape(-1).to(torch.int64)); invalid.append(bad)
        values = dict(pair_normal_force_world_n=torch.stack(normal).reshape(2, 4, 4, 3),
            pair_friction_force_world_n=torch.stack(friction).reshape(2, 4, 4, 3),
            pair_friction_moment_about_foot_origin_nm=torch.stack(moment).reshape(2, 4, 4, 3),
            pair_friction_point_count=torch.stack(counts).reshape(2, 4, 4),
            pair_buffer_invalid=torch.stack(invalid).reshape(2, 4))
        self.store(self.filtered_buffers, self.capacity, self.count - 1, values)

    def capture_ground(self, frame):
        if frame != 160 * (self.ground_count + 1) or self.ground_count >= 800:
            raise ValueError('Wrong full-episode observation clock')
        forces = []
        for view, order in zip(self.ground_views, self.ground_orders):
            matrix = view.get_contact_force_matrix(self.env.physics_dt)
            if tuple(matrix.shape) != (128, 1, 3):
                raise ValueError('Wrong full-panel ground force shape')
            forces.append(matrix[order, 0])
        e = self.env
        values = dict(ground_normal_force_world_n=torch.stack(forces, 1),
            net_normal_force_world_n=e.feet_contact.data.net_forces_w,
            observed_contacts=e.observed_foot_contact,
            foot_position_world=e.robot.data.body_link_pos_w[:, self.body_ids],
            episode_index=e.episode_index)
        self.store(self.ground_buffers, 800, self.ground_count, values)
        self.ground_frames[self.ground_count] = frame
        self.ground_count += 1

    def save(self):
        parent = super().save()
        hashes = {}
        for stem, frames, buffers, count in [
                ('filtered_foot_pairs', self.frames, self.filtered_buffers, self.count),
                ('filtered_ground_panel', self.ground_frames, self.ground_buffers, self.ground_count)]:
            values = {k: v[:count].detach().cpu().numpy() for k, v in buffers.items()}
            if any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Nonfinite filtered contact data')
            if stem == 'filtered_foot_pairs' and values.get('pair_buffer_invalid', np.array([])).any():
                raise ValueError('Incomplete native friction buffers')
            path = self.output / (stem + '.npz')
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, physics_frame=frames[:count], **values)
            temp.replace(path)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = dict(high_rate_frames=self.count, panel_frames=self.ground_count,
            high_rate_complete=parent['complete_window'], panel_complete=self.ground_count == 800,
            lanes=self.lanes, feet=self.names, pair_filters_lane_then_foot=self.pair_filters,
            ground_filter=self.ground_path, data_sha256=hashes,
            force_semantics='Filtered normal and native friction are distinct fields; physics dt converts impulse to force.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            contact_or_scoring_changed=False, controller_changed=False, valid_expert_labels=False,
            passive_parent_replay_verified=False)
        path = self.output / 'filtered_contact_observation.json'
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(metadata, indent=2) + '\n'); temp.replace(path)
        return metadata
