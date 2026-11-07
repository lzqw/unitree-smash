"""Read omitted head-ground normal forces without changing task scoring.

This adds a tensor contact view, not new geometry or a controller. One sample
per millisecond records instantaneous normal force, not a time average. A
finished native run and a separate unchanged-action audit are still required.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch


class NativeHeadGroundObservationV1:
    def __init__(self, env, output):
        if env.num_envs != 128 or abs(env.physics_dt-.0000625) > 1e-12:
            raise ValueError('Original128 and native physics clock required')
        if hasattr(env, '_head_ground_observer'):
            raise ValueError('Head observer already installed')
        self.output = Path(output)
        if any(self.output.glob('head_ground_observation*')):
            raise ValueError('Preserve earlier head telemetry')
        self.env = env; self.names = ['Head_lower', 'Head_upper']
        self.body_ids = [env.robot.body_names.index(name) for name in self.names]
        pattern = '/World/envs/env_*/Robot/Go2/(Head_lower|Head_upper)'
        ground = '/World/Ground/geometry/mesh'
        self.view = env.feet_contact._physics_sim_view.create_rigid_contact_view(
            pattern, filter_patterns=[ground], max_contact_data_count=4096)
        mapping = {}
        for row, path in enumerate(self.view.sensor_paths):
            match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/Go2/(Head_lower|Head_upper)', path)
            if match is None: raise ValueError('Unexpected head sensor path '+path)
            key = (int(match[1]), match[2])
            if key in mapping: raise ValueError('Duplicate head sensor row')
            mapping[key] = row
        expected = [(lane, name) for lane in range(128) for name in self.names]
        if set(mapping) != set(expected) or self.view.filter_count != 1:
            raise ValueError('Incomplete head/ground contact view')
        self.order = torch.tensor([mapping[key] for key in expected], device=env.device)
        self.capacity = 8000; self.count = 0; self.seen = -1; self.buffers = {}
        self.frames = np.empty(self.capacity, dtype=np.int64)
        self.original = env._observe_physics_frame
        self.metadata = dict(sensor_paths=list(self.view.sensor_paths),
            row_order_lane_major=self.order.cpu().tolist(), body_names=self.names,
            ground_filter=ground, physics_dt_s=env.physics_dt, sampling_interval_s=.001,
            force_semantics='Instantaneous filtered normal force at every 16th physics frame; no tangential force.',
            original_nonfoot_scoring_changed=False, action_changed=False,
            geometry_or_physical_parameters_changed=False, valid_expert_labels=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

        def observed(frame):
            result = self.original(frame)
            if frame > self.seen:
                self.seen = frame
                if 0 < frame <= 128000 and frame % 16 == 0:
                    if env._observation_physics_frame != frame:
                        raise ValueError('Original physics observation not current')
                    self.capture(frame)
            return result
        self.wrapper = observed
        env._observe_physics_frame = observed
        env._head_ground_observer = self

    def capture(self, frame):
        if self.count >= self.capacity: raise ValueError('Telemetry window exceeded')
        force = self.view.get_contact_force_matrix(self.env.physics_dt)
        if tuple(force.shape) != (256, 1, 3): raise ValueError('Unexpected head force shape')
        values = dict(head_ground_normal_force=force[self.order, 0].reshape(128, 2, 3),
            head_position_world=self.env.robot.data.body_link_pos_w[:, self.body_ids],
            episode_index=self.env.episode_index)
        if not self.buffers:
            self.buffers = {name: torch.empty((self.capacity,)+tuple(value.shape),
                dtype=value.dtype, device=value.device) for name, value in values.items()}
        for name, value in values.items(): self.buffers[name][self.count].copy_(value.detach())
        self.frames[self.count] = frame; self.count += 1
        if frame == 128000: self.save()

    def save(self):
        frames = self.frames[:self.count].copy()
        values = {name: data[:self.count].detach().cpu().numpy() for name, data in self.buffers.items()}
        if any(not np.isfinite(data).all() for data in values.values()):
            raise ValueError('Nonfinite head observations')
        path = self.output/'head_ground_observation.npz'; temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream: np.savez_compressed(stream, physics_frame=frames, **values)
        temp.replace(path)
        result = dict(self.metadata, samples=self.count,
            complete_window=bool(np.array_equal(frames, np.arange(16, 128001, 16))),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native_success_not_inferred_from_observer=True)
        path = self.output/'head_ground_observation.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n'); temp.replace(path)
        return result

    def detach(self):
        if self.env._observe_physics_frame is not self.wrapper:
            raise ValueError('Unexpected observer replacement')
        self.env._observe_physics_frame = self.original
        del self.env._head_ground_observer
