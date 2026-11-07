"""Causal history from recorded first nonfoot force events, never outcome labels.

Only first episodes starting at global physics frame zero are supported. The
input is a first-event archive, not a continuous force record. After its first
event a lane remains marked until the rally resets. No current-force magnitude
is inferred for subsequent times.
"""
import json
from pathlib import Path

import numpy as np


class RecordedNonfootContactHistory:
    def __init__(self, source, count=128, policy_dt=.01):
        self.source = Path(source)
        metadata = json.loads((self.source / 'contact_telemetry_metadata.json').read_text())
        self.dt = float(metadata['physics_dt_s'])
        manifest = json.loads((self.source / 'fixed_policy_traces/manifest.json').read_text())
        entries = sorted(manifest['episodes'], key=lambda e: e['env_id'])
        if [e['env_id'] for e in entries] != list(range(count)):
            raise ValueError('Require all original first-episode command lengths')
        end_frame = np.array([e['samples']*policy_dt/self.dt for e in entries])
        events = json.loads((self.source / 'first_nonfoot_contact_details.json').read_text())
        self.first_frame = np.full(count, np.inf)
        previous = -1
        for event in events:
            frame = int(event['physics_frame'])
            if frame <= previous:
                raise ValueError('First-event frames must increase')
            previous = frame
            forces = [np.asarray(sensor['force_matrix_world_n']) for sensor in event['sensors']]
            if (len(forces) != 3 or any(f.ndim != 4 or f.shape[0] != count
                    or f.shape[2:] != (1, 3) or not np.isfinite(f).all() for f in forces)):
                raise ValueError('Original three nonfoot groups with global-ground filters required')
            magnitude = sum(np.linalg.norm(f, axis=-1).sum((1, 2)) for f in forces)
            newly_observed = np.flatnonzero(np.isinf(self.first_frame) & (frame < end_frame)
                                           & (magnitude > 1.))
            if newly_observed.tolist() != event['first_event_env_ids']:
                raise ValueError('Raw force groups do not reproduce recorded first-event selection')
            self.first_frame[newly_observed] = frame

    def first_control_tick(self, case, policy_dt=.01):
        frame = self.first_frame[case]
        return np.inf if not np.isfinite(frame) else int(np.ceil(frame*self.dt/policy_dt-1e-9))

    def seen(self, case, tick, public_clock, policy_dt=.01):
        if abs(float(public_clock)-tick*policy_dt) > 1e-6:
            raise ValueError('Only original ground-start first-episode command clocks supported')
        return bool(tick >= self.first_control_tick(case, policy_dt))
