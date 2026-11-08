"""Ground-only input for a recovery controller; original scoring is untouched.

The one-newton contact observation is distinct from five-newton load reporting.
Lightly loaded contact is not removed from the planner. A controller contact
flag never comes from another foot, even when that pair has a large upward force.
Updates occur once per real control tick, including before recovery entry.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.native_ground_pair_observer_v1 import aggregate_friction


def ground_contact_update(previous, normal_force_world_n):
    previous = np.asarray(previous, dtype=bool)
    normal = np.asarray(normal_force_world_n, dtype=float)
    if previous.shape != (4,) or normal.shape != (4, 3) or not np.isfinite(normal).all():
        raise ValueError('Four finite ground-filtered foot forces required')
    # Keep the original contact-force thresholds, but apply them only to the
    # ground pair. This is a control-rate observer, not the original task flag.
    norm = np.linalg.norm(normal, axis=1)
    contacts = np.where(previous, norm > .5, norm > 1.)
    loaded = contacts & (normal[:, 2] >= 5.)
    light = contacts & ~loaded
    # A nonzero but below-entry force is evidence to retain for planning,
    # not a license to impose a permanent no-slip support constraint.
    establishing = ~contacts & (norm > 0.)
    return contacts, dict(loaded=loaded, light=light, establishing=establishing)


class NativeGroundSupportStateV1:
    def __init__(self, env, case, output):
        self.env, self.case, self.output = env, int(case), Path(output)
        self.names = list(env.feet_contact.body_names)
        if self.names != ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'] or env.num_envs != 128:
            raise ValueError('Original foot ordering and fixed panel required')
        self.body_ids = [env.robot.body_names.index(n) for n in self.names]
        self.views = []
        for name in self.names:
            path = f'/World/envs/env_{case}/Robot/Go2/{name}'
            view = env.feet_contact._physics_sim_view.create_rigid_contact_view(
                path, filter_patterns=['/World/Ground/geometry/mesh'], max_contact_data_count=64)
            if list(view.sensor_paths) != [path] or view.filter_count != 1:
                raise ValueError('Unexpected native ground filter')
            self.views.append(view)
        self.contacts = np.zeros(4, bool)
        self.last_tick = -1
        self.episode = None
        self.records = []
        self.current = None

    def update(self, tick):
        if type(tick) is not int or tick != self.last_tick + 1 or not 0 <= tick < 800:
            raise ValueError('Exactly one update per original control tick required')
        e, case = self.env, self.case
        episode = int(e.episode_index[case].item())
        if episode != self.episode:
            self.contacts[:] = False
        prior = self.contacts.copy()
        normals, friction, moments, counts = [], [], [], []
        for i, view in enumerate(self.views):
            normal = view.get_contact_force_matrix(e.physics_dt)
            if tuple(normal.shape) != (1, 1, 3):
                raise ValueError('Wrong native ground force shape')
            f, p, count, start = view.get_friction_data(e.physics_dt)
            origin = e.robot.data.body_link_pos_w[case, self.body_ids[i]]
            force, moment, bad = aggregate_friction(f, p, count, start, origin)
            if bool(bad):
                raise ValueError('Native ground friction buffer overflow')
            normals.append(normal[0, 0].clone()); friction.append(force[0]); moments.append(moment[0])
            counts.append(count.reshape(-1)[0])
        cpu = lambda values: torch.stack(values).detach().cpu().numpy().copy()
        normal, friction, moment = [cpu(x) for x in [normals, friction, moments]]
        if not np.isfinite(friction).all() or not np.isfinite(moment).all():
            raise ValueError('Nonfinite native ground friction')
        self.contacts, modes = ground_contact_update(prior, normal)
        row = dict(tick=tick, episode_index=episode, prior_contacts=prior,
            ground_contacts=self.contacts.copy(), ground_normal_force_world_n=normal,
            ground_friction_force_world_n=friction, ground_friction_moment_world_nm=moment,
            ground_friction_count=cpu(counts), original_contacts=e.observed_foot_contact[case].detach().cpu().numpy().copy(),
            original_net_normal_world_n=e.feet_contact.data.net_forces_w[case].detach().cpu().numpy().copy(),
            foot_position_world=e.robot.data.body_link_pos_w[case, self.body_ids].detach().cpu().numpy().copy(),
            **modes)
        self.records.append(row); self.current = row
        self.last_tick, self.episode = tick, episode

    def apply(self, state, tick):
        if self.last_tick != tick or self.current is None or self.episode != 0:
            raise ValueError('Current first-episode ground observation required')
        result = copy.deepcopy(state)
        result.original_observed_contacts = state.contacts.copy()
        result.original_measured_force_world = state.measured_force_world.copy()
        result.contacts = self.contacts.copy()
        result.measured_force_world = self.current['ground_normal_force_world_n'].copy()
        result.ground_friction_world = self.current['ground_friction_force_world_n'].copy()
        result.contact_source = 'independent_ground_pair_control_rate_hysteresis_v1'
        # Do not retain the old, misleading alias for a net-force observation.
        if hasattr(result, 'geometric_contacts'):
            del result.geometric_contacts
        return result

    def save(self):
        if not self.records:
            return
        data = {k: np.asarray([r[k] for r in self.records]) for k in self.records[0]}
        path = self.output / 'controller_ground_observation.npz'
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **data)
        temp.replace(path)
        meta = dict(case=self.case, control_ticks=len(self.records), complete=len(self.records) == 800,
            contact_entry_n=1., contact_release_n=.5, load_reporting_n=5.,
            clock='actual control boundaries, 100 Hz; reset on episode change',
            contact_source='ground-only normal force; friction measured separately',
            light_contacts_kept=True, force_thresholds_are_not_geometric_evidence=True,
            scoring_changed=False, student_observation_changed=False,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path = self.output / 'controller_ground_observation.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(meta, indent=2) + '\n'); temp.replace(path)
