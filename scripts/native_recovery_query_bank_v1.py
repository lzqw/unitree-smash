"""Passive all-lane native control-boundary states for recovery planning.

Records current tensors before the next command is dispatched. This is not a
PhysX checkpoint: solver/contact caches and policy memories are not captured,
so these arrays must never be used to claim an exact teleported native reset.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from scripts.loaded_pinocchio_model_v1 import PublicState, rotation_wxyz


def state_from_sample(model, sample, origin):
    rotation = rotation_wxyz(sample['root_quaternion_wxyz'])
    root = np.asarray(sample['root_position_world'], dtype=float) - np.asarray(origin, dtype=float)
    omega = np.asarray(sample['root_com_angular_velocity_world'], dtype=float)
    com_velocity = np.asarray(sample['root_com_linear_velocity_world'], dtype=float)
    offset = rotation @ np.asarray(sample['root_com_offset_body'], dtype=float)
    q = pin.neutral(model.model); q[:3] = root; q[3:7] = pin.Quaternion(rotation).coeffs()
    q[model.q_indices] = sample['joint_position']
    v = np.zeros(model.model.nv); v[:3] = rotation.T @ (com_velocity - np.cross(omega, offset))
    v[3:6] = rotation.T @ omega; v[model.v_indices] = sample['joint_velocity']
    state = PublicState(q, v, rotation, root, q[model.q_indices].copy(), v[model.v_indices].copy(),
        np.asarray(sample['observed_contacts'], dtype=bool).copy(), 0.)
    state.previous_command_target = np.asarray(sample['previous_target'], dtype=float).copy()
    state.measured_force_world = np.asarray(sample['foot_force_world'], dtype=float).copy()
    return state


class NativeRecoveryQueryBankV1:
    def __init__(self, env, model, output):
        if env.num_envs != 128 or abs(env.physics_dt - .0000625) > 1e-12:
            raise ValueError('Original cohort and physical clock required')
        self.env, self.model, self.output = env, model, Path(output)
        if any(self.output.glob('recovery_query_bank*')):
            raise ValueError('Preserve previous state bank')
        self.first, self.last = 200, 799
        self.count = 0; self.buffers = {}; self.checks = []
        self.origins = env.scene.env_origins.detach().cpu().numpy().copy()
        self.joint_names = [env.robot.joint_names[i] for i in env.joint_ids]
        if self.joint_names != list(model.joint_order):
            raise ValueError('Native/model joint order mismatch')
        if list(env.feet_contact.body_names) != ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            raise ValueError('Unexpected contact ordering')

    def capture(self, tick, public, student_action, finished):
        if tick < self.first:
            return
        if tick != self.first + self.count or tick > self.last:
            raise ValueError('Require consecutive original control boundaries')
        e, d = self.env, self.env.robot.data
        values = dict(root_position_world=d.root_link_pos_w, root_quaternion_wxyz=d.root_link_quat_w,
            root_com_linear_velocity_world=d.root_com_lin_vel_w,
            root_com_angular_velocity_world=d.root_com_ang_vel_w,
            root_com_offset_body=d.body_com_pos_b[:, 0],
            joint_position=d.joint_pos[:, e.joint_ids], joint_velocity=d.joint_vel[:, e.joint_ids],
            previous_target=e.targets[:, e.joint_ids], foot_force_world=e.feet_contact.data.net_forces_w,
            observed_contacts=e.observed_foot_contact, episode_index=e.episode_index,
            unfinished_first_episode=~finished, phase=public[:, 770:775],
            episode_clock=public[:, 775], student_action=student_action)
        if not self.buffers:
            count = self.last - self.first + 1
            size = sum(count * x.numel() * x.element_size() for x in values.values())
            if size > 64 * 1024**2:
                raise ValueError('Declared 64 MiB state-bank memory cap exceeded')
            self.bytes = size
            self.buffers = {k: torch.empty((count,) + tuple(x.shape), dtype=x.dtype, device=x.device)
                for k, x in values.items()}
        for k, x in values.items():
            self.buffers[k][self.count].copy_(x.detach())
        if tick in [280, 350, 550, 750]:
            for lane in [3, 58, 125]:
                sample = {k: x[lane].detach().cpu().numpy().copy() for k, x in values.items()}
                rebuilt = state_from_sample(self.model, sample, self.origins[lane])
                direct = self.model.state_from_native(e, lane)
                errors = dict(q=float(np.max(abs(rebuilt.q - direct.q))),
                    v=float(np.max(abs(rebuilt.v - direct.v))))
                if max(errors.values()) > 1e-12 or not np.array_equal(rebuilt.contacts, direct.contacts):
                    raise ValueError('Stored native tensors do not reproduce state adapter')
                self.checks.append(dict(control_tick=tick, lane=lane, errors=errors, contacts_exact=True))
        self.count += 1

    def save(self):
        data = {k: x[:self.count].detach().cpu().numpy() for k, x in self.buffers.items()}
        if any(not np.isfinite(x).all() for x in data.values()):
            raise ValueError('Nonfinite native state-bank data')
        path = self.output / 'recovery_query_bank.npz'; tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, control_ticks=np.arange(self.first, self.first + self.count), **data)
        tmp.replace(path)
        result = dict(scope=__doc__, samples=self.count, first_tick=self.first, last_tick=self.last,
            complete_window=self.count == self.last - self.first + 1, lanes=list(range(128)),
            joint_names=self.joint_names, origins_world=self.origins.tolist(),
            allocated_bytes=getattr(self, 'bytes', 0), direct_adapter_checks=self.checks,
            capture_phase='After student proposal; before next action dispatch and physics advance.',
            previous_target_semantics='Native environment position target buffer from prior control step.',
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            controls_changed=False, native_reset_capable=False, valid_expert_labels=False,
            exact_observer_replay_pending=True)
        path = self.output / 'recovery_query_bank.json'; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); tmp.replace(path)
        return result
