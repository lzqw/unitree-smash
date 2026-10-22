#!/usr/bin/env python3
"""Compare production foot contacts with independent ground-filtered sensors.

Diagnostic only: no training, reward, observation or acceptance changes.
"""
import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--capability', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg

GROUPS = ['zero_action', 'existing_capability_trajectories', 'deterministic_policy', 'stochastic_policy']
COUNT_NAMES = [
    'substeps', 'raw_contact_foot_samples', 'ground_contact_foot_samples',
    'raw_only_high_foot_samples', 'contact_mask_mismatch_foot_samples',
    'raw_only_support_substeps', 'ground_only_support_substeps',
    'raw_airborne_substeps', 'ground_airborne_substeps', 'airborne_mismatch_substeps',
    'production_contact_mismatch_foot_samples', 'production_airborne_mismatch_substeps',
]


class AuditEnv(JumpEnv):
    def _setup_scene(self):
        super()._setup_scene()
        # One sensor per foot avoids the documented many-to-many filter limit.
        for name in ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'):
            self.scene.sensors['audit_' + name] = ContactSensor(ContactSensorCfg(
                prim_path='/World/envs/env_.*/Robot/Go2/' + name,
                filter_prim_paths_expr=['/World/Ground/geometry/mesh'],
                force_threshold=1.,
            ))

    def __init__(self, cfg):
        super().__init__(cfg)
        self.audit_names = list(self.feet_contact.body_names)
        self.audit_sensors = [self.scene.sensors['audit_' + name] for name in self.audit_names]
        assert [s.body_names[0] for s in self.audit_sensors] == self.audit_names
        assert [self.robot.body_names[i] for i in self.foot_ids] == self.audit_names
        for sensor in self.audit_sensors:
            assert sensor.data.force_matrix_w.shape == (self.num_envs, 1, 1, 3)
        self.audit_contacts = torch.zeros((2, self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.audit_clear_time = torch.zeros((2, self.num_envs), device=self.device)
        self.audit_counts = torch.zeros((self.num_envs, len(COUNT_NAMES)), dtype=torch.int64, device=self.device)
        self.audit_max_delta = torch.zeros((self.num_envs, 4), device=self.device)
        # For each foot, retain the actual worst discrepancy with its context.
        self.audit_worst_context = torch.zeros((self.num_envs, 4, 4), device=self.device)
        self.audit_frame = -1
        self.audit_ready = False

    def _observe_physics_frame(self, frame):
        super()._observe_physics_frame(frame)
        if not getattr(self, 'audit_ready', False) or frame <= self.audit_frame or frame == 0:
            return
        self.audit_frame = frame
        raw_vector = self.feet_contact.data.net_forces_w
        ground_vector = torch.cat([s.data.force_matrix_w.sum(2) for s in self.audit_sensors], dim=1)
        raw = raw_vector.norm(dim=-1)
        ground = ground_vector.norm(dim=-1)
        heights = self.robot.data.body_pos_w[:, self.foot_ids, 2]
        forces = torch.stack((raw, ground))
        self.audit_contacts.copy_(torch.where(self.audit_contacts, forces > .5, forces > 1.))
        supported = self.audit_contacts.any(2)
        clear = (~supported) & (heights.min(1).values[None, :] > .035)
        self.audit_clear_time.copy_(torch.where(clear, self.audit_clear_time + self.physics_dt, 0.))
        airborne = self.audit_clear_time >= .02
        values = [
            torch.ones(self.num_envs, device=self.device, dtype=torch.int64),
            self.audit_contacts[0].sum(1), self.audit_contacts[1].sum(1),
            ((raw > 1.) & (ground <= .5) & (heights > .035)).sum(1),
            (self.audit_contacts[0] != self.audit_contacts[1]).sum(1),
            supported[0] & ~supported[1], supported[1] & ~supported[0],
            airborne[0], airborne[1], airborne[0] != airborne[1],
            (self.audit_contacts[0] != self.observed_foot_contact).sum(1),
            airborne[0] != self.observed_airborne,
        ]
        self.audit_counts += torch.stack(values, dim=1).to(torch.int64)
        delta = (raw_vector - ground_vector).norm(dim=-1)
        changed = delta > self.audit_max_delta
        context = torch.stack((torch.full_like(raw, frame * self.physics_dt), heights, raw, ground), dim=-1)
        self.audit_worst_context.copy_(torch.where(changed[..., None], context, self.audit_worst_context))
        torch.maximum(self.audit_max_delta, delta, out=self.audit_max_delta)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, 'audit_contacts'):
            self.audit_contacts[:, env_ids] = False
            self.audit_clear_time[:, env_ids] = 0.


env = None
try:
    cfg = JumpEnvCfg()
    cfg.scene.num_envs = 128
    cfg.seed = 911
    cfg.sim.device = args.device or 'cuda:0'
    env = AuditEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    config = json.loads(args.config.read_text())
    runner = NativeRunner(wrapped, config['runner'], log_dir=str(args.output / 'ppo'), device=env.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    capability = json.loads(args.capability.read_text())['best']
    parameters = torch.tensor([json.loads(capability[i % len(capability)]['parameters']) for i in range(32)], device=env.device)
    crouch_time, push_time, crouch_thigh, crouch_calf, extend_thigh, extend_calf, front_bias, tuck_time = parameters.unbind(1)
    with torch.inference_mode():
        wrapped.reset()
        obs = wrapped.get_observations()
        env.audit_ready = True
        for step in range(400):
            action = policy(obs).clone()
            sampled = runner.alg.policy.act(obs)
            action[:32] = 0.
            action[96:] = sampled[96:]
            t = env._phase()[32:64]
            crouch = (t >= .8) & (t < .8 + crouch_time)
            push = (t >= .8 + crouch_time) & (t < .8 + crouch_time + push_time)
            tuck = (t >= .8 + crouch_time + push_time) & (t < .8 + crouch_time + push_time + tuck_time)
            command = torch.zeros((32, 18), device=env.device)
            command[:, 4:8] = torch.where(crouch, crouch_thigh, torch.where(push, -extend_thigh, torch.where(tuck, .3, 0.)))[:, None]
            command[:, 8:12] = torch.where(crouch, -crouch_calf, torch.where(push, extend_calf, torch.where(tuck, -.6, 0.)))[:, None]
            command[:, 4:6] += (push * front_bias)[:, None]
            command[:, 6:8] -= (push * front_bias)[:, None]
            action[32:64] = command.clamp(-1., 1.)
            obs, _, _, _ = wrapped.step(action)
        counts = env.audit_counts.cpu().tolist()
        deltas = env.audit_max_delta.cpu().tolist()
        contexts = env.audit_worst_context.cpu().tolist()
        rows = [dict(env_id=i, group=GROUPS[i // 32], **dict(zip(COUNT_NAMES, row)),
                     maximum_vector_discrepancy_n=max(deltas[i])) for i, row in enumerate(counts)]
        groups = {}
        for group in GROUPS:
            selected = [r for r in rows if r['group'] == group]
            episodes = [r for r in env.episode_records if GROUPS[r['env_id'] // 32] == group]
            groups[group] = {name: sum(r[name] for r in selected) for name in COUNT_NAMES}
            groups[group].update(maximum_vector_discrepancy_n=max(r['maximum_vector_discrepancy_n'] for r in selected),
                completed_episodes=len(episodes), qualified_jump_landings=sum(r['qualified_jump_landed'] for r in episodes),
                complete_successes=sum(r['success'] for r in episodes))
        worst = sorted([dict(env_id=i, group=GROUPS[i // 32], foot=name, vector_discrepancy_n=deltas[i][j],
            time_s=contexts[i][j][0], foot_center_height_m=contexts[i][j][1],
            raw_force_n=contexts[i][j][2], ground_force_n=contexts[i][j][3])
            for i in range(128) for j, name in enumerate(env.audit_names) if deltas[i][j] > 0.],
            key=lambda r: r['vector_discrepancy_n'], reverse=True)[:32]
        report = dict(kind='foot ground sensor diagnostic; no learning or acceptance claim', seed=911,
            env_steps=51200, physics_dt_s=env.physics_dt, foot_order=env.audit_names,
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            capability_source=str(args.capability), capability_sha256=hashlib.sha256(args.capability.read_bytes()).hexdigest(),
            ground_filter='/World/Ground/geometry/mesh', one_sensor_per_foot=True,
            contact_on_n=1., contact_off_n=.5, clearance_m=.035, airborne_debounce_s=.02,
            groups=groups, worst_discrepancies=worst,
            limitations='Finite diagnostic trajectories; absence of discrepancies is not proof for every configuration. All policies use the unchanged production net-force observation.')
        for name, data in [('per_environment.csv', rows), ('episodes.csv', env.episode_records)]:
            with (args.output / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(data[0]))
                writer.writeheader()
                writer.writerows(data)
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
except BaseException:
    (args.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    if env is not None:
        env.close()
    app.close(skip_cleanup=True)
