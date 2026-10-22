"""Optional external recorder; never supplies rewards/actions to the learner."""
import csv
import json
import torch
from .jump_reward_audit import RecoveryWindows, v2c_components, recovery_draft


class JumpDiagnosticRecorder:
    def __init__(self, env, output, groups, protocol):
        self.output, self.groups, self.protocol = output, groups, protocol
        self.windows = RecoveryWindows(env.num_envs, env.device)
        self.active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.weak_paid = torch.zeros(env.num_envs, device=env.device)
        self.mean = torch.zeros((env.num_envs, 18), device=env.device)
        self.std = torch.zeros_like(self.mean)
        self.model_std = torch.zeros_like(self.mean)
        self.episode_rows = []
        self.reward_sums = {}
        self.max_reconstruction_error = 0.
        self.stream = (output / 'reward_trace.csv').open('w')
        self.writer = None

    def on_physics(self, env, roll_pitch):
        d = env.robot.data
        self.windows.observe(env, dt=env.physics_dt, support=env.observed_foot_contact.any(1),
            roll_pitch=roll_pitch, horizontal_speed=d.root_link_lin_vel_w[:, :2].norm(dim=-1),
            angular_speed=d.root_ang_vel_w.norm(dim=-1), joint_speed=d.joint_vel.abs().amax(1),
            com_vertical_velocity=env.com_velocity[:, 2], protocol=self.protocol)

    def on_reward(self, env, features, reward):
        components = v2c_components(features, env.step_dt)
        shadow, self.weak_paid = recovery_draft(components, valid_jump_landed=env.valid_jump_landed,
            body_ground_seen=env.body_ground_seen, recovery_quality=features['recovery_quality'],
            weak_paid=self.weak_paid, dt=env.step_dt)
        ids = self.active.nonzero().flatten()
        error = float((sum(components.values()) - reward).abs().amax())
        self.max_reconstruction_error = max(self.max_reconstruction_error, error)
        if error > 2e-6:
            raise RuntimeError('reward component reconstruction differs from unchanged production reward')
        for name, values in components.items():
            self.reward_sums.setdefault(name, torch.zeros_like(reward))
            self.reward_sums[name] += values * self.active
        values = dict(time_s=env._phase(), com_height_m=env.com_position[:, 2],
            reset_com_height_m=env.reset_com_height, target_com_height_m=features['target_com'],
            com_vertical_velocity_m_s=env.com_velocity[:, 2], takeoff_time_s=env.takeoff_time,
            landing_time_s=env.landing_time, flight_candidate=env.flight_candidate,
            qualified_flight=env.flight_qualified, qualified_landing=env.valid_jump_landed,
            supported_feet=env.observed_foot_contact.sum(1), body_ground_seen=env.body_ground_seen,
            recovery_run_s=env.recovery_run, production_reward=reward,
            draft_shadow_reward=sum(shadow.values()), draft_weak_paid=self.weak_paid,
            actual_action_clip_fraction=(env.action_before_clip.abs()>1.).float().mean(1),
            mean_outside_bounds_fraction=(self.mean.abs()>1.).float().mean(1),
            **{'condition_run_' + name + '_s': self.windows.run[:, j] for j, name in enumerate(self.windows.names)},
            **{'reward_' + k: v for k, v in components.items()},
            **{'draft_' + k: shadow[k] for k in ('stable_recovery', 'recovery_quality', 'weak_safe_settling')})
        scalars = torch.stack([v.float()[ids] for v in values.values()], 1).cpu().tolist()
        vectors = torch.stack((env.action_before_clip[ids], env.last_action[ids],
                               self.mean[ids], self.model_std[ids], self.std[ids]), 1).cpu().tolist()
        targets = env.targets[ids][:, env.joint_ids].cpu().tolist()
        for i, row_values, vector, target in zip(ids.cpu().tolist(), scalars, vectors, targets):
            row = dict(group=self.groups[i], env_id=i, **dict(zip(values, row_values)),
                raw_action=json.dumps(vector[0]), applied_action=json.dumps(vector[1]),
                policy_mean=json.dumps(vector[2]), policy_std=json.dumps(vector[3]),
                execution_noise_std=json.dumps(vector[4]), joint_targets=json.dumps(target))
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)

    def on_episodes(self, env, ids):
        latest = {r['env_id']: r for r in env.episode_records[-len(ids):]}
        for i in ids.cpu().tolist():
            if not bool(self.active[i]):
                continue
            row = dict(group=self.groups[i], **latest[i],
                airborne_seen=bool(self.windows.seen[i, 0]),
                qualified_flight_ever=bool(self.windows.seen[i, 1]),
                qualified_landing_ever=bool(self.windows.seen[i, 2]),
                last_takeoff_com_velocity_m_s=float(self.windows.takeoff_velocity[i]),
                **{'maximum_continuous_' + name + '_s': float(self.windows.maximum[i, j])
                   for j, name in enumerate(self.windows.names)},
                **{'return_' + name: float(value[i]) for name, value in self.reward_sums.items()})
            self.episode_rows.append(row)
        self.active[ids] = False
        self.windows.reset(ids)
        self.weak_paid[ids] = 0.

    def close(self):
        self.stream.close()

    def finish(self):
        self.close()
        if self.active.any():
            raise RuntimeError('not all initial diagnostic episodes completed')
        rows = sorted(self.episode_rows, key=lambda r: r['env_id'])
        with (self.output / 'diagnostic_episodes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = {}
        for name in dict.fromkeys(self.groups):
            group = [r for r in rows if r['group'] == name]
            landings = sum(r['qualified_landing_ever'] for r in group)
            success = sum(r['success'] for r in group)
            summary[name] = dict(episodes=len(group), airborne=sum(r['airborne_seen'] for r in group),
                qualified_flight=sum(r['qualified_flight_ever'] for r in group),
                qualified_landing=landings, complete_recovery=success,
                recovery_given_qualified_landing=success / landings if landings else None,
                body_contacts=sum(r['body_landing'] for r in group),
                mean_component_returns={key: sum(r[key] for r in group) / len(group)
                                        for key in group[0] if key.startswith('return_')})
        return dict(groups=summary, reward_reconstruction_max_error=self.max_reconstruction_error,
                    recovery_windows_note='Longest individual condition intervals after any landing; not combined into formal success.')
