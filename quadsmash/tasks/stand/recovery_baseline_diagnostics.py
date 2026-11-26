"""Read-only attribution at the native contact-flight recovery metric cadence."""
import math

import torch


STRICT_FIELDS = ('support', 'roll_pitch', 'horizontal_speed', 'angular_speed',
                 'joint_speed', 'body_ground')


def strict_failures(limits, *, support, tilt, speed, angular, leg_speed, arm_speed,
                    body_ground):
    joint_speed = torch.maximum(leg_speed, arm_speed)
    failed = torch.stack((
        ~support, tilt > math.radians(limits['maximum_abs_roll_pitch_deg']),
        speed > limits['maximum_horizontal_root_speed_m_s'],
        angular > limits['maximum_root_angular_speed_rad_s'],
        joint_speed > limits['maximum_abs_joint_velocity_rad_s'], body_ground), dim=-1)
    arm_only = (failed[:, 4] & (leg_speed <= limits['maximum_abs_joint_velocity_rad_s'])
                & ~failed[:, (0, 1, 2, 3, 5)].any(-1))
    return failed, arm_only


class RecoveryBaselineDiagnostics:
    def __init__(self, env, limits):
        n = env.num_envs
        self.limits = limits
        self.arm_ids = list(map(int, env.joint_ids[12:]))
        self.non_arm_ids = [i for i in range(len(env.robot.joint_names)) if i not in self.arm_ids]
        self.seconds = torch.zeros(n, len(STRICT_FIELDS), device=env.device)
        self.interrupts = torch.zeros_like(self.seconds, dtype=torch.long)
        self.observed = torch.zeros_like(self.seconds[:, 0])
        self.arm_only = torch.zeros_like(self.observed)
        # First native body-ground flag, with pose used to identify tilt direction.
        self.first_body = torch.zeros(n, 3, device=env.device)
        self.body_known = torch.zeros(n, dtype=torch.bool, device=env.device)

    def observe(self, env, previous_time, previous_run):
        from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

        dt = (env.jump_clock - previous_time).clamp_min(0.)
        data = env.robot.data
        roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w)
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        failed, arm_only = strict_failures(
            self.limits, support=env.observed_foot_contact.any(-1),
            tilt=torch.maximum(roll.abs(), pitch.abs()),
            speed=data.root_link_lin_vel_w[:, :2].norm(dim=-1),
            angular=data.root_ang_vel_w.norm(dim=-1),
            leg_speed=data.joint_vel[:, self.non_arm_ids].abs().amax(-1),
            arm_speed=data.joint_vel[:, self.arm_ids].abs().amax(-1),
            body_ground=env.body_ground_seen)
        first = env.body_ground_seen & ~self.body_known
        self.first_body.copy_(torch.where(first[:, None], torch.stack((env.jump_clock, roll, pitch), -1),
                                          self.first_body))
        self.body_known |= first
        recovery = env.contact_recovery
        active = recovery.qualified_landing & env.landed_once
        interval = torch.minimum(dt, (env.jump_clock - recovery.landing_time).clamp_min(0.)) * active
        self.observed.add_(interval)
        self.seconds.add_(interval[:, None] * failed)
        self.arm_only.add_(interval * arm_only)
        self.interrupts.add_(active[:, None] & (dt > 0)[:, None]
                             & (previous_run > 0)[:, None] & failed)

    def take(self, env, ids):
        complete = ids[env._episode_has_stepped[ids]]
        result = {}
        for lane in complete.tolist():
            index = int(env.episode_index[lane])
            time, roll, pitch = self.first_body[lane].tolist()
            grounded = bool(self.body_known[lane])
            landed = bool(env.contact_recovery.qualified_landing[lane])
            result[lane, index] = dict(
                qualified_contact_landing=landed,
                observed_after_landing_s=float(self.observed[lane]),
                strict_failed_seconds=dict(zip(STRICT_FIELDS, self.seconds[lane].tolist(), strict=True)),
                strict_window_interrupts=dict(zip(STRICT_FIELDS, self.interrupts[lane].tolist(), strict=True)),
                strict_arm_only_failed_s=float(self.arm_only[lane]),
                first_body_ground_s=time if grounded else None,
                first_body_ground_after_contact_landing_s=(
                    time - float(env.contact_recovery.landing_time[lane]) if grounded and landed else None),
                roll_pitch_at_body_ground_deg=[math.degrees(roll), math.degrees(pitch)] if grounded else None)
        for field in (self.seconds, self.interrupts, self.observed, self.arm_only,
                      self.first_body, self.body_known):
            field[ids] = 0
        return result
