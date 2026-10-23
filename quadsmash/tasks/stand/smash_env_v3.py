"""Opt-in ordered G07 protocol; legacy BallEnv defaults/evidence are preserved."""
from __future__ import annotations

import torch

from .ball_env import BallEnv, BallEnvCfg
from quadsmash.physics.native import quat_rotate
from .ball_env import PHYSICS
from quadsmash.physics.smash_events import OrderedSmashEvents


class OrderedSmashEnv(BallEnv):
    """Adds measurements without applying forces or changing collision response."""
    def __init__(self, cfg: BallEnvCfg, **kwargs):
        self.ordered_smash = None
        self.smash_episode_records = []
        super().__init__(cfg, **kwargs)
        self.ordered_smash = OrderedSmashEvents(
            self.num_envs, self.device, radius=cfg.shuttle_radius_m,
            net_x=cfg.net_x_m, net_height=cfg.net_height_m,
            target_x=(cfg.target_x_min_m, cfg.target_x_max_m),
            target_y=cfg.target_y_half_width_m)
        self.contact_tool_velocity = torch.full((self.num_envs, 3), float('nan'), device=self.device)
        self.contact_face_normal = self.contact_tool_velocity.clone()
        self.contact_root_position = self.contact_tool_velocity.clone()
        self.interception_min_distance = torch.full((self.num_envs,), float('inf'), device=self.device)

    def _consume_physics_frame(self, frame):
        if self.contact_events is None or frame <= self._last_physics_frame:
            return
        previous = self.previous_ball_position.clone()
        counts = self.contact_events.count.clone()
        super()._consume_physics_frame(frame)
        events = self.ordered_smash
        if events is None or frame == 0:
            return
        position = self.shuttle.data.root_pos_w
        tool_position = self.robot.data.body_pos_w[:, self.racket_body_id]
        tool_quaternion = self.robot.data.body_quat_w[:, self.racket_body_id]
        inverse = torch.cat((tool_quaternion[:, :1], -tool_quaternion[:, 1:]), dim=-1)
        local = quat_rotate(inverse, position - tool_position)
        # The two flat +/-X faces are the declared effective striking surface.
        # Rim-only PhysX contacts remain in legacy diagnostic counters.
        extents = PHYSICS['racket_half_extents_m']
        effective_face = ((local[:, 1].abs() <= extents[1] - 1e-4)
                          & (local[:, 2].abs() <= extents[2] - 1e-4)
                          & (local[:, 0].abs() >= extents[0] - 1e-4))
        force = self.shuttle_contact.data.force_matrix_w[:, 0]
        touching = force[:, 0].norm(dim=-1) > PHYSICS['contact_force_threshold_n']
        ground_touch = force[:, 1].norm(dim=-1) > PHYSICS['contact_force_threshold_n']
        delta = local.abs() - torch.tensor(extents, device=self.device)
        gap = (delta.clamp_min(0).norm(dim=-1) + delta.max(dim=-1).values.clamp_max(0)
               - self.cfg.shuttle_radius_m)
        separated = (~touching) & (gap > PHYSICS['separation_hysteresis_m'])
        local_ball = position - self.scene.env_origins
        incoming_window = (((self.ball_clock - self.goal_time_s).abs() <= .2)
                           & (~events.ground_seen)
                           & (local_ball[:, 2] > self.cfg.net_height_m + self.cfg.shuttle_radius_m)
                           & (local_ball[:, 0] < self.cfg.net_x_m))
        self.interception_min_distance.copy_(torch.where(
            incoming_window, torch.minimum(self.interception_min_distance, local.norm(dim=-1)),
            self.interception_min_distance))
        first = (self.contact_events.count > counts) & (~events.contact)
        if bool(first.any()):
            normal = quat_rotate(tool_quaternion, torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, -1))
            normal *= torch.where(local[:, :1] >= 0, 1., -1.)
            contact_point = position - self.cfg.shuttle_radius_m * normal
            data = self.robot.data
            velocity = (data.body_link_lin_vel_w[:, self.racket_body_id]
                        + torch.linalg.cross(data.body_link_ang_vel_w[:, self.racket_body_id],
                                             contact_point - tool_position))
            self.contact_tool_velocity[first] = velocity[first]
            self.contact_face_normal[first] = normal[first]
            self.contact_root_position[first] = (data.root_pos_w - self.scene.env_origins)[first]
        events.advance(previous_position=previous - self.scene.env_origins,
                       position=position - self.scene.env_origins,
                       velocity=self.shuttle.data.root_lin_vel_w,
                       contact_event=self.contact_events.count > counts,
                       touching=touching, airborne=self.observed_airborne,
                       effective_face=effective_face, ground_touch=ground_touch,
                       time=self.ball_clock, dt=self.physics_dt, separated=separated)

    def _reset_idx(self, ids):
        events = self.ordered_smash
        if events is not None:
            complete = ids[self._episode_has_stepped[ids]]
            numerical = self.sweep_misses + self.sweep_unresolved
            success = events.complete(self.jump_success) & (numerical == 0)
            for i in complete.cpu().tolist():
                self.smash_episode_records.append(dict(
                    protocol='ordered-smash-v3', env_id=i,
                    net_geometry_version=self.cfg.net_geometry_version,
                    airborne_contact=bool(events.airborne_contact[i]),
                    effective_face_contact=bool(events.effective_face[i]),
                    contact_count=int(events.contact_count[i]),
                    contact_position_m=events.contact_position[i].cpu().tolist(),
                    contact_point_tool_velocity_mps=self.contact_tool_velocity[i].cpu().tolist(),
                    contact_face_normal_world=self.contact_face_normal[i].cpu().tolist(),
                    contact_root_position_m=self.contact_root_position[i].cpu().tolist(),
                    contact_time_s=float(events.contact_time[i]),
                    outgoing_velocity_mps=events.outgoing_velocity[i].cpu().tolist(),
                    outgoing_position_m=events.outgoing_position[i].cpu().tolist(),
                    outgoing_measurement_time_s=float(events.outgoing_time[i]),
                    interception_min_distance_m=float(self.interception_min_distance[i]),
                    release_time_s=float(events.release_time[i]),
                    ordered_crossed_net=bool(events.crossed[i]),
                    geometric_crossed_net=bool(events.geometric_crossed[i]),
                    geometric_landed_target=bool(events.geometric_landed_target[i]),
                    net_margin_m=float(events.net_margin[i]),
                    net_crossing_position_m=events.net_position[i].cpu().tolist(),
                    net_crossing_time_s=float(events.crossing_time[i]),
                    first_landing_position_m=events.landing_position[i].cpu().tolist(),
                    first_landing_time_s=float(events.landing_time[i]),
                    ordered_landed_target=bool(events.landed_target[i]),
                    recovery_success=bool(self.jump_success[i]),
                    complete_smash=bool(success[i]),
                    body_contact=bool(self.body_contact_seen[i]),
                    numerical_failures=int(numerical[i]),
                    sweep_misses=int(self.sweep_misses[i]),
                    sweep_unresolved=int(self.sweep_unresolved[i])))
        super()._reset_idx(ids)
        if events is not None:
            events.reset(ids)
            self.contact_tool_velocity[ids] = float('nan')
            self.contact_face_normal[ids] = float('nan')
            self.contact_root_position[ids] = float('nan')
            self.interception_min_distance[ids] = float('inf')


class FullJointSmashEnv(OrderedSmashEnv):
    """Opt-in absolute joint-position interface with unchanged native actuators."""
    action_family = 'absolute-joint-limits-v1'

    def _pre_physics_step(self, actions):
        self.action_delta.copy_((actions.clamp(-1, 1) - self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count += (actions.abs() > 1).sum(1)
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        q = limits[..., 0] + .5 * (self.last_action + 1) * (limits[..., 1] - limits[..., 0])
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:, self.joint_ids] = q
        self._contact_bonus.zero_()
