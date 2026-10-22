"""Native Isaac Lab event stages with an independently simulated shuttlecock.

The ball is a separate PhysX rigid body.  It is initialized once at episode
reset and thereafter advances only through the simulator; no near-racket
teleport or scripted contact correction is used.
"""
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from .jump_env import JumpEnv, JumpEnvCfg


@configclass
class BallEnvCfg(JumpEnvCfg):
    episode_length_s = 4.0
    shuttle_radius_m = 0.025
    shuttle_mass_kg = 0.005
    shuttle_start = (0.95, 0.0, 1.15)
    shuttle_velocity = (-2.0, 0.0, 0.0)
    include_net = False
    net_height_m = 1.524
    net_x_m = 1.5
    target_x_min_m = 2.0
    target_x_max_m = 4.0
    target_y_half_width_m = 1.0
    shuttle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Shuttle",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.72,
                static_friction=0.08,
                dynamic_friction=0.08,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.95, 0.0, 1.15),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(-2.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


class BallEnv(JumpEnv):
    """Full-load robot plus a real PhysX shuttle and contact reporter."""

    cfg: BallEnvCfg

    def __init__(self, cfg: BallEnvCfg, **kwargs):
        self.contact_seen = None
        self.contact_time = None
        self.contact_impulse = None
        self.ball_min_distance = None
        self.previous_ball_velocity = None
        self.ball_episode_records: list[dict[str, float | bool | int]] = []
        super().__init__(cfg, **kwargs)
        self.racket_body_id = self.robot.find_bodies("Racket", preserve_order=True)[0][0]
        self.contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_time = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.contact_impulse = torch.zeros(self.num_envs, device=self.device)
        self.ball_min_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.previous_ball_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.outgoing_speed = torch.zeros(self.num_envs, device=self.device)
        self.outgoing_vz = torch.zeros(self.num_envs, device=self.device)
        self.crossed_net = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landed_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self.shuttle = RigidObject(self.cfg.shuttle)
        self.scene.rigid_objects["shuttle"] = self.shuttle
        if self.cfg.include_net:
            net_cfg = sim_utils.CuboidCfg(
                size=(0.02, 2.0, self.cfg.net_height_m),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5, dynamic_friction=0.5, restitution=0.1
                ),
            )
            for env_id, origin in enumerate(self.scene.env_origins.detach().cpu().tolist()):
                path = f"/World/envs/env_{env_id}/Net"
                if not get_current_stage().GetPrimAtPath(path).IsValid():
                    net_cfg.func(
                        path,
                        net_cfg,
                        translation=(origin[0] + self.cfg.net_x_m, origin[1], self.cfg.net_height_m / 2.0),
                    )

    def _write_shuttle_reset(self, env_ids):
        origins = self.scene.env_origins[env_ids]
        count = len(env_ids)
        state = torch.zeros((count, 13), device=self.device)
        state[:, :3] = origins + torch.tensor(self.cfg.shuttle_start, device=self.device)
        state[:, 3] = 1.0
        state[:, 7:10] = torch.tensor(self.cfg.shuttle_velocity, device=self.device)
        self.shuttle.write_root_state_to_sim(state, env_ids)
        if self.previous_ball_velocity is not None:
            self.previous_ball_velocity[env_ids] = state[:, 7:10]

    def _reset_idx(self, env_ids):
        completed_ids = env_ids[self._episode_has_stepped[env_ids]] if self._metrics_started and env_ids.numel() else env_ids[:0]
        if completed_ids.numel() and self.contact_seen is not None:
            recovery_success = (
                (self.max_height[completed_ids] >= 0.48)
                & (self.airborne_time[completed_ids] >= 0.10)
                & (self.recovery_time[completed_ids] >= 0.80)
            )
            for offset, env_id in enumerate(completed_ids.detach().cpu().tolist()):
                self.ball_episode_records.append(
                    {
                        "env_id": int(env_id),
                        "contact": bool(self.contact_seen[env_id].item()),
                        "contact_time_s": float(self.contact_time[env_id].item()),
                        "contact_impulse": float(self.contact_impulse[env_id].item()),
                        "ball_min_distance_m": float(self.ball_min_distance[env_id].item()),
                        "recovery_success": bool(recovery_success[offset].item()),
                        "outgoing_speed_mps": float(self.outgoing_speed[env_id].item()),
                        "outgoing_vz_mps": float(self.outgoing_vz[env_id].item()),
                        "crossed_net": bool(self.crossed_net[env_id].item()),
                        "landed_target": bool(self.landed_target[env_id].item()),
                    }
                )
        super()._reset_idx(env_ids)
        self._write_shuttle_reset(env_ids)
        if self.contact_seen is not None:
            self.contact_seen[env_ids] = False
            self.contact_time[env_ids] = float("nan")
            self.contact_impulse[env_ids] = 0.0
            self.ball_min_distance[env_ids] = float("inf")
            self.outgoing_speed[env_ids] = 0.0
            self.outgoing_vz[env_ids] = 0.0
            self.crossed_net[env_ids] = False
            self.landed_target[env_ids] = False

    def _get_observations(self):
        obs = super()._get_observations()["policy"]
        # Preserve the 48-D G03 interface.  The three event slots carry the
        # current racket-relative ball position for real-contact stages.
        relative = self.shuttle.data.root_pos_w - self.robot.data.body_pos_w[:, self.racket_body_id]
        obs[:, -3:] = relative.clamp(-2.0, 2.0)
        return {"policy": obs}

    def _get_rewards(self):
        reward = super()._get_rewards()
        distance = torch.linalg.vector_norm(self.shuttle.data.root_pos_w - self.robot.data.body_pos_w[:, self.racket_body_id], dim=1)
        self.ball_min_distance = torch.minimum(self.ball_min_distance, distance)
        current_velocity = self.shuttle.data.root_lin_vel_w
        delta_velocity = torch.linalg.vector_norm(current_velocity - self.previous_ball_velocity, dim=1)
        self.previous_ball_velocity = current_velocity.clone()
        contact_now = (
            (delta_velocity > 0.15)
            & (distance < 0.18)
            & (self.shuttle.data.root_pos_w[:, 2] > 0.20)
        )
        first_contact = contact_now & (~self.contact_seen)
        self.contact_seen |= contact_now
        self.contact_impulse += 0.005 * delta_velocity
        self.contact_time = torch.where(first_contact, self._phase(), self.contact_time)
        local_ball = self.shuttle.data.root_pos_w - self.scene.env_origins
        speed = torch.linalg.vector_norm(current_velocity, dim=1)
        self.outgoing_speed = torch.where(self.contact_seen, torch.maximum(self.outgoing_speed, speed), self.outgoing_speed)
        self.outgoing_vz = torch.where(self.contact_seen, torch.minimum(self.outgoing_vz, current_velocity[:, 2]), self.outgoing_vz)
        self.crossed_net |= self.contact_seen & (local_ball[:, 0] > self.cfg.net_x_m) & (local_ball[:, 2] > self.cfg.net_height_m)
        self.landed_target |= (
            self.contact_seen
            & (local_ball[:, 2] < 0.08)
            & (local_ball[:, 0] >= self.cfg.target_x_min_m)
            & (local_ball[:, 0] <= self.cfg.target_x_max_m)
            & (local_ball[:, 1].abs() <= self.cfg.target_y_half_width_m)
        )
        return reward + 0.25 * first_contact.to(reward.dtype) * self.step_dt

