"""Native no-ball virtual racket-event task built on the jump environment."""
from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .jump_env import JumpEnv, JumpEnvCfg


@configclass
class VirtualHitEnvCfg(JumpEnvCfg):
    event_goal_enabled = True
    episode_length_s = 4.0
    virtual_goal_time_s = 1.25
    virtual_goal_offset = (0.23, 0.0, 1.18)
    train_position_tolerance_m = 0.12
    train_normal_tolerance_cos = 0.8660254
    train_time_tolerance_s = 0.08
    train_velocity_tolerance_mps = 1.5
    # Fixed before training: these four targets are a small reachable panel
    # around the validated G04 racket trajectory.  The target is exposed in
    # the native command observation on every episode.
    target_panel = (
        (1.48, (0.243, 0.001, 0.553), (-0.525, 0.000, -0.851), (0.00, 0.00, 0.00)),
        (1.49, (0.246, 0.001, 0.580), (-0.490, 0.000, -0.872), (0.00, 0.00, 0.00)),
        (1.50, (0.235, 0.001, 0.500), (-0.585, 0.000, -0.811), (0.00, 0.00, 0.00)),
        (1.52, (0.222, 0.002, 0.423), (-0.645, 0.000, -0.764), (0.00, 0.00, 0.00)),
    )


class VirtualHitEnv(JumpEnv):
    """Target a fixed, pre-validated panel of events while recovering.

    The panel is deliberately finite and declared in the config before a run.
    It is not filtered or resampled from failures; every target is presented
    through the same event-goal command fields used by the policy.
    """

    cfg: VirtualHitEnvCfg

    def __init__(self, cfg: VirtualHitEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.racket_body_id = self.robot.find_bodies("Racket", preserve_order=True)[0][0]
        self.virtual_event_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.virtual_position_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.virtual_normal_error_deg = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.virtual_time_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.virtual_velocity_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.virtual_target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.virtual_episode_records: list[dict[str, float | bool | int]] = []

    def _goal_position(self) -> torch.Tensor:
        return self.goal_position_w

    def _event_terms(self):
        data = self.robot.data
        pos = data.body_pos_w[:, self.racket_body_id]
        vel = data.body_vel_w[:, self.racket_body_id, :3]
        quat = data.body_quat_w[:, self.racket_body_id]
        normal = math_utils.quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1))
        goal_pos = self._goal_position()
        position_error = torch.linalg.vector_norm(pos - goal_pos, dim=1)
        goal_normal = self.goal_normal_w
        normal_alignment = (normal * goal_normal).sum(1).clamp(-1.0, 1.0)
        normal_error_deg = torch.rad2deg(torch.arccos(normal_alignment))
        time_error = torch.abs(self._phase() - self.goal_time_s)
        velocity_error = torch.linalg.vector_norm(vel - self.goal_velocity_w, dim=1)
        self._observe_physics_frame(self._sim_step_counter)
        airborne = self.observed_airborne
        valid = (
            airborne
            & (position_error <= self.cfg.train_position_tolerance_m)
            & (normal_alignment >= self.cfg.train_normal_tolerance_cos)
            & (time_error <= self.cfg.train_time_tolerance_s)
            & (velocity_error <= self.cfg.train_velocity_tolerance_mps)
        )
        return position_error, normal_error_deg, time_error, velocity_error, valid

    def _get_observations(self):
        # CoupledEnv.observe_native already encodes time, position, normal and
        # velocity relative to these per-episode fields.  Keeping this method
        # explicit documents that no hidden target state is appended.
        return super()._get_observations()

    def _get_rewards(self):
        reward = super()._get_rewards()
        pos_err, normal_err, time_err, vel_err, valid = self._event_terms()
        torch.minimum(self.virtual_position_error, pos_err, out=self.virtual_position_error)
        torch.minimum(self.virtual_normal_error_deg, normal_err, out=self.virtual_normal_error_deg)
        torch.minimum(self.virtual_time_error, time_err, out=self.virtual_time_error)
        torch.minimum(self.virtual_velocity_error, vel_err, out=self.virtual_velocity_error)
        first = valid & (~self.virtual_event_seen)
        self.virtual_event_seen |= valid
        shaping = (
            torch.exp(-pos_err / 0.08)
            + torch.exp(-time_err / 0.05)
            + torch.exp(-vel_err / 1.0)
            + (normal_err < 30.0).to(pos_err.dtype)
        )
        return reward + 0.20 * shaping * self.step_dt + 2.0 * first.to(pos_err.dtype)

    def _reset_idx(self, env_ids):
        completed_ids = env_ids[self._episode_has_stepped[env_ids]] if self._metrics_started and env_ids.numel() else env_ids[:0]
        if completed_ids.numel() and hasattr(self, "virtual_event_seen"):
            jump_success = self.jump_success[completed_ids]
            for offset, env_id in enumerate(completed_ids.detach().cpu().tolist()):
                self.virtual_episode_records.append(
                    {
                        "env_id": int(env_id),
                        "target_index": int(self.virtual_target_index[env_id].item()),
                        "target_time_s": float(self.goal_time_s[env_id].item()),
                        "target_x_m": float((self.goal_position_w[env_id] - self.scene.env_origins[env_id])[0].item()),
                        "target_y_m": float((self.goal_position_w[env_id] - self.scene.env_origins[env_id])[1].item()),
                        "target_z_m": float((self.goal_position_w[env_id] - self.scene.env_origins[env_id])[2].item()),
                        "virtual_event": bool(self.virtual_event_seen[env_id].item()),
                        "recovery_success": bool(jump_success[offset].item()),
                        "position_error_m": float(self.virtual_position_error[env_id].item()),
                        "normal_error_deg": float(self.virtual_normal_error_deg[env_id].item()),
                        "time_error_s": float(self.virtual_time_error[env_id].item()),
                        "velocity_error_mps": float(self.virtual_velocity_error[env_id].item()),
                    }
                )
        super()._reset_idx(env_ids)
        if hasattr(self, "virtual_target_index") and env_ids.numel():
            panel = self.cfg.target_panel
            ids = self.virtual_target_index[env_ids] % len(panel)
            times = torch.tensor([panel[int(i)][0] for i in ids.detach().cpu().tolist()], device=self.device)
            positions = torch.tensor([panel[int(i)][1] for i in ids.detach().cpu().tolist()], device=self.device)
            normals = torch.tensor([panel[int(i)][2] for i in ids.detach().cpu().tolist()], device=self.device)
            velocities = torch.tensor([panel[int(i)][3] for i in ids.detach().cpu().tolist()], device=self.device)
            self.goal_time_s[env_ids] = times
            self.goal_position_w[env_ids] = self.scene.env_origins[env_ids] + positions
            self.goal_normal_w[env_ids] = normals
            self.goal_velocity_w[env_ids] = velocities
            self.virtual_target_index[env_ids] += 1
        if hasattr(self, "virtual_event_seen"):
            self.virtual_event_seen[env_ids] = False
            self.virtual_position_error[env_ids] = float("inf")
            self.virtual_normal_error_deg[env_ids] = float("inf")
            self.virtual_time_error[env_ids] = float("inf")
            self.virtual_velocity_error[env_ids] = float("inf")
