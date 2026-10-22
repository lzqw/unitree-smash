"""Full-load jump and ground-recovery curriculum for the native articulation."""
from __future__ import annotations

import torch
import isaaclab.sensors as sensors
from isaaclab.utils import configclass

from .coupled_env import CoupledEnv, CoupledEnvCfg, LEG_JOINTS


@configclass
class JumpEnvCfg(CoupledEnvCfg):
    episode_length_s = 4.0
    is_finite_horizon = False
    feet_contact = sensors.ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Go2/.*_foot",
        track_air_time=True,
        track_contact_points=False,
        force_threshold=1.0,
    )


class JumpEnv(CoupledEnv):
    cfg: JumpEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        feet_contact = sensors.ContactSensor(self.cfg.feet_contact)
        self.scene.sensors["feet_contact"] = feet_contact

    def __init__(self, cfg: JumpEnvCfg, **kwargs):
        # G01 standing audit uses a fixed base, but jump/recovery must expose
        # the real floating-base articulation.  Keep the same full-load USD,
        # actuator limits, and joint contract while disabling only the root
        # lock for this task configuration.
        cfg.robot.spawn.articulation_props.fix_root_link = False
        super().__init__(cfg, **kwargs)
        self.feet_contact = self.scene.sensors["feet_contact"]
        self.leg_ids, _ = self.robot.find_joints(list(LEG_JOINTS), preserve_order=True)
        self.airborne_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, device=self.device)
        self.max_height = torch.zeros(self.num_envs, device=self.device)
        self.recovery_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_time = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        self._metrics_started = False
        self._episode_has_stepped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_records: list[dict[str, float | bool | int]] = []

    def _phase(self) -> torch.Tensor:
        return self.episode_length_buf.to(torch.float32) * self.step_dt

    def _feet_in_contact(self) -> torch.Tensor:
        forces = self.feet_contact.data.net_forces_w
        return torch.linalg.vector_norm(forces, dim=-1) > self.cfg.feet_contact.force_threshold

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, 18) or not torch.isfinite(actions).all():
            raise ValueError("expected finite [num_envs, 18] actions")
        phase = self._phase().unsqueeze(1)
        leg_scale = torch.where(phase < 0.8, torch.full_like(phase, 0.25), torch.full_like(phase, 0.55))
        scale = torch.cat((leg_scale.expand(-1, 12), torch.full((self.num_envs, 6), 0.12, device=self.device)), dim=1)
        self.targets = self.robot.data.default_joint_pos.clone()
        self.targets[:, self.joint_ids] += scale * actions.clamp(-1.0, 1.0)

    def _get_observations(self):
        data = self.robot.data
        phase = self._phase().clamp(0.0, 4.0).unsqueeze(1) / 4.0
        airborne = self.airborne_seen.to(torch.float32).unsqueeze(1)
        height_error = (data.root_pos_w[:, 2] - 0.4).unsqueeze(1).clamp(-1.0, 1.0)
        obs = torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                (data.joint_pos - data.default_joint_pos)[:, self.joint_ids],
                data.joint_vel[:, self.joint_ids],
                phase,
                airborne,
                height_error,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        self._episode_has_stepped[:] = True
        data = self.robot.data
        phase = self._phase()
        height = data.root_pos_w[:, 2]
        tilt = torch.sum(data.projected_gravity_b[:, :2].square(), dim=1)
        upright = (data.projected_gravity_b[:, 2] < -0.5).to(torch.float32)
        feet_in_contact = self._feet_in_contact()
        all_feet_airborne = ~feet_in_contact.any(dim=1)
        airborne_actual = (all_feet_airborne & (height > 0.43)).to(torch.float32)
        airborne_reward = airborne_actual * (height < 0.75).to(torch.float32)
        self.airborne_seen |= airborne_actual.bool()
        self.airborne_time += airborne_actual * self.step_dt
        self.max_height = torch.maximum(self.max_height, height)
        recovery = (
            (phase > 1.8)
            & feet_in_contact.any(dim=1)
            & (height > 0.30)
            & (data.root_pos_w[:, 2] < 0.52)
            & (upright > 0.5)
        ).to(torch.float32)
        self.recovery_seen |= recovery.bool()
        self.recovery_time += recovery * self.step_dt
        target_height = 0.40 + 0.12 * torch.exp(-((phase - 1.15) / 0.45).square())
        height_reward = torch.exp(-((height - target_height) / 0.08).square())
        recovery_reward = torch.where(phase > 1.8, 0.8 * upright + 0.8 * recovery, torch.zeros_like(phase))
        overshoot_penalty = torch.relu(height - 0.70).square()
        vertical_speed_penalty = torch.where(phase > 1.4, 0.03 * data.root_lin_vel_b[:, 2].square(), torch.zeros_like(height))
        return (
            0.8 * height_reward
            + 0.8 * airborne_reward
            + recovery_reward
            - 2.0 * tilt
            - 4.0 * overshoot_penalty
            - vertical_speed_penalty
            - 0.0001 * data.applied_torque.square().sum(1)
        ) * self.step_dt

    def _get_dones(self):
        # During the airborne phase, tilt is part of the recovery problem and
        # must not terminate the rollout before the feet can contact the ground.
        # Keep only the physically unrecoverable low-height termination here;
        # the frozen success metric still requires upright feet contact and a
        # full recovery interval.
        fallen = self.robot.data.root_pos_w[:, 2] < 0.12
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _reset_idx(self, env_ids):
        # DirectRLEnv calls this for automatic episode resets.  Record the
        # completed episode before the base class clears its state.  The
        # construction-time reset happens before these buffers exist and is
        # intentionally ignored.
        completed_ids = env_ids[self._episode_has_stepped[env_ids]] if self._metrics_started and env_ids.numel() else env_ids[:0]
        if completed_ids.numel():
            success = (
                (self.max_height[completed_ids] >= 0.48)
                & (self.airborne_time[completed_ids] >= 0.10)
                & (self.recovery_time[completed_ids] >= 0.80)
            )
            self.completed_episodes += int(completed_ids.numel())
            self.successful_episodes += int(success.sum().item())
            self.failed_episodes += int((~success).sum().item())
            for offset, env_id in enumerate(completed_ids.detach().cpu().tolist()):
                self.episode_records.append(
                    {
                        "env_id": int(env_id),
                        "max_height_m": float(self.max_height[env_id].item()),
                        "airborne_s": float(self.airborne_time[env_id].item()),
                        "recovery_s": float(self.recovery_time[env_id].item()),
                        "success": bool(success[offset].item()),
                    }
                )
        super()._reset_idx(env_ids)
        self.airborne_seen[env_ids] = False
        self.airborne_time[env_ids] = 0.0
        self.max_height[env_ids] = 0.0
        self.recovery_seen[env_ids] = False
        self.recovery_time[env_ids] = 0.0
        self._episode_has_stepped[env_ids] = False
        self._metrics_started = True
