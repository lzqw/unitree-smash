"""Full Go2+PiPER+racket environment used only by the SuperMimic branch."""
from __future__ import annotations

import torch

from .full_racket_env_v4 import FullRacketEnvV4, full_racket_cfg


def make_supermimic_cfg(num_envs: int, physics_dt: float = 6.25e-5):
    cfg = full_racket_cfg(num_envs, physics_dt)
    cfg.episode_length_s = 8.0
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = num_envs
    return cfg


class SuperMimicEnvV1(FullRacketEnvV4):
    """Absolute 18-joint targets with native event-locked reward shaping."""
    algorithm_family = "supermimic-native-world-model-v1"

    def _pre_physics_step(self, actions):
        """Map a complete normalized target to legal joint limits around the
        native standing pose.  Zero is the standing pose; +/-1 still spans
        the complete legal interval on each joint side.  This is the new
        branch's absolute target interface, not a residual or old policy.
        """
        self.action_delta.copy_((actions.clamp(-1, 1) - self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count += (actions.abs() > 1).sum(1)
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        center = self.robot.data.default_joint_pos[:, self.joint_ids]
        span = torch.maximum(center - limits[..., 0], limits[..., 1] - center)
        targets = (center + self.last_action * span).clamp(limits[..., 0], limits[..., 1])
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:, self.joint_ids] = targets
        self._contact_bonus.zero_()
        if self._v4_ready and self.cases is not None:
            clock = self.episode_length_buf.float() * self.step_dt
            ids = torch.where((~self.launch_done) &
                              (clock >= self.court_config['rally_preparation_seconds'] - 1e-6))[0]
            if len(ids):
                self.initialize_rally_launch(ids)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sm_prev = {}
        self._sm_reset_events()

    def _sm_reset_events(self, ids=None):
        n = self.num_envs if ids is None else ids.numel()
        device = self.device
        if ids is None:
            ids = torch.arange(self.num_envs, device=device)
        for name in ("_sm_jump", "_sm_contact", "_sm_cross", "_sm_target", "_sm_recovery", "_sm_body"):
            if not hasattr(self, name):
                setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=device))
            getattr(self, name)[ids] = False

    def _get_rewards(self):
        """Progress rates plus one-shot physical events, all from native flags."""
        d = self.robot.data
        rally = self.rally
        phase = self.episode_length_buf.float() * self.step_dt
        contact = rally.legal_initial_contact & rally.string_hit & rally.airborne_string_hit
        attack = contact & rally.released & (rally.outgoing_velocity.norm(dim=1) >= 5.0) & (rally.outgoing_velocity[:, 0] > 0)
        cross = attack & rally.crossed_net
        target = cross & rally.landed_target
        recover = contact & self.rally_recovery_success
        body = self.body_ground_seen
        event_jump = self.flight_qualified & ~self._sm_jump
        event_contact = contact & ~self._sm_contact
        event_cross = cross & ~self._sm_cross
        event_target = target & ~self._sm_target
        event_recovery = recover & ~self._sm_recovery
        event_body = body & ~self._sm_body
        self._sm_jump |= event_jump; self._sm_contact |= event_contact; self._sm_cross |= event_cross
        self._sm_target |= event_target; self._sm_recovery |= event_recovery; self._sm_body |= event_body
        # Bounded progress toward the live racket/ball event; it is not used as
        # an official event label and vanishes after contact or failure.
        ball = self.shuttle.data.root_pos_w
        tool = d.body_link_pos_w[:, self.racket_body_id]
        proximity = torch.exp(-((ball - tool).norm(dim=1) / 0.18).square())
        approach = proximity * (rally.in_play & ~rally.had_contact & ~body).float()
        head_gap = d.body_pos_w[:, self.safety_body_ids, 2].min(1).values.clamp_min(0.)
        risk = torch.exp(-head_gap / 0.04)
        stable_quality = recover.float() * torch.exp(-4.0 * d.projected_gravity_b[:, :2].square().sum(1)
            -0.4 * d.root_com_lin_vel_w[:, :2].square().sum(1)
            -0.2 * d.root_com_ang_vel_w.square().sum(1))
        continuous = (0.5 * stable_quality - 2.0 * risk - 0.05 * self.action_delta
                      - 0.02 * d.applied_torque.square().sum(1) / 1000.0
                      + 0.8 * approach) * self.step_dt
        complete = target & self.rally_recovery_success & (~rally.faulted()) & self.reset_time_outs & (~self.reset_terminated)
        reward = continuous + 2.0 * event_jump.float() + 20.0 * event_contact.float() + 5.0 * event_cross.float()
        reward = reward + 15.0 * event_target.float() + 10.0 * event_recovery.float() + 20.0 * complete.float()
        net_violation = rally.net_blocked_fault | rally.robot_net_fault | rally.body_over_net_fault
        reward = reward - 4.0 * (rally.frame_hit & ~event_contact).float() - 5.0 * ((~contact) & (phase > 3.0)).float() * self.step_dt
        reward = reward - 15.0 * net_violation.float() - 40.0 * event_body.float()
        return torch.nan_to_num(reward, nan=-40.0, posinf=40.0, neginf=-40.0)

    @torch.inference_mode()
    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        self._sm_reset_events(ids)
