"""Full Go2+PiPER+racket environment used only by the SuperMimic branch."""
from __future__ import annotations

import torch

from .full_racket_env_v4 import FullRacketEnvV4, full_racket_cfg
from .observation_contract import observe_native
from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4


def make_supermimic_cfg(num_envs: int, physics_dt: float = 6.25e-5):
    cfg = full_racket_cfg(num_envs, physics_dt)
    cfg.episode_length_s = 8.0
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = num_envs
    return cfg


class SuperMimicEnvV1(FullRacketEnvV4):
    """Absolute 18-joint targets with native event-locked reward shaping."""
    algorithm_family = "supermimic-native-world-model-v1"

    def __init__(self, *args, residual_policy_checkpoint=None, **kwargs):
        self._sm_residual_policy_checkpoint = residual_policy_checkpoint
        self._sm_residual_controller = None
        self._sm_residual_default = None
        self._sm_residual_limits = None
        self._sm_requested_action = None
        self._sm_final_action = None
        super().__init__(*args, **kwargs)
        self._sm_prev = {}
        self._sm_reset_events()
        if residual_policy_checkpoint is not None:
            self._sm_residual_controller = FullRacketResidualControllerV4(
                residual_policy_checkpoint, self.num_envs, self.device)
            self._sm_residual_default = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
            self._sm_residual_limits = self.robot.data.joint_pos_limits[0, self.joint_ids].clone()
            self._sm_residual_controller.reset()

    def _pre_physics_step(self, actions):
        """Use the exact native jump-event-v1 command contract.

        The PPO checkpoint was trained with phase-dependent bounded position
        targets around the default pose.  Mapping the same action to the full
        joint limits changes the physical policy even when the network weights
        are identical, so this branch intentionally mirrors JumpEnv.
        """
        self._sm_requested_action = actions.detach().clone()
        if self._sm_residual_controller is not None:
            # The trainable actor emits the residual request. The frozen
            # full-racket base and native residual mixer produce the exact
            # normalized final command consumed by the original executor.
            public = observe_native(self)
            base = self._sm_residual_controller.base.action(
                public, self._sm_residual_default, self._sm_residual_limits)
            actions = self._sm_residual_controller.mixer.action(
                base, actions, public['policy'], self._sm_residual_default,
                self._sm_residual_limits)
            actions = actions.detach()
        self._sm_final_action = actions.detach().clone()
        self.action_delta.copy_((actions.clamp(-1, 1) - self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count += (actions.abs() > 1).sum(1)
        phase = (self.episode_length_buf.float() * self.step_dt).unsqueeze(1)
        leg_scale = torch.where(phase < .8, self.cfg.leg_action_scale_crouch,
                                self.cfg.leg_action_scale_push)
        scale = torch.cat((leg_scale.expand(-1, 12),
                           torch.full((self.num_envs, 6), self.cfg.arm_action_scale,
                                      device=self.device)), 1)
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:, self.joint_ids] += scale * self.last_action
        limits = self.robot.data.joint_pos_limits
        self.target_limit_count += ((self.targets < limits[:, :, 0]) |
                                    (self.targets > limits[:, :, 1])).sum(1)
        self.targets.clamp_(min=limits[:, :, 0], max=limits[:, :, 1])
        self._contact_bonus.zero_()
        if self._v4_ready and self.cases is not None:
            clock = self.episode_length_buf.float() * self.step_dt
            ids = torch.where((~self.launch_done) &
                              (clock >= self.court_config['rally_preparation_seconds'] - 1e-6))[0]
            if len(ids):
                self.initialize_rally_launch(ids)

    def _sm_reset_events(self, ids=None):
        n = self.num_envs if ids is None else ids.numel()
        device = self.device
        if ids is None:
            ids = torch.arange(self.num_envs, device=device)
        for name in ("_sm_jump", "_sm_contact", "_sm_cross", "_sm_target", "_sm_recovery",
                     "_sm_body", "_sm_frame", "_sm_fault"):
            if not hasattr(self, name):
                setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=device))
            getattr(self, name)[ids] = False

    def _get_rewards(self):
        """Progress rates plus one-shot physical events, all from native flags."""
        # JumpEnv uses this latch to decide whether a reset corresponds to a
        # completed episode.  The SuperMimic reward must preserve that
        # lifecycle contract instead of replacing it with a parallel counter.
        self._episode_has_stepped[:] = True
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
        faulted = rally.faulted()
        event_fault = faulted & ~self._sm_fault
        event_frame = rally.frame_hit & ~event_contact & ~self._sm_frame
        self._sm_jump |= event_jump; self._sm_contact |= event_contact; self._sm_cross |= event_cross
        self._sm_target |= event_target; self._sm_recovery |= event_recovery; self._sm_body |= event_body
        self._sm_frame |= event_frame; self._sm_fault |= event_fault
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
        # The current task contract is the 5 m/s attack threshold.  Keep this
        # event separate from RallyRulesV4's historical 8 m/s attack helper.
        complete5 = target & self.rally_recovery_success & (~faulted) & self.reset_time_outs & (~self.reset_terminated)
        reward = continuous + 2.0 * event_jump.float() + 20.0 * event_contact.float() + 5.0 * event_cross.float()
        reward = reward + 15.0 * event_target.float() + 10.0 * event_recovery.float() + 20.0 * complete5.float()
        net_violation = rally.net_blocked_fault | rally.robot_net_fault | rally.body_over_net_fault
        reward = reward - 4.0 * event_frame.float() - 5.0 * ((~contact) & (phase > 3.0)).float() * self.step_dt
        reward = reward - 15.0 * event_fault.float() - 40.0 * event_body.float()
        return torch.nan_to_num(reward, nan=-40.0, posinf=40.0, neginf=-40.0)

    @torch.inference_mode()
    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        self._sm_reset_events(ids)
        if self._sm_residual_controller is not None:
            self._sm_residual_controller.reset(ids)
