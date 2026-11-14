"""Full Go2+PiPER+racket environment used only by the SuperMimic branch."""
from __future__ import annotations

import torch

from .full_racket_env_v4 import FullRacketEnvV4, full_racket_cfg
from .observation_contract import observe_native
from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4
from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
from quadsmash.rl.supermimic_terminal_scoring import attach_reset_jump_records
from quadsmash.rl.supermimic_native_metrics import NativeReturnAccumulator


def make_supermimic_cfg(num_envs: int, physics_dt: float = 6.25e-5):
    cfg = full_racket_cfg(num_envs, physics_dt)
    cfg.episode_length_s = 8.0
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = num_envs
    return cfg


class SuperMimicEnvV1(FullRacketEnvV4):
    """Schema-specific 18-joint targets with native event-locked rewards."""
    algorithm_family = "supermimic-native-world-model-v1"
    # These latches make one-shot native event rewards Markov for the world
    # model without changing the actor's public 794D observation contract.
    event_state_names = (
        "jump", "contact", "cross", "target", "recovery", "body", "frame", "fault"
    )

    def __init__(self, *args, residual_policy_checkpoint=None, g07_execution_root=None,
                 native_return_gamma=0.99**0.5, **kwargs):
        self._sm_residual_policy_checkpoint = residual_policy_checkpoint
        self._sm_residual_controller = None
        self._sm_residual_default = None
        self._sm_residual_limits = None
        self._sm_requested_action = None
        self._sm_final_action = None
        self._sm_joint_targets = None
        self._sm_g07_execution_root = g07_execution_root
        # Opt-in read-only capture, attached after native reset by the caller.
        self.physical_transition_archive = None
        self._sm_native_returns = None
        super().__init__(*args, **kwargs)
        self._sm_final_event_state = torch.zeros(
            (self.num_envs, len(self.event_state_names)), dtype=torch.float32, device=self.device)
        self._sm_final_event_state_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sm_native_returns = NativeReturnAccumulator(self.num_envs, self.device, native_return_gamma)
        self._sm_prev = {}
        self._sm_reset_events()
        if residual_policy_checkpoint is not None:
            if residual_policy_checkpoint.get('schema') == 'g07-shared-target-residual-ppo-v1':
                self._sm_residual_controller = make_g07_residual_controller(
                    residual_policy_checkpoint, self.num_envs, self.device,
                    execution_root=g07_execution_root)
            else:
                self._sm_residual_controller = FullRacketResidualControllerV4(
                    residual_policy_checkpoint, self.num_envs, self.device)
            self._sm_residual_default = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
            self._sm_residual_limits = self.robot.data.joint_pos_limits[0, self.joint_ids].clone()
            self._sm_residual_controller.reset()

    def supermimic_event_state(self):
        """Return native one-shot event latches before/after a control tick."""
        return torch.stack((
            self._sm_jump, self._sm_contact, self._sm_cross, self._sm_target,
            self._sm_recovery, self._sm_body, self._sm_frame, self._sm_fault,
        ), dim=-1).to(dtype=torch.float32)

    @property
    def final_event_state(self):
        return self._sm_final_event_state

    @property
    def final_event_state_valid(self):
        return self._sm_final_event_state_valid

    def step(self, actions):
        self._sm_final_event_state_valid.zero_()
        result = super().step(actions)
        self.extras["final_event_state"] = self._sm_final_event_state
        self.extras["final_event_state_valid"] = self._sm_final_event_state_valid
        return result

    def _pre_physics_step(self, actions):
        """Keep residual-v4 and legacy jump-event action contracts separate.

        A residual-v4 mixer already emits a normalized full-range command.
        Decode it with FullRacketEnvV4, never with default+phase_scale*action.
        The legacy direct jump-event branch below retains its own contract.
        """
        self._sm_requested_action = actions.detach().clone()
        if self._sm_residual_controller is not None:
            public = observe_native(self)
            if self.physical_transition_archive is not None:
                self.physical_transition_archive.begin(self, actions, public['policy'])
            if hasattr(self._sm_residual_controller, 'apply_residual'):
                final = self._sm_residual_controller.apply_residual(
                    public, actions, self._sm_residual_default,
                    self._sm_residual_limits).detach()
                if self.physical_transition_archive is not None:
                    self._sm_base_action = self._sm_residual_controller.last_base_action.detach().clone()
            else:
                base = self._sm_residual_controller.base.action(
                    public, self._sm_residual_default, self._sm_residual_limits)
                if self.physical_transition_archive is not None:
                    self._sm_base_action = base.detach().clone()
                final = self._sm_residual_controller.mixer.action(
                    base, actions, public['policy'], self._sm_residual_default,
                    self._sm_residual_limits).detach()
            self._sm_final_action = final.clone()
            FullRacketEnvV4._pre_physics_step(self, final)
            self._sm_joint_targets = self.targets[:, self.joint_ids].detach().clone()
            return
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
        self._sm_joint_targets = self.targets[:, self.joint_ids].detach().clone()
        self._contact_bonus.zero_()
        if self._v4_ready and self.cases is not None:
            clock = self.episode_length_buf.float() * self.step_dt
            ids = torch.where((~self.launch_done) &
                              (clock >= self.court_config['rally_preparation_seconds'] - 1e-6))[0]
            if len(ids):
                self.initialize_rally_launch(ids)

    def _sm_reset_events(self, ids=None):
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
        complete5 = target & self.rally_recovery_success & (~faulted) & self.reset_time_outs & (~self.reset_terminated)
        reward = continuous + 2.0 * event_jump.float() + 20.0 * event_contact.float() + 5.0 * event_cross.float()
        reward = reward + 15.0 * event_target.float() + 10.0 * event_recovery.float() + 20.0 * complete5.float()
        reward = reward - 4.0 * event_frame.float() - 5.0 * ((~contact) & (phase > 3.0)).float() * self.step_dt
        reward = reward - 15.0 * event_fault.float() - 40.0 * event_body.float()
        reward = torch.nan_to_num(reward, nan=-40.0, posinf=40.0, neginf=-40.0)
        self._sm_native_returns.add(reward)
        if self.physical_transition_archive is not None:
            # Record the existing computation; do not call reward/latch logic twice.
            self._sm_reward_components = dict(
                continuous=continuous, jump=2.0*event_jump.float(),
                qualified_contact=20.0*event_contact.float(), cross=5.0*event_cross.float(),
                target=15.0*event_target.float(), recovery=10.0*event_recovery.float(),
                complete5_terminal=20.0*complete5.float(), frame=-4.0*event_frame.float(),
                miss_rate=-5.0*((~contact)&(phase>3.0)).float()*self.step_dt,
                fault=-15.0*event_fault.float(), body=-40.0*event_body.float())
            # DirectRLEnv obtains reward before resetting terminated lanes.
            # Capturing after wrapper.step would join terminal states to resets.
            self.physical_transition_archive.finish(self, reward)
        return reward

    @torch.inference_mode()
    def _reset_idx(self, ids):
        terminal = {}
        if self._inside_step and self._v4_ready and ids.numel():
            qualified = (self.rally.legal_initial_contact & self.rally.string_hit
                         & self.rally.airborne_string_hit)
            ball_ok = self.rally.complete5(torch.ones_like(self.rally_recovery_success))
            faults = self.rally.faulted()
            return_records = self._sm_native_returns.records(ids)
            for i in ids[self._episode_has_stepped[ids]].cpu().tolist():
                terminal[(i, int(self.episode_index[i]))] = dict(
                    qualified_airborne_string_contact=bool(qualified[i]),
                    ball_side_complete_except_recovery=bool(ball_ok[i]),
                    body_ground_seen=bool(self.body_ground_seen[i]),
                    rally_fault=bool(faults[i]),
                    head_contact=None, independent_nonfoot_contact=None,
                    event_source='native_pre_reset_rally_and_contact_flight_v1',
                    terminal_control_tick=int(self.common_step_counter),
                    **return_records[i])
        record_start = len(self.rally_records) if self._v4_ready else 0
        jump_start = len(self.episode_records) if self._v4_ready else 0
        if self._inside_step and ids.numel():
            self._sm_final_event_state[ids] = self.supermimic_event_state()[ids]
            self._sm_final_event_state_valid[ids] = True
        super()._reset_idx(ids)
        # Both lists were populated before the parent reset cleared physics
        # metrics. Join only this reset batch, not the latest record per lane.
        attach_reset_jump_records(self.rally_records[record_start:],
                                  self.episode_records[jump_start:])
        for row in self.rally_records[record_start:]:
            key = (row['env_id'], row['episode_index'])
            if key not in terminal:
                raise RuntimeError('Completed episode has no pre-reset task evidence')
            row.update(terminal[key])
            row['ground_init'] = True  # This env uses the native ground reset, no state teleport.
            row['case_id'] = (self.cases[row['env_id']]['case_id']
                              if self.cases is not None else f"lane-{row['env_id']}")
        if not self._inside_step:
            self._sm_final_event_state_valid[ids] = False
        self._sm_reset_events(ids)
        if self._sm_native_returns is not None:
            self._sm_native_returns.reset(ids)
        if self._sm_residual_controller is not None:
            self._sm_residual_controller.reset(ids)
