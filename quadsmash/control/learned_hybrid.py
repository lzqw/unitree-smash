"""Deployable learned walking/strike/standing composition with causal timing."""
import hashlib
import time

import numpy as np
import torch

from .go2_walk_transfer import Go2WalkTransfer
from .go2_standing_transfer import Go2StandingTransfer
from .piecewise_affine import PiecewiseAffinePolicy
from .initial_intercept import predict_initial_intercept, observed_world_heading
from .causal_parameter_selector import CausalParameterSelector


class LearnedHybridPolicy:
    def __init__(self, checkpoint, manifest, count, device, default, limits,
                 physics, calibration=None, planner_parameters=None):
        self.checkpoint = checkpoint
        self.default, self.limits = default, limits
        self.physics, self.calibration = physics, calibration
        self.walk = Go2WalkTransfer(manifest, count, device)
        self.stand = Go2StandingTransfer(manifest, count, device)
        self.strike = PiecewiseAffinePolicy(checkpoint['strike_policy_checkpoint']).to(device).eval()
        self.walk_parameters = checkpoint['walk_parameters'].to(device).repeat(count, 1)
        self.standing_parameters = checkpoint['recovery_parameters'].to(device).repeat(count, 1)
        self.standing_arm = checkpoint['recovery_arm_target'].to(device)
        self.standing_start = checkpoint['recovery_start_s']
        self.phase_start = torch.full((count,), checkpoint['walk_seconds'], device=device)
        self.waypoints = torch.tensor([checkpoint['walk_waypoint_x'], 0., .3], device=device).repeat(count, 1)
        self.intercepts = torch.tensor(checkpoint['result']['virtual_ball_center_m'], device=device).repeat(count, 1)
        self.arrival = self.phase_start.clone()
        self.plan_log = None
        initial = checkpoint.get('planner_parameters', torch.tensor([0.]*10+[1.]))
        self.planner_parameters = initial.to(device).repeat(count, 1) if planner_parameters is None else planner_parameters.clone()
        self.arm_pitch_parameters = checkpoint.get('arm_pitch_parameters', torch.zeros(4)).to(device).repeat(count, 1)
        self.arm_pitch_basis = checkpoint.get('arm_pitch_basis', torch.zeros(6)).to(device)
        self.arm_pitch_amplitude = torch.zeros(count, device=device)
        self.has_arm_residual = bool(self.arm_pitch_basis.abs().sum() > 0)
        original_preparation = torch.stack((checkpoint['walk_parameters'][0], checkpoint['walk_parameters'][14],
                                            torch.tensor(0.), checkpoint['walk_parameters'][7]))
        self.preparation_parameters = checkpoint.get('preparation_parameters', original_preparation).to(device).repeat(count, 1)
        self.has_preparation_adaptation = 'preparation_parameters' in checkpoint
        self.replan_at_s = checkpoint.get('replan_at_s')
        self.replanned = torch.zeros(count, dtype=torch.bool, device=device)
        self.replan_log = []
        reference = checkpoint.get('body_reference')
        self.body_reference_times = reference['times_s'].to(device) if reference else None
        self.body_reference_values = reference['pitch_omega_y_vx'].to(device) if reference else None
        self.body_feedback_parameters = checkpoint.get('body_feedback_parameters', torch.zeros(3)).to(device).repeat(count, 1)
        self.body_feedback_mode = checkpoint.get('body_feedback_mode', 'differential-v1')
        self.body_velocity_bias = checkpoint.get('body_velocity_reference_bias_mps', 0.)
        heading_reference = checkpoint.get('arm_heading_reference')
        self.arm_heading_times = heading_reference['times_s'].to(device) if heading_reference else None
        self.arm_heading_values = heading_reference['heading_omega_z'].to(device) if heading_reference else None
        self.arm_heading_gains = checkpoint.get('arm_heading_gains', torch.zeros(2)).to(device).repeat(count, 1)
        self.arm_joint_residual = checkpoint.get('arm_joint_position_residual_rad', torch.zeros(6)).to(device).repeat(count, 1)
        self.arm_pitch_bias = torch.full((count,), checkpoint.get('arm_pitch_output_bias_rad', 0.), device=device)
        self.selector = CausalParameterSelector(checkpoint['causal_parameter_selector'], device) if 'causal_parameter_selector' in checkpoint else None
        self.selection_at_s = self.selector.time_s if self.selector else None
        self.selected_parameters = torch.zeros(count, dtype=torch.bool, device=device)
        self.selection_log = []
        self.contact_recovery_settings = checkpoint.get('contact_recovery')
        self.observed_contact_time = torch.full((count,), float('inf'), device=device)
        self.contact_recovery_log = []
        self.recovery_wait_for_landing = checkpoint.get('recovery_wait_for_landing', False)
        self.landing_observed = torch.zeros(count, dtype=torch.bool, device=device)
        if self.contact_recovery_settings is not None:
            settings = self.contact_recovery_settings
            if settings.get('arm_blend_s', .08)<=0 or settings.get('arm_delay_s', 0.)<0 or settings.get('leg_delay_s', 0.)<0:
                raise ValueError('Contact recovery must use positive blend and nonnegative causal delays')

    def initialize(self, observation):
        self.walk.reset()
        self.stand.reset()
        self.arm_pitch_amplitude.fill_(self.checkpoint.get('arm_pitch_output_bias_rad', 0.))
        self.arm_pitch_bias.fill_(self.checkpoint.get('arm_pitch_output_bias_rad', 0.))
        self.body_feedback_parameters.copy_(self.checkpoint.get('body_feedback_parameters', torch.zeros(3)).to(self.body_feedback_parameters).expand_as(self.body_feedback_parameters))
        self.arm_heading_gains.copy_(self.checkpoint.get('arm_heading_gains', torch.zeros(2)).to(self.arm_heading_gains).expand_as(self.arm_heading_gains))
        self.arm_joint_residual.copy_(self.checkpoint.get('arm_joint_position_residual_rad', torch.zeros(6)).to(self.arm_joint_residual).expand_as(self.arm_joint_residual))
        self.selected_parameters.zero_()
        self.selection_log = []
        self.observed_contact_time.fill_(float('inf'))
        self.contact_recovery_log = []
        self.landing_observed.zero_()
        self.replanned.zero_()
        self.replan_log = []
        if self.has_preparation_adaptation:
            self.walk_parameters[:, 0].copy_(self.preparation_parameters[:, 0])
            self.walk_parameters[:, 14].copy_(self.preparation_parameters[:, 1])
            self.walk_parameters[:, 7].copy_(self.preparation_parameters[:, 3])
        self.phase_start.fill_(self.checkpoint['walk_seconds'])
        self.waypoints.copy_(torch.tensor([self.checkpoint['walk_waypoint_x'], 0., .3],
                                         device=self.phase_start.device))
        self.waypoints[:, 0].add_(self.checkpoint.get('preparation_waypoint_offset_x_m', 0.))
        self.intercepts.copy_(torch.tensor(self.checkpoint['result']['virtual_ball_center_m'],
                                          device=self.phase_start.device))
        if self.calibration is None:
            self.plan_log = dict(causal_phase=False)
            return
        started = time.perf_counter()
        public = observation['policy'].detach().cpu().numpy()
        plan = predict_initial_intercept(
            public, [self.checkpoint['walk_waypoint_x'], 0., .3],
            self.checkpoint['result']['face_normal_world'],
            self.calibration['intercept_x_m'], self.physics)
        weights = self.planner_parameters.detach().cpu().numpy()
        reference_time = self.checkpoint['walk_seconds']+self.calibration['reference_local_arrival_s']
        reference_vx = self.checkpoint.get('planner_feature_reference', {}).get('incoming_vx_mps', -3.2652405806200373)
        reference_z = self.checkpoint['result']['virtual_ball_center_m'][2]

        def features(prediction):
            return np.c_[np.ones(len(public)), prediction['arrival_time_s']-reference_time,
                         prediction['incoming_velocity_mps'][:, 0]-reference_vx,
                         prediction['position_m'][:, 2]-reference_z,
                         prediction['position_m'][:, 1]-self.calibration['reference_intercept_y_m']]

        x_features = features(plan)
        shift_x = np.clip(np.nansum(weights[:, 4:9]*x_features, axis=1), -.08, .08)
        if np.any(np.abs(shift_x) > 1e-9):
            plan = predict_initial_intercept(public, [self.checkpoint['walk_waypoint_x'], 0., .3],
                self.checkpoint['result']['face_normal_world'],
                self.calibration['intercept_x_m']+shift_x, self.physics)
        phase_features = features(plan)[:, :4]
        reference_vz = self.checkpoint.get('planner_feature_reference', {}).get('incoming_vz_mps', -16.89315645360073)
        pitch_features = np.c_[np.ones(len(public)),
            np.maximum(plan['incoming_velocity_mps'][:, 0]-reference_vx, 0.),
            np.minimum(plan['incoming_velocity_mps'][:, 2]-reference_vz, 0.),
            np.minimum(plan['position_m'][:, 2]-reference_z, 0.)]
        pitch_weights = self.arm_pitch_parameters.detach().cpu().numpy()
        pitch = np.clip(np.nansum(pitch_weights*pitch_features, axis=1), -.05, .04)
        pitch = np.where(plan['valid'], pitch, 0.)
        pitch += self.checkpoint.get('arm_pitch_output_bias_rad', 0.)
        self.arm_pitch_amplitude.copy_(torch.as_tensor(pitch, device=self.phase_start.device, dtype=torch.float32))
        phase_correction = np.nansum(weights[:, :4]*phase_features, axis=1)
        raw_phase = plan['arrival_time_s']-self.calibration['reference_local_arrival_s']
        raw_phase += phase_correction
        raw_phase = np.where(plan['valid'], raw_phase, self.checkpoint['walk_seconds'])
        phase = np.round(raw_phase/.01)*.01
        phase = np.clip(phase, self.calibration['minimum_prepare_s'], self.calibration['maximum_prepare_s'])
        shift_y = plan['position_m'][:, 1]-self.calibration['reference_intercept_y_m']
        shift_y = weights[:, 9]+weights[:, 10]*shift_y
        shift_y = np.where(plan['valid'], np.clip(shift_y, -.6, .6), 0.)
        self.phase_start.copy_(torch.as_tensor(phase, device=self.phase_start.device, dtype=torch.float32))
        self.waypoints[:, 0].add_(torch.as_tensor(shift_x, device=self.phase_start.device, dtype=torch.float32))
        self.waypoints[:, 1].copy_(torch.as_tensor(shift_y, device=self.phase_start.device, dtype=torch.float32))
        self.intercepts[:, 0].add_(torch.as_tensor(shift_x, device=self.phase_start.device, dtype=torch.float32))
        self.intercepts[:, 1].add_(self.waypoints[:, 1])
        arrival = np.where(plan['valid'], plan['arrival_time_s'],
                           self.checkpoint['walk_seconds']+self.calibration['reference_local_arrival_s'])
        self.arrival.copy_(torch.as_tensor(arrival, device=self.phase_start.device, dtype=torch.float32))
        self.plan_log = dict(causal_phase=True, input='initial public observation only',
                             public_observation_sha256=hashlib.sha256(public.tobytes()).hexdigest(),
                             computation_wall_s=time.perf_counter()-started,
                             phase_start_s=phase.tolist(), waypoints_m=self.waypoints.cpu().tolist(),
                             planner_parameters=weights.tolist(),
                             arm_pitch_correction_rad=pitch.tolist(),
                             **{key: value.tolist() for key, value in plan.items()})

    def command_positions(self, observation):
        started = observation['policy'][:, 775] >= self.phase_start
        return torch.where(started[:, None], self.intercepts, self.waypoints)

    def action(self, observation):
        self._maybe_select_parameters(observation)
        self._maybe_replan(observation)
        clock = observation['policy'][:, 775]
        if self.contact_recovery_settings is not None:
            first = (observation['policy'][:, 773]>.5)&torch.isinf(self.observed_contact_time)
            if bool(first.any()):
                self.observed_contact_time[first] = clock[first]
                self.contact_recovery_log.append(dict(input='current public flight-post-contact phase only',
                                                      env_ids=torch.where(first)[0].cpu().tolist(),
                                                      first_observed_clock_s=clock[first].cpu().tolist()))
        local = observation['policy'].clone()
        local[:, 775] -= self.phase_start
        if self.has_preparation_adaptation:
            heading = observed_world_heading(observation['policy'], self.checkpoint['result']['face_normal_world'])
            heading_kp = self.preparation_parameters[:, 2]
            extended = self.checkpoint.get('preparation_readiness_extension', True)
            maximum_time = torch.where(heading_kp>0, 4.5, 3.4) if extended else 3.4
            walk_action = self.walk.action(observation, self.walk_parameters, self.default, self.limits,
                heading_error=-heading, heading_kp=heading_kp, maximum_time=maximum_time,
                heading_readiness=extended)
        else:
            walk_action = self.walk.action(observation, self.walk_parameters, self.default, self.limits)
        strike_action = self.strike(local)
        if self.body_reference_times is not None:
            time_local = local[:, 775].contiguous()
            index = torch.searchsorted(self.body_reference_times, time_local).clamp(1, len(self.body_reference_times)-1)
            left, right = self.body_reference_times[index-1], self.body_reference_times[index]
            fraction = ((time_local-left)/(right-left)).clamp(0, 1)
            reference = torch.lerp(self.body_reference_values[index-1], self.body_reference_values[index], fraction[:, None])
            measured = torch.stack((torch.atan2(local[:, 6], -local[:, 8]), local[:, 4], local[:, 0]), 1)
            envelope = ((time_local-.75)/.1).clamp(0, 1)*((1.30-time_local)/.1).clamp(0, 1)
            if self.body_feedback_mode == 'differential-pitch-common-vx-v2':
                pitch = ((measured[:, :2]-reference[:, :2])*self.body_feedback_parameters[:, :2]).sum(1).clamp(-.12, .12)*envelope
                vx_envelope = ((time_local-1.)/.04).clamp(0, 1)*((1.25-time_local)/.07).clamp(0, 1)
                common = (self.body_feedback_parameters[:, 2]*(measured[:, 2]-reference[:, 2]-self.body_velocity_bias)).clamp(-.12, .12)*vx_envelope
                front, rear = (pitch+common).clamp(-.12, .12), (-pitch+common).clamp(-.12, .12)
            elif self.body_feedback_mode == 'differential-v1':
                delta = ((measured-reference)*self.body_feedback_parameters).sum(1).clamp(-.12, .12)
                front, rear = delta*envelope, -delta*envelope
            else:
                raise ValueError('Unknown body feedback mode')
            strike_action[:, 4:6] += 2*front[:, None]/(self.limits[4:6, 1]-self.limits[4:6, 0])
            strike_action[:, 6:8] += 2*rear[:, None]/(self.limits[6:8, 1]-self.limits[6:8, 0])
            strike_action.clamp_(-1, 1)
        push_delta = self.checkpoint.get('vertical_push_calf_delta_rad', 0.)
        if push_delta:
            a, b, c, d = self.checkpoint.get('vertical_push_window_s', [.98, 1.04, 1.16, 1.24])
            if not a<b<=c<d:
                raise ValueError('Invalid bounded vertical-push timing window')
            envelope = ((local[:, 775]-a)/(b-a)).clamp(0, 1)*((d-local[:, 775])/(d-c)).clamp(0, 1)
            delta = push_delta*envelope[:, None]
            strike_action[:, 4:8] -= delta/(self.limits[4:8, 1]-self.limits[4:8, 0])
            strike_action[:, 8:12] += 2*delta/(self.limits[8:12, 1]-self.limits[8:12, 0])
            strike_action.clamp_(-1, 1)
        if self.has_arm_residual:
            envelope = ((local[:, 775]-.8)/.35).clamp(0, 1)*((1.6-local[:, 775])/.18).clamp(0, 1)
            delta = (self.arm_pitch_amplitude*envelope)[:, None]*self.arm_pitch_basis[None, :]
            strike_action[:, 12:] = (strike_action[:, 12:]+2*delta/(self.limits[12:, 1]-self.limits[12:, 0])).clamp(-1, 1)
        if 'arm_joint_position_residual_rad' in self.checkpoint or self.selector is not None:
            envelope = ((local[:, 775]-.8)/.35).clamp(0, 1)*((1.6-local[:, 775])/.18).clamp(0, 1)
            strike_action[:, 12:] = (strike_action[:, 12:]+2*envelope[:, None]*self.arm_joint_residual /
                                     (self.limits[12:, 1]-self.limits[12:, 0])).clamp(-1, 1)
        if self.arm_heading_times is not None:
            time_local = local[:, 775].contiguous()
            index = torch.searchsorted(self.arm_heading_times, time_local).clamp(1, len(self.arm_heading_times)-1)
            fraction = ((time_local-self.arm_heading_times[index-1]) /
                        (self.arm_heading_times[index]-self.arm_heading_times[index-1])).clamp(0, 1)
            reference = torch.lerp(self.arm_heading_values[index-1], self.arm_heading_values[index], fraction[:, None])
            heading = observed_world_heading(observation['policy'], self.checkpoint['result']['face_normal_world'])
            error = torch.atan2(torch.sin(heading-reference[:, 0]), torch.cos(heading-reference[:, 0]))
            delta = -(self.arm_heading_gains[:, 0]*error+self.arm_heading_gains[:, 1]*(local[:, 5]-reference[:, 1]))
            envelope = ((time_local-.8)/.25).clamp(0, 1)*((1.6-time_local)/.18).clamp(0, 1)
            delta = delta.clamp(-.10, .10)*envelope
            strike_action[:, 12] = (strike_action[:, 12]+2*delta/(self.limits[12, 1]-self.limits[12, 0])).clamp(-1, 1)
        action = torch.where((clock >= self.phase_start)[:, None], strike_action, walk_action)
        recovering = local[:, 775] >= self.standing_start
        if self.recovery_wait_for_landing:
            self.landing_observed |= (observation['policy'][:, 774]>.5)&(local[:, 775]>=1.25)
            recovering &= self.landing_observed
        if self.contact_recovery_settings is not None and 'leg_delay_s' in self.contact_recovery_settings:
            recovering |= clock>=self.observed_contact_time+self.contact_recovery_settings['leg_delay_s']
        standing_action = self.stand.action(observation, self.standing_parameters, self.default,
                                            self.limits, self.standing_arm, recovering)
        result = torch.where(recovering[:, None], standing_action, action)
        if self.contact_recovery_settings is not None:
            settings = self.contact_recovery_settings
            blend = ((clock-self.observed_contact_time-settings.get('arm_delay_s', 0.)) /
                     settings.get('arm_blend_s', .08)).clamp(0, 1)
            result[:, 12:] = torch.lerp(result[:, 12:], standing_action[:, 12:], blend[:, None])
        return result

    def _maybe_select_parameters(self, observation):
        if self.selector is None:
            return
        clock = observation['policy'][:, 775]
        ids = torch.where((clock>=self.selection_at_s)&(~self.selected_parameters)&(clock<self.phase_start+.7))[0]
        if not len(ids):
            return
        choice, scores = self.selector.choose(observation['policy'][ids])
        tables = self.selector.tables
        new_bias = tables['arm_pitch_bias'][choice]
        self.arm_pitch_amplitude[ids] += new_bias-self.arm_pitch_bias[ids]
        self.arm_pitch_bias[ids] = new_bias
        self.body_feedback_parameters[ids] = tables['body_feedback'][choice]
        self.arm_heading_gains[ids] = tables['arm_heading_gains'][choice]
        self.arm_joint_residual[ids] = tables['arm_joint_residual'][choice]
        self.selected_parameters[ids] = True
        self.selection_log.append(dict(input='current public observation only', env_ids=ids.cpu().tolist(),
                                       clock_s=clock[ids].cpu().tolist(), option=choice.cpu().tolist(),
                                       scores=scores.cpu().tolist()))

    def _maybe_replan(self, observation):
        if self.replan_at_s is None or self.calibration is None:
            return
        clock = observation['policy'][:, 775]
        ids = torch.where((clock>=self.replan_at_s)&(~self.replanned)&(clock<self.phase_start))[0]
        if not len(ids):
            return
        started = time.perf_counter()
        public = observation['policy'][ids].detach().cpu().numpy()
        plan = predict_initial_intercept(public, self.waypoints[ids].cpu().numpy(),
            self.checkpoint['result']['face_normal_world'], self.intercepts[ids, 0].cpu().numpy(),
            self.physics, max_seconds=3.)
        plan['arrival_time_s'] += public[:, 775]
        reference = self.checkpoint.get('planner_feature_reference', {})
        vx = reference.get('incoming_vx_mps', -3.2652405806200373)
        vz = reference.get('incoming_vz_mps', -16.89315645360073)
        z = self.checkpoint['result']['virtual_ball_center_m'][2]
        feature = np.c_[np.ones(len(public)), plan['arrival_time_s']-self.checkpoint['walk_seconds']-
            self.calibration['reference_local_arrival_s'], plan['incoming_velocity_mps'][:, 0]-vx,
            plan['position_m'][:, 2]-z]
        weights = self.planner_parameters[ids].cpu().numpy()
        raw = plan['arrival_time_s']-self.calibration['reference_local_arrival_s']+np.nansum(weights[:, :4]*feature, axis=1)
        phase = np.clip(np.round(raw/.01)*.01, self.calibration['minimum_prepare_s'], self.calibration['maximum_prepare_s'])
        before = self.phase_start[ids].cpu().numpy()
        phase = np.where(plan['valid'], np.maximum(phase, public[:, 775]+.01), before)
        pitch_feature = np.c_[np.ones(len(public)), np.maximum(plan['incoming_velocity_mps'][:, 0]-vx, 0.),
            np.minimum(plan['incoming_velocity_mps'][:, 2]-vz, 0.), np.minimum(plan['position_m'][:, 2]-z, 0.)]
        pitch = np.clip(np.nansum(self.arm_pitch_parameters[ids].cpu().numpy()*pitch_feature, axis=1), -.05, .04)
        pitch += self.arm_pitch_bias[ids].cpu().numpy()
        pitch = np.where(plan['valid'], pitch, self.arm_pitch_amplitude[ids].cpu().numpy())
        arrival = np.where(plan['valid'], plan['arrival_time_s'], self.arrival[ids].cpu().numpy())
        self.phase_start[ids] = torch.as_tensor(phase, dtype=clock.dtype, device=clock.device)
        self.arrival[ids] = torch.as_tensor(arrival, dtype=clock.dtype, device=clock.device)
        self.arm_pitch_amplitude[ids] = torch.as_tensor(pitch, dtype=clock.dtype, device=clock.device)
        self.replanned[ids] = True
        self.replan_log.append(dict(input='current public observation only; before strike, positions unchanged',
            env_ids=ids.cpu().tolist(), observation_clock_s=public[:, 775].tolist(),
            phase_before_s=before.tolist(), phase_after_s=phase.tolist(), computation_wall_s=time.perf_counter()-started,
            **{key: value.tolist() for key, value in plan.items()}))
