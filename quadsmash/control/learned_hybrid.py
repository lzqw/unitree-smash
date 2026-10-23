"""Deployable learned walking/strike/standing composition with causal timing."""
import hashlib
import time

import numpy as np
import torch

from .go2_walk_transfer import Go2WalkTransfer
from .go2_standing_transfer import Go2StandingTransfer
from .piecewise_affine import PiecewiseAffinePolicy
from .initial_intercept import predict_initial_intercept, observed_world_heading


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

    def initialize(self, observation):
        self.walk.reset()
        self.stand.reset()
        self.arm_pitch_amplitude.zero_()
        if self.has_preparation_adaptation:
            self.walk_parameters[:, 0].copy_(self.preparation_parameters[:, 0])
            self.walk_parameters[:, 14].copy_(self.preparation_parameters[:, 1])
            self.walk_parameters[:, 7].copy_(self.preparation_parameters[:, 3])
        self.phase_start.fill_(self.checkpoint['walk_seconds'])
        self.waypoints.copy_(torch.tensor([self.checkpoint['walk_waypoint_x'], 0., .3],
                                         device=self.phase_start.device))
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
        clock = observation['policy'][:, 775]
        local = observation['policy'].clone()
        local[:, 775] -= self.phase_start
        if self.has_preparation_adaptation:
            heading = observed_world_heading(observation['policy'], self.checkpoint['result']['face_normal_world'])
            heading_kp = self.preparation_parameters[:, 2]
            maximum_time = torch.where(heading_kp>0, 4.5, 3.4)
            walk_action = self.walk.action(observation, self.walk_parameters, self.default, self.limits,
                heading_error=-heading, heading_kp=heading_kp, maximum_time=maximum_time)
        else:
            walk_action = self.walk.action(observation, self.walk_parameters, self.default, self.limits)
        strike_action = self.strike(local)
        if self.has_arm_residual:
            envelope = ((local[:, 775]-.8)/.35).clamp(0, 1)*((1.6-local[:, 775])/.18).clamp(0, 1)
            delta = (self.arm_pitch_amplitude*envelope)[:, None]*self.arm_pitch_basis[None, :]
            strike_action[:, 12:] = (strike_action[:, 12:]+2*delta/(self.limits[12:, 1]-self.limits[12:, 0])).clamp(-1, 1)
        action = torch.where((clock >= self.phase_start)[:, None], strike_action, walk_action)
        recovering = local[:, 775] >= self.standing_start
        standing_action = self.stand.action(observation, self.standing_parameters, self.default,
                                            self.limits, self.standing_arm, recovering)
        return torch.where(recovering[:, None], standing_action, action)
