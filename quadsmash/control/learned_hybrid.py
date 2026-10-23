"""Deployable learned walking/strike/standing composition with causal timing."""
import hashlib
import time

import numpy as np
import torch

from .go2_walk_transfer import Go2WalkTransfer
from .go2_standing_transfer import Go2StandingTransfer
from .piecewise_affine import PiecewiseAffinePolicy
from .initial_intercept import predict_initial_intercept


class LearnedHybridPolicy:
    def __init__(self, checkpoint, manifest, count, device, default, limits,
                 physics, calibration=None):
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

    def initialize(self, observation):
        self.walk.reset()
        self.stand.reset()
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
        raw_phase = plan['arrival_time_s']-self.calibration['reference_local_arrival_s']
        raw_phase = np.where(plan['valid'], raw_phase, self.checkpoint['walk_seconds'])
        phase = np.round(raw_phase/.01)*.01
        phase = np.clip(phase, self.calibration['minimum_prepare_s'], self.calibration['maximum_prepare_s'])
        shift_y = plan['position_m'][:, 1]-self.calibration['reference_intercept_y_m']
        shift_y = np.where(plan['valid'], np.clip(shift_y, -.6, .6), 0.)
        self.phase_start.copy_(torch.as_tensor(phase, device=self.phase_start.device, dtype=torch.float32))
        self.waypoints[:, 1].copy_(torch.as_tensor(shift_y, device=self.phase_start.device, dtype=torch.float32))
        self.intercepts[:, 1].add_(self.waypoints[:, 1])
        arrival = np.where(plan['valid'], plan['arrival_time_s'],
                           self.checkpoint['walk_seconds']+self.calibration['reference_local_arrival_s'])
        self.arrival.copy_(torch.as_tensor(arrival, device=self.phase_start.device, dtype=torch.float32))
        self.plan_log = dict(causal_phase=True, input='initial public observation only',
                             public_observation_sha256=hashlib.sha256(public.tobytes()).hexdigest(),
                             computation_wall_s=time.perf_counter()-started,
                             phase_start_s=phase.tolist(), waypoints_m=self.waypoints.cpu().tolist(),
                             **{key: value.tolist() for key, value in plan.items()})

    def command_positions(self, observation):
        started = observation['policy'][:, 775] >= self.phase_start
        return torch.where(started[:, None], self.intercepts, self.waypoints)

    def action(self, observation):
        clock = observation['policy'][:, 775]
        local = observation['policy'].clone()
        local[:, 775] -= self.phase_start
        walk_action = self.walk.action(observation, self.walk_parameters, self.default, self.limits)
        strike_action = self.strike(local)
        action = torch.where((clock >= self.phase_start)[:, None], strike_action, walk_action)
        recovering = local[:, 775] >= self.standing_start
        standing_action = self.stand.action(observation, self.standing_parameters, self.default,
                                            self.limits, self.standing_arm, recovering)
        return torch.where(recovering[:, None], standing_action, action)
