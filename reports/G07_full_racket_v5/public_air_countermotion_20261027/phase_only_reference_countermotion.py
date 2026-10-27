"""Bounded post-contact leg motion over an intact measured source controller.

The model predicts internal momentum redistribution at a fixed configuration.
The native actuators must still realize every requested position change; model
predictions are not physical tracking, contact, or recovery outcomes.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import torch

from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'centroidal_recovery_model_20261027'
SCHEMA = 'g07-public-air-countermotion-proposal-v5'


class LegCountermotionRequest:
    def __init__(self, source_asset_sha256, maximum_velocity_change_rad_s):
        if maximum_velocity_change_rad_s not in [0., 2., 5.]:
            raise ValueError('Require a declared bounded leg velocity perturbation')
        check = json.loads((MODEL / 'ACTUAL_MODEL_CHECK.json').read_text())
        for path in [MODEL / 'known_model.py', MODEL / 'MODEL.json']:
            if hashlib.sha256(path.read_bytes()).hexdigest() != check['source_sha256'][str(path)]:
                raise ValueError('Previously validated known model changed')
        spec = importlib.util.spec_from_file_location('countermotion_known_model', MODEL / 'known_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.KnownModel()
        if self.model.record['asset_sha256'] != source_asset_sha256:
            raise ValueError('Known model and actual source physical assets differ')
        self.cap = float(maximum_velocity_change_rad_s)
        self.horizon_s = .03
        self.position_scale_rad = .15
        self.original_leg_velocity_limit_rad_s = 30.
        self.regularization = .05

    def request(self, public, defaults):
        public, defaults = np.asarray(public), np.asarray(defaults)
        if public.ndim != 2 or public.shape[1] != 794 or defaults.shape != (18,):
            raise ValueError('Require the original public794D input and18 joint defaults')
        if not np.isfinite(public).all() or not np.isfinite(defaults).all():
            raise ValueError('Require finite public inputs')
        result = np.zeros((len(public), 18), dtype=np.float32)
        # This observed phase is airborne AND contact_seen. It is not a
        # persistent latch or a legality oracle. No effect before contact or
        # during grounded recovery is prescribed by this air-only prototype.
        active = (public[:, 773] > .5) & (self.cap > 0.)
        rows = np.flatnonzero(active)
        diagnostics = []
        if not len(rows):
            return result, active, diagnostics
        q = defaults[None] + public[rows, 9:27]
        matrix = self.model.angular_momentum_matrix(q)
        for lane, row in enumerate(rows):
            root_angular = public[row, 3:6].astype(np.float64)
            up = -public[row, 6:9].astype(np.float64)
            up /= max(np.linalg.norm(up), 1e-8)
            desired_angular = 2. * np.cross([0., 0., 1.], up)
            desired_angular[2] = root_angular[2]
            locked, leg_matrix = matrix[lane, :, :3], matrix[lane, :, 3:15]
            # Fixed total H = I_locked*w_root + A_joint*qd. A leg velocity
            # increase changes root velocity by -I_locked^-1*A_leg*dqd.
            coupling = -np.linalg.solve(locked, leg_matrix)
            weights = np.array([1., 1., .25])
            a = np.concatenate((weights[:, None]*coupling, self.regularization*np.eye(12)))
            b = np.concatenate((weights*(desired_angular-root_angular), np.zeros(12)))
            measured = public[row, 27:39].astype(np.float64)
            lower = np.maximum(-self.cap, -self.original_leg_velocity_limit_rad_s-measured)
            upper = np.minimum(self.cap, self.original_leg_velocity_limit_rad_s-measured)
            if not (lower < upper).all():
                active[row] = False
                diagnostics.append(dict(lane=int(row), infeasible_velocity_box=True))
                continue
            solution = lsq_linear(a, b, bounds=(lower, upper), method='bvls', tol=1e-10)
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError('Countermotion bounded solve failed')
            delta_velocity = solution.x
            predicted = root_angular+coupling@delta_velocity
            displacement = self.horizon_s*delta_velocity
            result[row, :12] = (displacement/self.position_scale_rad).astype(np.float32)
            if (abs(result[row]) > 1.+1e-6).any():
                raise RuntimeError('Countermotion request exceeds original residual bounds')
            diagnostics.append(dict(lane=int(row), infeasible_velocity_box=False,
                target_root_angular_body_rad_s=desired_angular.tolist(),
                predicted_root_angular_body_rad_s=predicted.tolist(),
                requested_leg_velocity_delta_rad_s=delta_velocity.tolist(),
                requested_leg_position_delta_rad=displacement.tolist(),
                weighted_error_before=float(np.linalg.norm(weights*(root_angular-desired_angular))),
                weighted_error_after=float(np.linalg.norm(weights*(predicted-desired_angular)))))
        return result, active, diagnostics


class AirCountermotionController:
    def __init__(self, source_controller, source_asset_sha256, count, device, maximum_velocity_change_rad_s):
        self.source = source_controller
        self.requester = LegCountermotionRequest(source_asset_sha256, maximum_velocity_change_rad_s)
        self.mixer = ResidualActionMixerV4(count, device, [.15]*18)
        self.last_diagnostic = None

    @property
    def reference_controller(self):
        return self.source.reference_controller

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    def reset(self, ids=None):
        self.source.reset(ids)
        self.mixer.reset(ids)
        if ids is None:
            self.last_diagnostic = None

    @torch.inference_mode()
    def action(self, observation, default, limits):
        public = observation['policy']
        base = self.source.action(observation, default, limits)
        if self.requester.cap == 0.:
            return base
        requests, active, diagnostics = self.requester.request(public.detach().cpu().numpy(), default.detach().cpu().numpy())
        request = torch.as_tensor(requests, device=public.device)
        mixed = self.mixer.action(base, request, public, default, limits)
        result = base.clone()
        mask = torch.as_tensor(active, device=public.device)
        # Preserve all arm commands and inactive leg commands bitwise.
        result[mask, :12] = mixed[mask, :12]
        self.last_diagnostic = dict(active=active, requests=requests, model_predictions=diagnostics)
        return result
