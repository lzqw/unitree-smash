"""Causal flight intercept and successful native racket targets for Approach."""
import torch
from torch.nn import functional as F

from quadsmash.control.full_racket_height_intercept_v4 import predict_height_intercept
from quadsmash.control.initial_intercept import predict_initial_intercept
from .supermimic_racket_dynamics_v1 import racket_state


@torch.no_grad()
def flight_intercept(state, horizon):
    """Recompute the intercept from the current public observation and fixed G07 plan."""
    controller = state.executor_memory.reference_controller
    observation = state.observation.detach().cpu().numpy()
    plane = (controller.parameters[:, 24].detach().cpu().numpy()
             if controller.parameters.shape[1] >= 25 else controller.nominal_target[0])
    inputs = (observation, [1.10, 0., 1.80], [1., 0., 0.], plane, controller.physics)
    if controller.height_intercept is None:
        forecast = predict_initial_intercept(*inputs, dt=.001, max_seconds=horizon)
    else:
        forecast = predict_height_intercept(*inputs, controller.height_intercept, dt=.001, max_seconds=horizon)
    elapsed = state.clock_goal_context['elapsed'].reshape(-1)
    valid = torch.as_tensor(forecast['valid'], device=elapsed.device) & state.clock_goal_context['ball_present'].reshape(-1).bool()
    duration = torch.as_tensor(forecast['arrival_time_s']).to(elapsed)
    position = torch.as_tensor(forecast['position_m']).to(elapsed)
    valid &= torch.isfinite(duration) & (duration >= 0) & (duration <= horizon) & position.isfinite().all(-1)
    return dict(time=torch.where(valid, elapsed + duration, torch.full_like(elapsed, float('inf'))),
                position=torch.where(valid[:, None], position, torch.zeros_like(position)), valid=valid)


def racket_features(physical, geometry):
    body = racket_state(physical, geometry)
    normal = body[:, 3:12].reshape(-1, 3, 3)[:, :, 0]
    return torch.cat((body[:, :3], body[:, 12:15], normal, body[:, 15:18]), dim=-1)


def intercept_loss(before, after, start, end, reference, mask, survival, geometry, *, fixed_weights=None):
    """Sample racket kinematics at the requested sub-tick arrival. Each lane contributes once."""
    arrival = reference['intercept']['time']
    active = (mask & reference['matched'] & reference['intercept']['valid'] &
              (start < arrival) & (arrival <= end))
    weights = active.to(after.dtype) * (survival.detach().reshape(-1) if fixed_weights is None else fixed_weights)
    if not active.any():
        return after[:, :1].sum() * 0, weights
    alpha = ((arrival - start) / (end - start).clamp_min(1e-8)).clamp(0, 1)
    kinematics = torch.lerp(racket_features(before, geometry), racket_features(after, geometry), alpha[:, None])
    terms = intercept_terms(kinematics, reference['intercept']['position'], reference['toi_racket'])
    position, linear, normal, angular = terms.unbind(-1)
    return ((position+linear+normal+angular)*weights).sum(), weights


def intercept_terms(kinematics, intercept_position, native):
    target_position = intercept_position + native[:, :3]
    position = F.smooth_l1_loss((kinematics[:, :3]-target_position)/.05, torch.zeros_like(target_position), reduction='none').mean(-1)
    linear = F.smooth_l1_loss(kinematics[:, 3:6], native[:, 3:6], reduction='none').mean(-1)
    normal = 1-F.cosine_similarity(kinematics[:, 6:9], native[:, 6:9], dim=-1)
    angular = F.smooth_l1_loss(kinematics[:, 9:12]/3, native[:, 9:12]/3, reduction='none').mean(-1)
    return torch.stack((position, linear, normal, angular), dim=-1)
