"""Causal flight intercept and successful native racket targets for Approach."""
import math
import statistics

import torch
from torch.nn import functional as F

from quadsmash.control.full_racket_height_intercept_v4 import predict_height_intercept
from quadsmash.control.initial_intercept import predict_initial_intercept
from .supermimic_racket_dynamics_v1 import racket_state


def calibrated_plane_offset(calibration, held, checkpoint_sha256):
    if (calibration['schema'] != 'training-native-contact-target-plane-v1'
            or calibration['checkpoint_sha256'] != checkpoint_sha256):
        raise ValueError('Arrival-plane calibration has a different checkpoint identity')
    training = calibration['training']
    if (not training or len({p['seed'] for p in training}) != len(training)
            or any(set(p['lanes']) & set(held) for p in training)):
        raise ValueError('Invalid training lanes or repeated prefixes for arrival-plane fit')
    values = []
    for panel in training:
        ids = [r['lane'] for r in panel['cases']]
        if len(ids) != len(set(ids)) or not set(ids) <= set(panel['lanes']):
            raise ValueError('Invalid arrival-plane case identity')
        values.extend(r['native_minus_plane_m'] for r in panel['cases'] if r['qualified'] and r['valid'])
    if not values or not all(map(math.isfinite, values)) or len(values) != calibration['fit_samples']:
        raise ValueError('Invalid native contact observations for plane offset')
    offset = statistics.median(values)
    if offset != calibration['plane_offset_m']:
        raise ValueError('Stored offset differs from the fitted native contact locations')
    return offset


@torch.no_grad()
def flight_intercept(state, horizon, *, plane_offset_m=0.):
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
    if plane_offset_m:
        # Forecast the earlier incoming ball center on the same public free-flight path.
        # The execution controller keeps its original plan.
        adjusted_plane = forecast['position_m'][:, 0] + plane_offset_m
        shifted = predict_initial_intercept(observation, inputs[1], inputs[2], adjusted_plane,
                                            controller.physics, dt=.001, max_seconds=horizon)
        shifted['valid'] &= forecast['valid']
        forecast = shifted
    elapsed = state.clock_goal_context['elapsed'].reshape(-1)
    valid = torch.as_tensor(forecast['valid'], device=elapsed.device) & state.clock_goal_context['ball_present'].reshape(-1).bool()
    duration = torch.as_tensor(forecast['arrival_time_s']).to(elapsed)
    position = torch.as_tensor(forecast['position_m']).to(elapsed)
    valid &= torch.isfinite(duration) & (duration >= 0) & (duration <= horizon) & position.isfinite().all(-1)
    return dict(time=torch.where(valid, elapsed + duration, torch.full_like(elapsed, float('inf'))),
                position=torch.where(valid[:, None], position, torch.zeros_like(position)), valid=valid)


def bind_intercept_reference(state, bank, geometry, horizon, *, plane_offset_m=0.):
    forecast = flight_intercept(state, horizon, plane_offset_m=plane_offset_m)
    elapsed = state.clock_goal_context['elapsed'].reshape(-1)
    reference = bank.bind(state.physical, 'approach', geometry=geometry, elapsed=elapsed,
                          event_time=forecast['time'], require_toi=True)
    if reference is not None:
        reference['intercept'] = forecast
        reference['matched'] = reference['matched'] & forecast['valid'] & (forecast['time'] > elapsed)
    return reference


def racket_features(physical, geometry):
    return _features_from_racket(racket_state(physical, geometry))


def _features_from_racket(body):
    normal = body[:, 3:12].reshape(-1, 3, 3)[:, :, 0]
    return torch.cat((body[:, :3], body[:, 12:15], normal, body[:, 15:18]), dim=-1)


def precontact_mask(state, selected, gate, episode_s):
    active = state.clock_goal_context['elapsed'].reshape(-1) < episode_s - 1e-7
    contact = state.task_memory.get('rally', {}).get('had_contact')
    if contact is None:
        return torch.zeros_like(selected)
    return selected & active & (gate.detach().reshape(-1) > .25) & (contact.reshape(-1) < .5)


def intercept_interval(start, end, reference, mask):
    arrival = reference['intercept']['time']
    return (mask & reference['matched'] & reference['intercept']['valid'] &
            (start < arrival) & (arrival <= end))


@torch.no_grad()
def preview_intercept_lanes(initial, reference, allowed, trainer):
    """Check the same frozen model, causal latch and interval used by the TOI loss."""
    state = initial.branch()
    selected = allowed & reference['matched']
    covered = torch.zeros_like(selected)
    survival = state.physical.new_ones(len(selected))
    for _ in range(trainer.timebase.imagination_steps):
        start = state.clock_goal_context['elapsed'].reshape(-1)
        pending = selected & (start < reference['intercept']['time']) & (survival > 0)
        pending &= start < trainer.timebase.episode_s - 1e-7
        if not pending.any():
            break
        mask = precontact_mask(state, pending, trainer._head_gate(state), trainer.timebase.episode_s)
        action = trainer._actor_action(state, sample=False)[0]
        state, _, done, _ = trainer.adapter(state, action, detach_ball=True)
        end = state.clock_goal_context['elapsed'].reshape(-1)
        covered |= intercept_interval(start, end, reference, mask)
        survival = survival * (1 - done.reshape(-1))
    return covered


@torch.no_grad()
def compact_intercept_preview(row, allowed, launch, template, trainer, device):
    """Roll only admissible, already-launched lanes and return original lane indices."""
    from .supermimic_compact_starts import compact_start
    from .supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1

    present = row['before']['physical']['ball_present'].reshape(-1).to(allowed)
    selected = allowed & present
    result = torch.zeros_like(allowed)
    if not selected.any():
        return result
    small, small_launch = compact_start(row, selected, launch)
    state = PhysicalImaginationStateV1.from_record(small, executor_template=template, launch=small_launch, device=device)
    cfg = trainer.timebase
    reference = bind_intercept_reference(state, trainer.reference_bank, trainer.adapter.physical_model.geometry,
                                         cfg.control_dt_s * cfg.imagination_steps, plane_offset_m=trainer.toi_plane_offset_m)
    if reference is not None:
        result[selected] = preview_intercept_lanes(state, reference, torch.ones_like(reference['matched']), trainer).to(result)
    return result


def string_boundary(config_path, like):
    from quadsmash.physics.full_racket_v4 import load_racket, string_polygon
    vertices = like.new_tensor(string_polygon(load_racket(config_path)))
    edge = vertices.roll(-1, 0) - vertices
    inward = F.normalize(torch.stack((-edge[:, 1], edge[:, 0]), dim=-1), dim=-1)
    return inward, (vertices * inward).sum(-1)


def string_margin_cost(local_yz, boundary):
    inward, offset = boundary
    margin = (local_yz @ inward.T - offset).amin(-1)
    # Same 5 cm position scale as the TOI loss. No penalty beyond 5 mm inside.
    return F.smooth_l1_loss((.005 - margin).clamp_min(0) / .05,
                            torch.zeros_like(margin), reduction='none')


def intercept_loss(before, after, start, end, reference, mask, survival, geometry, *, fixed_weights=None, detach_empty=False, boundary=None):
    """Sample racket kinematics at the requested sub-tick arrival. Each lane contributes once."""
    arrival = reference['intercept']['time']
    active = intercept_interval(start, end, reference, mask)
    weights = active.to(after.dtype) * (survival.detach().reshape(-1) if fixed_weights is None else fixed_weights)
    if not active.any():
        # An empty interval contributes exactly zero at every physical state.
        endpoint = after.detach() if detach_empty else after
        return endpoint[:, :1].sum() * 0, weights
    alpha = ((arrival - start) / (end - start).clamp_min(1e-8)).clamp(0, 1)
    if boundary is None:
        kinematics = torch.lerp(racket_features(before, geometry), racket_features(after, geometry), alpha[:, None])
        edge_cost = 0.
    else:
        a, b = racket_state(before, geometry), racket_state(after, geometry)
        kinematics = torch.lerp(_features_from_racket(a), _features_from_racket(b), alpha[:, None])
        point = reference['intercept']['position'].detach()
        def local_yz(body):
            axes = body[:, 3:12].reshape(-1, 3, 3)[:, :, 1:]
            return ((point - body[:, :3])[:, :, None] * axes).sum(1)
        edge_cost = string_margin_cost(torch.lerp(local_yz(a), local_yz(b), alpha[:, None]), boundary)
    terms = intercept_terms(kinematics, reference['intercept']['position'], reference['toi_racket'])
    position, linear, normal, angular = terms.unbind(-1)
    return ((position+linear+normal+angular+edge_cost)*weights).sum(), weights


def intercept_terms(kinematics, intercept_position, native):
    target_position = intercept_position + native[:, :3]
    position = F.smooth_l1_loss((kinematics[:, :3]-target_position)/.05, torch.zeros_like(target_position), reduction='none').mean(-1)
    linear = F.smooth_l1_loss(kinematics[:, 3:6], native[:, 3:6], reduction='none').mean(-1)
    normal = 1-F.cosine_similarity(kinematics[:, 6:9], native[:, 6:9], dim=-1)
    angular = F.smooth_l1_loss(kinematics[:, 9:12]/3, native[:, 9:12]/3, reduction='none').mean(-1)
    return torch.stack((position, linear, normal, angular), dim=-1)
