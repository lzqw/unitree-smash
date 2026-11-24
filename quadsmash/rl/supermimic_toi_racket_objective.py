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
    incoming = torch.as_tensor(forecast['incoming_velocity_mps']).to(elapsed)
    valid &= torch.isfinite(duration) & (duration >= 0) & (duration <= horizon) & position.isfinite().all(-1)
    return dict(time=torch.where(valid, elapsed + duration, torch.full_like(elapsed, float('inf'))),
                position=torch.where(valid[:, None], position, torch.zeros_like(position)),
                incoming_velocity=torch.where(valid[:, None], incoming, torch.zeros_like(incoming)), valid=valid)


@torch.no_grad()
def search_outgoing_target(reference, prior, court, *, minimum_speed, interior_m=.03):
    """Choose a court-feasible outgoing shot and invert the normal impulse."""
    from .supermimic_outgoing_flight import predict_flight_events

    forecast = reference['intercept']
    point, incoming = forecast['position'], forecast['incoming_velocity']
    native = reference['toi_racket']
    normal_target = torch.zeros_like(point)
    surface_target = torch.zeros_like(point[:, 0])
    valid = torch.zeros_like(forecast['valid'])
    lanes = (reference['matched'] & forecast['valid'] & incoming.isfinite().all(-1)).nonzero().flatten()
    if not len(lanes):
        return dict(valid=valid, normal=normal_target, surface_speed=surface_target)
    p, v = point[lanes], incoming[lanes]
    center_x = sum(court['target_x_limits_m']) / 2
    yaw = torch.atan2(-p[:, 1], center_x - p[:, 0])
    levels = p.new_tensor([.15, .6, 1.2, 2., 3., 4.])
    yaw_offsets = p.new_tensor([-.12, 0., .12])
    pitch = p.new_tensor([-.20, -.06, .08, .20, .34])
    grid = torch.cartesian_prod(levels, yaw_offsets, pitch)
    azimuth = yaw[:, None] + grid[:, 1]
    elevation = grid[:, 2].expand_as(azimuth)
    direction = torch.stack((azimuth.cos() * elevation.cos(), azimuth.sin() * elevation.cos(), elevation.sin()), -1)
    outgoing = direction * (minimum_speed + grid[:, 0])[None, :, None]
    count = len(grid)
    launch = p[:, None].expand_as(outgoing)
    results = predict_flight_events(launch.flatten(0, 1), outgoing.flatten(0, 1), prior, court, dt=.02)
    landing = results['landing_position'].reshape(len(lanes), count, 3)
    low, high = court['target_x_limits_m']
    feasible = (results['crossed_net'].reshape(-1, count) & results['landed'].reshape(-1, count)
                & (results['net_clearance'].reshape(-1, count) > .02)
                & (results['net_lateral_position'].reshape(-1, count).abs() < court['net_width_m']/2)
                & (landing[..., 0] >= low + interior_m) & (landing[..., 0] <= high - interior_m)
                & (landing[..., 1].abs() <= court['target_y_half_width_m'] - interior_m))
    change = outgoing - v[:, None]
    normal = F.normalize(change, dim=-1)
    face = native[lanes, 6:9]
    face_alignment = (normal * face[:, None]).sum(-1)
    required = (normal * v[:, None]).sum(-1) + change.norm(dim=-1) / (
        prior.normal_impulse_scale * (1. + prior.restitution))
    historic_speed = (normal * native[lanes, None, 3:6]).sum(-1)
    feasible &= (face_alignment.abs() >= .5) & ((required - historic_speed).abs() <= 3.)
    score = ((landing[..., 0] - center_x).square() + landing[..., 1].square()
             + .25 * (required - historic_speed).square() + 1 - face_alignment.abs())
    choice = score.masked_fill(~feasible, torch.inf).argmin(-1)
    row = torch.arange(len(lanes), device=point.device)
    valid[lanes] = feasible[row, choice]
    normal_target[lanes] = normal[row, choice]
    surface_target[lanes] = required[row, choice]
    return dict(valid=valid, normal=normal_target, surface_speed=surface_target)


def bind_intercept_reference(state, bank, geometry, horizon, *, plane_offset_m=0.):
    forecast = flight_intercept(state, horizon, plane_offset_m=plane_offset_m)
    elapsed = state.clock_goal_context['elapsed'].reshape(-1)
    reference = bank.bind(state.physical, 'approach', geometry=geometry, elapsed=elapsed,
                          event_time=forecast['time'], require_toi=True)
    if reference is not None:
        reference['intercept'] = forecast
        reference['matched'] = reference['matched'] & forecast['valid'] & (forecast['time'] > elapsed)
    return reference


def approach_horizon(reference, selected, elapsed, *, maximum, dt, followthrough_ticks):
    """Pick the shortest bucket covering causal arrival, separation and followthrough."""
    if reference is None:
        return maximum
    forecast = reference['intercept']
    if not selected.any() or not forecast['valid'][selected].all():
        return maximum
    duration = (forecast['time'][selected] - elapsed.reshape(-1)[selected]).max()
    required = math.ceil(float(duration) / dt) + 2 + followthrough_ticks
    return next((bucket for bucket in (16, 32, 64) if required <= bucket <= maximum), maximum)


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
    swing_started = torch.zeros_like(selected)
    for _ in range(trainer.timebase.imagination_steps):
        start = state.clock_goal_context['elapsed'].reshape(-1)
        pending = selected & (start < reference['intercept']['time']) & (survival > 0)
        pending &= start < trainer.timebase.episode_s - 1e-7
        if not pending.any():
            break
        mask = precontact_mask(state, pending, trainer._head_gate(state), trainer.timebase.episode_s)
        if getattr(trainer, 'landing_and_first_contact_objectives', False):
            swing_started |= mask
            mask = swing_started & pending
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
    state = PhysicalImaginationStateV1.from_record(small, executor_template=template, launch=small_launch,
                                                   device=device, array_plan_observations=True)
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


def string_margin_cost(local_yz, boundary, *, minimum=.005):
    inward, offset = boundary
    margin = (local_yz @ inward.T - offset).amin(-1)
    # Same 5 cm position scale as the TOI loss.
    return F.smooth_l1_loss((minimum - margin).clamp_min(0) / .05,
                            torch.zeros_like(margin), reduction='none')


def first_racket_cost(before, after, impact, geometry, preview, strips, boundary, dt, *, minimum=.005):
    """First native-shape preview, with a body-only pose derivative at frozen TOI."""
    from .supermimic_ball_geometry_preview_v1 import box_gap, round_gap
    from .supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
    a, b = racket_state(before, geometry), racket_state(after, geometry)
    fraction = (impact['time'].detach() / dt).clamp(0, 1)[:, None]
    rotation = BallImpulsePriorV1._interpolate_rotation(a[:, 3:12].reshape(-1, 3, 3),
                                                       b[:, 3:12].reshape(-1, 3, 3), fraction)
    offset = impact['pre_position'].detach() - torch.lerp(a[:, :3], b[:, :3], fraction)
    local_pose = (rotation.transpose(1, 2) @ offset.unsqueeze(-1)).squeeze(-1)
    local = impact['local'].detach() + (local_pose - local_pose.detach())
    boxes = box_gap(local, preview.racket_box_centers, preview.racket_box_halves)
    strings = boxes[:, :strips].amin(-1)
    frame = round_gap(local, preview.racket_round_centers, preview.racket_round_axes,
                      preview.racket_round_halves, preview.racket_round_radii, preview.racket_round_capsule).amin(-1)
    if boxes.shape[1] > strips:
        frame = torch.minimum(frame, boxes[:, strips:].amin(-1))
    competition = torch.nn.functional.softplus((strings - frame + .005) / .005)
    error = competition + string_margin_cost(local[:, 1:], boundary, minimum=minimum)
    return 1 - torch.exp(-error)


def intercept_loss(before, after, start, end, reference, mask, survival, geometry, *,
                   fixed_weights=None, detach_empty=False, boundary=None, relative_face_speed=False,
                   minimum_margin=.005, prefer_measured_contact=False):
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
        edge_cost = string_margin_cost(torch.lerp(local_yz(a), local_yz(b), alpha[:, None]),
                                      boundary, minimum=minimum_margin)
    terms = intercept_terms(kinematics, reference['intercept']['position'], reference['toi_racket'])
    position, linear, normal, angular = terms.unbind(-1)
    target = reference.get('ball_only_contact')
    if target is not None:
        axis = target['normal']
        point_arm = reference['intercept']['position'] - kinematics[:, :3]
        surface = kinematics[:, 3:6] + torch.linalg.cross(kinematics[:, 9:12], point_arm)
        if relative_face_speed:
            incoming = reference['intercept']['incoming_velocity']
            face = F.normalize(kinematics[:, 6:9], dim=-1)
            face = torch.where((face * axis).sum(-1, keepdim=True) >= 0., face, -face)
            along = ((surface - incoming) * face).sum(-1)
            desired = target['surface_speed'] - (incoming * axis).sum(-1)
        else:
            along = (surface * axis).sum(-1)
            desired = target['surface_speed']
        speed = F.smooth_l1_loss(along, desired, reduction='none')
        facing = 1 - F.cosine_similarity(kinematics[:, 6:9], axis, dim=-1).abs()
        linear = torch.where(target['valid'], speed, linear)
        normal = torch.where(target['valid'], .5 * (normal + facing), normal)
    measured = reference.get('measured_face_target') if relative_face_speed else None
    if measured is not None:
        native = reference['toi_racket']
        incoming = reference['intercept']['incoming_velocity']
        compatible = ((incoming - measured['incoming']).norm(dim=-1) <= 1.)
        valid = torch.isfinite(measured['speed']) & compatible
        offset = reference['intercept']['position'] + measured['touch_offset'] - kinematics[:, :3]
        surface = kinematics[:, 3:6] + torch.linalg.cross(kinematics[:, 9:12], offset)
        face = F.normalize(kinematics[:, 6:9], dim=-1)
        face = torch.where((face * measured['normal']).sum(-1, keepdim=True) >= 0., face, -face)
        if prefer_measured_contact:
            native_offset = measured['touch_offset'] - native[:, :3]
            rotation = torch.linalg.cross(measured['normal'], face)
            tangent = torch.linalg.cross(rotation, native_offset)
            aligned_offset = (native_offset + tangent + torch.linalg.cross(rotation, tangent)
                              / (1. + (face * measured['normal']).sum(-1, keepdim=True)).clamp_min(1.e-6))
            point_error = (aligned_offset - offset) / .025
            point_loss = F.smooth_l1_loss(point_error, torch.zeros_like(point_error), reduction='none').mean(-1)
            position = torch.where(valid, .5 * (position + point_loss), position)
            surface = kinematics[:, 3:6] + torch.linalg.cross(kinematics[:, 9:12], aligned_offset)
            native_surface = native[:, 3:6] + torch.linalg.cross(native[:, 9:12], native_offset)
            surface_loss = F.smooth_l1_loss(surface, native_surface, reduction='none').mean(-1)
            linear = torch.where(valid, surface_loss, linear)
        along = ((surface - incoming) * face).sum(-1)
        desired = torch.where(valid, measured['speed'], along.detach())
        error = ((along - desired).abs() - .25).relu()
        speed = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
        linear = linear + torch.where(valid, speed, 0.)
    return ((position+linear+normal+angular+edge_cost)*weights).sum(), weights


def intercept_terms(kinematics, intercept_position, native):
    target_position = intercept_position + native[:, :3]
    position = F.smooth_l1_loss((kinematics[:, :3]-target_position)/.05, torch.zeros_like(target_position), reduction='none').mean(-1)
    linear = F.smooth_l1_loss(kinematics[:, 3:6], native[:, 3:6], reduction='none').mean(-1)
    normal = 1-F.cosine_similarity(kinematics[:, 6:9], native[:, 6:9], dim=-1)
    angular = F.smooth_l1_loss(kinematics[:, 9:12]/3, native[:, 9:12]/3, reduction='none').mean(-1)
    return torch.stack((position, linear, normal, angular), dim=-1)
