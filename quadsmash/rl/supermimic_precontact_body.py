"""Event-aligned Body supervision before native racket contact."""

import torch
import torch.nn.functional as F

from .supermimic_racket_dynamics_v1 import racket_at_ball, racket_state, robot_loss

FIELDS = ('state', 'next_state', 'targets', 'previous_action', 'feet', 'present', 'elapsed')
HORIZONS = (20, 32)
PAIR_HORIZONS = (4, 8, 16, *HORIZONS)
PAIR_GROUPS = ("low_speed", "marginal_success", "ordinary_success", "unclassified")
REVISION_POSITION_WEIGHT = 4.


def body_rollout_loss(body, window, launch, position_weight=1.):
    state = window['state'][0]
    horizon = len(window['state'])
    losses = []
    endpoint_metrics = None
    for step in range(horizon):
        # Preserve BPTT through robot kinematics. The incoming shuttle and
        # control history are recorded inputs, independent of the Body fit.
        state = torch.cat((state[:, :49], window['state'][step, :, 49:]), -1)
        state, _, _ = body(
            state, window['targets'][step], window['previous_action'][step], window['feet'][step],
            window['present'][step], window['elapsed'][step], launch, detach_contacts=True)
        if step + 1 not in (4, 8, 16, horizon):
            continue
        truth = window['next_state'][step]
        predicted_pose = racket_state(state, body.geometry)
        target_pose = racket_state(truth, body.geometry)
        # Both twists are evaluated at the same measured shuttle center.
        ball = truth[:, 49:52]
        predicted_racket = racket_at_ball(predicted_pose, ball, full_orientation=True)
        target_racket = racket_at_ball(target_pose, ball, full_orientation=True)
        position = predicted_racket[:, :3] - target_racket[:, :3]
        surface = predicted_racket[:, 3:6] - target_racket[:, 3:6]
        rotation = predicted_racket[:, 6:15] - target_racket[:, 6:15]
        angular = predicted_racket[:, 15:18] - target_racket[:, 15:18]
        terms = (F.smooth_l1_loss(position / .025, torch.zeros_like(position)),
                 F.smooth_l1_loss(surface, torch.zeros_like(surface)),
                 F.smooth_l1_loss(rotation / .12, torch.zeros_like(rotation)),
                 F.smooth_l1_loss(angular / 3., torch.zeros_like(angular)))
        racket_loss = (position_weight * terms[0] + sum(terms[1:])) / (position_weight + 3.)
        losses.append(racket_loss + robot_loss(state, truth)
                      + F.smooth_l1_loss(state[:, :3], truth[:, :3], beta=.025)
                      + F.smooth_l1_loss(state[:, 10:13], truth[:, 10:13], beta=.1))
        endpoint_metrics = (position.norm(dim=-1).mean().detach(), surface.norm(dim=-1).mean().detach())
    # The final state is the contact-adjacent supervision reference.
    return (torch.stack(losses).mean() + losses[-1]) / 2, endpoint_metrics


def make_window(data, members, horizon, device):
    starts, lanes, _ = torch.tensor(members).T
    indices = starts[None] + torch.arange(horizon)[:, None]
    return {k: data[k][indices, lanes].to(device) for k in FIELDS}, lanes


def measured_precontact_body_windows(replay, launch, current_policy, lanes, *, maximum_per_group=4):
    """Choose historical and current native contact prefixes on the requested lanes."""
    from .supermimic_physical_loop_replay_v1 import SOURCE_CODES
    from .supermimic_release_calibration import _native_event_columns

    data, streams, _, sources, policies, ticks = replay._ordered()
    columns = _native_event_columns(replay.schema)
    contact = columns['rally/had_contact']
    hit = data['next_values'][..., contact].squeeze(-1) > .5
    first = hit & (data['values'][..., contact].squeeze(-1) <= .5)
    frame = data['next_values'][..., columns['rally/frame_hit']].squeeze(-1) > .5
    string = data['next_values'][..., columns['rally/string_hit']].squeeze(-1) > .5
    episode = data['episode']
    online = sources == SOURCE_CODES['online']
    eligible = torch.zeros(first.shape[1], dtype=torch.bool)
    eligible[list(lanes)] = True
    events = (first & (frame | string) & online & (streams[:, None] > 1) & eligible).nonzero().tolist()
    if not events:
        return []

    batches = []
    for horizon in HORIZONS:
        pools = {(age, kind): [] for age in ('current','historical') for kind in ('string','frame')}
        for stop,lane in events:
            start = stop - horizon
            if start < 0:
                continue
            prefix = slice(start, stop + 1)
            if (not (episode[prefix,lane] == episode[stop,lane]).all()
                    or not online[prefix,lane].all()
                    or not (policies[prefix,lane] == policies[stop,lane]).all()
                    or not (streams[prefix] == streams[stop]).all()
                    or not (ticks[prefix].diff() == 1).all()
                    or not data['present'][start:stop,lane].all()
                    or hit[start:stop,lane].any()
                    or (data['terminated'][start:stop,lane] | data['timeout'][start:stop,lane]).any()):
                continue
            age = 'current' if int(policies[stop,lane]) == current_policy else 'historical'
            kind = 'string' if bool(string[stop,lane]) else 'frame'
            pools[age,kind].append((start,lane,int(episode[stop,lane])))
        selected, counts = [], {}
        for (age,kind), members in pools.items():
            # Reverse recorded order to keep the most recent available episodes.
            chosen = members[-maximum_per_group:]
            selected.extend(chosen)
            counts[f'{age}_{kind}'] = len(chosen)
        if selected:
            window, lane_ids = make_window(data,selected,horizon,launch.device)
            batches.append((window,launch[lane_ids.to(launch.device)],counts))
    return batches


def paired_precontact_mask(minus, plus, contact):
    """Both branches must be in the same uninterrupted, airborne-ball prefix."""
    masks = []
    for window in (minus, plus):
        before = window['values'][..., contact].squeeze(-1) <= .5
        after = window['next_values'][..., contact].squeeze(-1) <= .5
        masks.append(before & after & window['present'].reshape_as(before).bool())
    return (masks[0] & masks[1]).int().cumprod(dim=0).bool()


def native_pair_groups(minus, plus, columns, *, threshold=4.5, margin=.25):
    """Use only recorded release/landing outcomes; unfinished rows are unclassified."""
    def flags(window, name):
        return window['next_values'][..., columns['rally/' + name]].squeeze(-1) > .5

    classifications = []
    for window in (minus, plus):
        outgoing = columns['rally/outgoing_velocity']
        velocity = window['next_values'][..., outgoing]
        speed = velocity.norm(dim=-1)
        qualified = (flags(window, 'legal_initial_contact') & flags(window, 'airborne_string_hit')
                     & flags(window, 'string_hit') & flags(window, 'released')
                     & window['next_valid'][..., outgoing].bool().all(-1)
                     & torch.isfinite(velocity).all(-1))
        low = (qualified & (speed < threshold)).any(0)
        fault = torch.zeros_like(qualified)
        for path in columns:
            if path.startswith('rally/') and (path.endswith('_fault') or path == 'rally/engineering_body_crossing'):
                fault |= flags(window, path.removeprefix('rally/'))
        landed = (qualified & (velocity[..., 0] > 0) & ~fault
                  & flags(window, 'crossed_net') & flags(window, 'landed_target'))
        marginal = (landed & (speed >= threshold) & (speed < threshold + margin)).any(0)
        ordinary = (landed & (speed >= threshold + margin)).any(0)
        classifications.append((low, marginal, ordinary))
    low, marginal, ordinary = (left | right for left, right in zip(*classifications, strict=True))
    return torch.where(low, 0, torch.where(marginal, 1, torch.where(ordinary, 2, 3)))


def relative_touch_channels(racket, ball, axes, lever_point):
    """Relative twist in one measured basis, at the same point for truth and WM."""
    rotation = racket[:, 3:12].reshape(-1, 3, 3)
    offset = lever_point - racket[:, :3]
    surface = racket[:, 12:15] + torch.linalg.cross(racket[:, 15:18], offset)
    relative = (axes.transpose(-1, -2) @ (surface - ball[:, 3:])[:, :, None]).squeeze(-1)
    gap = ((ball[:, :3] - racket[:, :3]) * rotation[:, :, 0]).sum(-1, keepdim=True)
    return relative, gap


def racket_velocity_residual(before, after, dt):
    """Compare interval displacement with trapezoidal instantaneous racket speed."""
    displacement = (after[:, :3] - before[:, :3]) / dt[:, None]
    return displacement - .5 * (before[:, 12:15] + after[:, 12:15])


def body_pair_loss(world, minus, plus, launch, columns):
    """Match signed motion at identical control boundaries without fitting Contact WM."""
    contact = columns['rally/had_contact']
    masks = paired_precontact_mask(minus, plus, contact)
    groups = native_pair_groups(minus, plus, columns)
    available = [h for h in PAIR_HORIZONS if h <= len(masks) and masks[h - 1].any()]
    if not available:
        return None, {}, {}
    states = [w['state'][0] for w in (minus, plus)]
    native_hit_tick = []
    for window in (minus, plus):
        hits = window['next_values'][..., contact].squeeze(-1) > .5
        native_hit_tick.append(torch.where(hits.any(0), hits.int().argmax(0) + 1, 0))
    scale = launch.new_tensor([.03] * 3 + [.12] * 9 + [1.] * 3 + [3.] * 3)
    losses, support, diagnostics = [], {}, {}
    for tick in range(available[-1]):
        h = tick + 1
        score = h in available
        reference = []
        for branch, window in enumerate((minus, plus)):
            measured = window['state'][tick]
            state = torch.cat((states[branch][:, :49], measured[:, 49:]), -1)
            states[branch], _, _ = world(state, window['targets'][tick], window['previous_action'][tick],
                                         window['feet'][tick], window['present'][tick],
                                         window['elapsed'][tick], launch, detach_contacts=True)
            if score:
                valid = masks[tick]
                truth = window['next_state'][tick, valid]
                predicted = states[branch][valid]
                poses = racket_state(predicted, world.geometry), racket_state(truth, world.geometry)
                with torch.no_grad():
                    start = racket_state(state[valid], world.geometry), racket_state(measured[valid], world.geometry)
                    dt = truth.new_full((len(truth),), world.dt)
                    residual = [racket_velocity_residual(a, b, dt) for a, b in zip(start, poses, strict=True)]
                reference.append((poses, truth[:, 49:], robot_loss(predicted, truth, per_row=True), residual))
        if not score:
            continue
        masks_h = groups[valid]
        hit_a, hit_b = (ticks[valid] for ticks in native_hit_tick)
        pose_a, pose_b = reference[0][0], reference[1][0]
        axes = pose_a[1][:, 3:12].reshape(-1, 3, 3)
        touch = []
        for poses, ball, _, _ in reference:
            # Near contact use the observed shuttle point. Far away use the
            # measured center, avoiding an extrapolated long lever arm.
            center = poses[1][:, :3]
            near = (ball[:, :3] - center).norm(dim=-1, keepdim=True) <= .4
            point = torch.where(near, ball[:, :3], center)
            touch.append([relative_touch_channels(pose, ball, axes, point) for pose in poses])
        a, native_a = pose_a
        b, native_b = pose_b
        def huber(predicted, measured):
            return F.smooth_l1_loss(predicted, measured, reduction='none').mean(-1)
        relative_delta = touch[1][0][0] - touch[0][0][0]
        native_delta = touch[1][1][0] - touch[0][1][0]
        gap_delta = touch[1][0][1] - touch[0][0][1]
        native_gap_delta = touch[1][1][1] - touch[0][1][1]
        response = (huber((b-a)/scale, (native_b-native_a)/scale)
                    + huber(relative_delta[:, :1], native_delta[:, :1])
                    + huber(relative_delta[:, 1:], native_delta[:, 1:])
                    + huber(gap_delta / .03, native_gap_delta / .03))
        absolute = .5 * (huber(a/scale, native_a/scale) + huber(b/scale, native_b/scale))
        absolute = absolute + .5 * (reference[0][2] + reference[1][2])
        rows = {}
        group_losses = []
        for code, name in enumerate(PAIR_GROUPS):
            group = masks_h == code
            if not group.any():
                continue
            group_losses.append((response[group] + absolute[group]).mean())
            with torch.no_grad():
                error = (relative_delta - native_delta)[group]
                native_normal_speed = native_delta[group, 0]
                predicted_normal_speed = relative_delta[group, 0]
                # Keep unresolved very small directions in the MAE. Count
                # signed agreement only where the reference is measurable.
                measurable = native_normal_speed.abs() >= .01
                hit_pairs = group & (hit_a > 0) & (hit_b > 0)
                normal = (b[:, 3:12] - a[:, 3:12]).reshape(-1, 3, 3)[:, :, 0]
                native_normal = (native_b[:, 3:12] - native_a[:, 3:12]).reshape(-1, 3, 3)[:, :, 0]
                residuals = [branch[3] for branch in reference]
                rows[name] = dict(
                    pairs=int(group.sum()), normal_response_mae_mps=float(error[:, 0].abs().mean()),
                    normal_response_measurable=int(measurable.sum()),
                    normal_response_sign_correct=int((measurable & (native_normal_speed * predicted_normal_speed > 0)).sum()),
                    native_paired_contact_ticks=int(hit_pairs.sum()),
                    native_contact_tick_delta_mean=float((hit_b-hit_a)[hit_pairs].float().mean()) if hit_pairs.any() else None,
                    tangent_response_mae_mps=float(error[:, 1:].norm(dim=-1).mean()),
                    face_normal_response_mae=float((normal-native_normal)[group].norm(dim=-1).mean()),
                    signed_plane_gap_response_mae_m=float((gap_delta-native_gap_delta)[group].abs().mean()),
                    predicted_discrete_kinematic_residual_mps=float(torch.cat(
                        [pred[group].norm(dim=-1) for pred, _ in residuals]).mean()),
                    native_discrete_kinematic_residual_mps=float(torch.cat(
                        [native[group].norm(dim=-1) for _, native in residuals]).mean()),
                    excess_discrete_kinematic_residual_mps=float(torch.cat(
                        [(pred-native)[group].norm(dim=-1) for pred, native in residuals]).mean()))
        support[h] = int(valid.sum())
        diagnostics[h] = rows
        losses.append(torch.stack(group_losses).mean())
    return torch.stack(losses).mean(), support, diagnostics
