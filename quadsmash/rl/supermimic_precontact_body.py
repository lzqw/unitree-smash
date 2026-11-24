"""Event-aligned Body supervision before native racket contact."""

import torch
import torch.nn.functional as F

from .supermimic_racket_dynamics_v1 import racket_at_ball, racket_state, robot_loss

FIELDS = ('state', 'next_state', 'targets', 'previous_action', 'feet', 'present', 'elapsed')
HORIZONS = (20, 32)
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
