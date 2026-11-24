"""Native release-speed calibration for one frozen Body and contact model."""

import torch

from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_dataset_v1 import decode_physical

LEADS = (0, 1, 4, 8, 16, 24, 40)


def racket_release_boundary(ball, pose, racket_id, preview, *, separation_m):
    """A predicted control-boundary separation witness in the current racket frame."""
    offset = ball[:, :3] - pose['position'][:, racket_id]
    rotation = pose['rotation'][:, racket_id].transpose(-1, -2)
    local = (rotation @ offset[..., None]).squeeze(-1)
    rotational_speed = torch.linalg.cross(pose['angular_velocity'][:, racket_id], offset)
    relative = ball[:, 3:] - pose['velocity'][:, racket_id] - rotational_speed
    # A gap alone is insufficient: it may occur during re-approach after a bounce.
    next_point = local + .001 * (rotation @ relative[..., None]).squeeze(-1)
    gap = preview.racket_gap(local)
    return (gap > separation_m) & (preview.racket_gap(next_point) > gap)


def speed_bias_at_lead(lead_ticks, calibration):
    if not calibration:
        return torch.zeros_like(lead_ticks, dtype=torch.float32)
    rows = [row for row in calibration['by_lead'] if row['bias_mps'] is not None]
    if not rows:
        return torch.zeros_like(lead_ticks, dtype=torch.float32)
    leads = lead_ticks.new_tensor([row['lead_ticks'] for row in rows])
    values = torch.tensor([row['bias_mps'] for row in rows], device=lead_ticks.device)
    picks = (lead_ticks[:, None] - leads).abs().argmin(dim=-1)
    return values[picks]


def release_error_summary(predicted, native, *, threshold=4.5, fitted_bias=None):
    if not len(predicted):
        return dict(matched=0, low_speed=0, false_pass=0, corrected_false_pass=0,
                    native_pass=0, false_fail=0, corrected_false_fail=0, bias_mps=None,
                    signed_mean_mps=None, corrected_signed_mean_mps=None,
                    mae_mps=None, corrected_mae_mps=None)
    error = predicted - native
    # The median is a model-bias estimate. It does not include a task margin.
    bias = float(error.quantile(.5)) if fitted_bias is None else fitted_bias
    slow = native < threshold
    pass_before = predicted >= threshold
    pass_after = predicted - bias >= threshold
    corrected_error = error - bias
    return dict(matched=len(predicted), low_speed=int(slow.sum()),
                native_pass=int((~slow).sum()), false_pass=int((slow & pass_before).sum()),
                corrected_false_pass=int((slow & pass_after).sum()),
                false_fail=int((~slow & ~pass_before).sum()),
                corrected_false_fail=int((~slow & ~pass_after).sum()),
                bias_mps=bias, signed_mean_mps=float(error.mean()),
                corrected_signed_mean_mps=float(corrected_error.mean()),
                mae_mps=float(error.abs().mean()), corrected_mae_mps=float(corrected_error.abs().mean()))


def _native_event_columns(schema):
    columns = {}
    offset = 0
    for field in schema.fields:
        columns[field.path] = slice(offset, offset + field.width)
        offset += field.width
    return columns


def _release_windows(replay, excluded_lanes, max_cases, *, qualified_only=True):
    data, streams, _, sources, _, ticks = replay._ordered()
    columns = _native_event_columns(replay.schema)

    def bit(source, field):
        return data[source][..., columns['rally/' + field]].squeeze(-1) > .5

    released = ~bit('values', 'released') & bit('next_values', 'released')
    contacted = ~bit('values', 'had_contact') & bit('next_values', 'had_contact')
    outgoing = columns['rally/outgoing_velocity']
    release_valid = data['next_valid'][..., outgoing].bool().all(-1)
    qualified = (bit('next_values', 'legal_initial_contact') & bit('next_values', 'string_hit')
                 & bit('next_values', 'airborne_string_hit'))
    candidates = released & release_valid & (sources >= 0) & (streams[:, None] > 0)
    if qualified_only:
        candidates &= qualified
    candidates[:, list(excluded_lanes)] = False
    cases = []
    seen = set()
    for release_tick, lane in reversed(candidates.nonzero().tolist()):
        stream = int(streams[release_tick])
        episode = int(data['episode'][release_tick, lane])
        identity = (stream, lane, episode)
        if identity in seen:
            continue
        lookback = max(0, release_tick - 16)
        previous = contacted[lookback:release_tick + 1, lane].nonzero().flatten()
        if not len(previous):
            continue
        impact_tick = lookback + int(previous[-1])
        if (streams[impact_tick] != stream or data['episode'][impact_tick, lane] != episode
                or not torch.isfinite(data['next_values'][release_tick, lane, outgoing]).all()):
            continue
        cases.append((impact_tick, release_tick, lane))
        seen.add(identity)
        if len(cases) == max_cases:
            break
    return data, streams, ticks, outgoing, cases


def measured_contact_batches(replay, launch, *, excluded_lanes, updates, batch_size=16):
    """Return bounded same-episode full-contact windows, including failed releases."""
    data, streams, ticks, _, cases = _release_windows(
        replay, excluded_lanes, max_cases=256, qualified_only=False)
    usable = [(start, end, lane) for start, end, lane in cases
              if (streams[start:end + 1] == streams[start]).all()
              and (ticks[start:end + 1].diff() == 1).all()
              and (data['episode'][start:end + 1, lane] == data['episode'][start, lane]).all()]

    def batches():
        device = launch.device
        for _ in range(updates if usable else 0):
            picks = torch.randperm(len(usable), generator=replay.generator)[:batch_size].tolist()
            starts, stops, lanes = torch.tensor([usable[i] for i in picks]).T
            horizon = int((stops - starts).max()) + 1
            indices = (starts[None] + torch.arange(horizon)[:, None]).minimum(stops[None])
            fields = ('state', 'next_state', 'values', 'next_values', 'next_valid',
                      'targets', 'previous_action', 'feet', 'present', 'elapsed')
            yield {key: data[key][indices, lanes].to(device) for key in fields}, launch[lanes.to(device)]

    return len(usable), batches()


def measured_contact_pairs(replay, launch, *, offset, limit):
    """Select signed intervention windows with observed contact/release on both branches."""
    if limit < 1:
        return []
    columns = _native_event_columns(replay.schema)
    fields = ('state', 'next_state', 'values', 'next_values', 'next_valid',
              'targets', 'previous_action', 'feet', 'present', 'elapsed')
    device = launch.device

    def observed(window, name):
        index = columns['rally/' + name]
        return ((window['values'][..., index] <= .5)
                & (window['next_values'][..., index] > .5)).squeeze(-1).any(0)

    pairs = []
    records = replay.training_probe_pairs
    pivot = offset % max(1, len(records))
    for lanes, minus, plus in records[pivot:] + records[:pivot]:
        mask = observed(minus, 'had_contact') & observed(plus, 'had_contact')
        mask &= observed(minus, 'released') & observed(plus, 'released')
        if not mask.any():
            continue
        sample = tuple({key: window[key][:, mask].to(device) for key in fields} for window in (minus, plus))
        pairs.append((*sample, launch[torch.tensor(lanes, device=device)[mask.to(device)]]))
        if len(pairs) == limit:
            break
    return pairs


def measured_body_pairs(replay, launch, *, offset, limit=4):
    """Return authenticated same-clock interventions for every supported pre-impact horizon."""
    from .supermimic_precontact_body import PAIR_HORIZONS, paired_precontact_mask

    columns = _native_event_columns(replay.schema)
    contact = columns['rally/had_contact']
    fields = ('state', 'next_state', 'values', 'next_values', 'next_valid',
              'targets', 'previous_action', 'feet', 'present', 'elapsed')
    records = replay.training_probe_pairs
    if not records:
        return []
    pairs = []
    for index in range(len(records)):
        lanes, minus, plus = records[(index + offset) % len(records)]
        if len(minus['state']) != len(plus['state']) or not torch.equal(minus['elapsed'], plus['elapsed']):
            raise ValueError('Native Body pair branches must use identical control boundaries')
        available = min(len(minus['state']), PAIR_HORIZONS[-1])
        if available < PAIR_HORIZONS[0]:
            continue
        valid = paired_precontact_mask(minus, plus, contact)[PAIR_HORIZONS[0] - 1]
        if not valid.any():
            continue
        selected = torch.tensor(lanes)[valid]
        windows = [{name: branch[name][:available, valid].to(launch.device)
                    for name in (*fields, 'terminal_response_group') if name in branch}
                   for branch in (minus, plus)]
        pairs.append((*windows, launch[selected.to(launch.device)]))
        if len(pairs) == limit:
            break
    return pairs


@torch.no_grad()
def calibrate_release_speed(adapter, replay, launch, *, excluded_lanes, string_strips,
                            separation_m, threshold=4.5, max_cases=64, training_calibration=None,
                            flight_court=None):
    """Fit on native training lanes, using recorded commands only for WM replay."""
    data, streams, ticks, outgoing, cases = _release_windows(replay, excluded_lanes, max_cases)
    device = launch.device
    by_lead = []
    chain = None
    for lead in LEADS:
        applied_bias = None
        if training_calibration is not None:
            applied_bias = float(speed_bias_at_lead(torch.tensor([lead]), training_calibration)[0])
        # Keep a few control boundaries after measured release to detect timing error.
        valid = []
        for impact, release, lane in cases:
            first, last = impact - lead, min(len(ticks) - 1, release + 4)
            if first < 0 or not (streams[first:last + 1] == streams[impact]).all():
                continue
            if not (ticks[first:last + 1].diff() == 1).all():
                continue
            if not (data['episode'][first:last + 1, lane] == data['episode'][impact, lane]).all():
                # A completed lane may reset right after release. Keep the
                # observed release itself and stop there for that sample.
                last = release
                if not (data['episode'][first:last + 1, lane] == data['episode'][impact, lane]).all():
                    continue
            valid.append((first, last, release, lane))
        if not valid:
            by_lead.append(dict(lead_ticks=lead, native_cases=0, **release_error_summary([], [])))
            continue
        starts, stops, releases, lanes = torch.tensor(valid).T
        native = data['next_values'][releases, lanes, outgoing].norm(dim=-1).to(device)
        predicted, captured = _replay_release(
            adapter, data, launch, starts, stops, lanes,
            string_strips=string_strips, separation_m=separation_m)
        by_lead.append(dict(lead_ticks=lead, native_cases=len(valid), **release_error_summary(
            predicted[captured, 3:].norm(dim=-1), native[captured], threshold=threshold, fitted_bias=applied_bias)))
        if lead == 8 and flight_court is not None:
            native_body, body_captured = _replay_release(
                adapter, data, launch, starts, stops, lanes,
                string_strips=string_strips, separation_m=separation_m, native_motion=True)
            chain = _flight_chain_report(adapter, replay.schema, data, streams, ticks, releases, lanes,
                                         predicted, captured, native_body, body_captured, flight_court)
    return dict(schema='native_release_speed_bias_by_lead_v1', by_lead=by_lead,
                excluded_lanes=list(excluded_lanes), independent_releases=len(cases),
                flight_chain=chain,
                scope=('fit paired native releases on training lanes' if training_calibration is None
                       else 'apply frozen training bias to held-out native releases'))


def _replay_release(adapter, data, launch, starts, stops, lanes, *, string_strips, separation_m,
                    native_motion=False):
    device = launch.device
    preview = adapter.action_ball_model.impulse_prior.preview
    geometry = adapter.physical_model.geometry
    horizon = int((stops - starts).max()) + 1
    at = (starts[None] + torch.arange(horizon)[:, None]).minimum(stops[None])
    keys = ('targets', 'previous_action', 'elapsed')
    if native_motion:
        keys += ('state', 'next_state', 'feet')
    rows = {key: data[key][at, lanes].to(device) for key in keys}
    state = data['state'][starts, lanes].to(device)
    feet = data['feet'][starts, lanes].float().to(device)
    present = data['present'][starts, lanes].to(device)
    schedules = launch[lanes.to(device)]
    lengths = (stops - starts + 1).to(device)
    seen = torch.zeros(len(lanes), dtype=torch.bool, device=device)
    strings, captured = seen.clone(), seen.clone()
    predicted = state.new_zeros((len(lanes), 6))
    for step in range(horizon):
        endpoint = None
        if native_motion:
            state = torch.cat((rows['state'][step, :, :49], state[:, 49:]), dim=-1)
            feet = rows['feet'][step].float()
            endpoint = rows['next_state'][step, :, :49]
        state, feet, present, details = ball_transition(
            adapter.physical_model, adapter.ball_model, state, rows['targets'][step],
            rows['previous_action'][step], feet, present, rows['elapsed'][step],
            schedules, action_model=adapter.action_ball_model, return_details=True,
            native_robot_endpoint=endpoint)
        impact = details['impulse_info']
        within = step < lengths
        racket = impact['active'] & (impact['object_id'] == 0) & within
        part = impact['racket_part']
        strings |= racket & ~seen & (part >= 0) & (part < string_strips)
        seen |= racket
        pending = strings & ~captured & within
        if not pending.any():
            continue
        body = decode_physical(state)
        pose = geometry(body['joint_position'], body['joint_velocity'], body['root_position_w'],
                        body['root_quaternion_w'], body['root_link_velocity_w'], body['root_angular_velocity_w'])
        ball = details['predicted_ball']
        release = pending & racket_release_boundary(
            ball, pose, geometry.ids['Racket'], preview, separation_m=separation_m)
        predicted = torch.where(release[:, None], ball, predicted)
        captured |= release
    return predicted, captured


def _flight_chain_report(adapter, schema, data, streams, ticks, releases, lanes,
                         predicted, captured, native_body, body_captured, court):
    from .supermimic_outgoing_flight import predict_flight_events

    columns = _native_event_columns(schema)
    prior = adapter.action_ball_model.impulse_prior
    device = predicted.device
    # Follow recorded prefixes until their first terminal row. A stream break
    # provides an incomplete flight rather than a synthetic landing target.
    ground = columns['rally/ground_seen']
    points = columns['rally/landing_point']
    crossing = columns['rally/crossed_net']
    clearance = columns['rally/net_margin']
    net_contact = columns['rally/shuttle_net_contact']
    landing_time = columns['rally/landing_time']
    clock = columns['auxiliary/environment/ball_clock']
    boundaries = data['next_state'][releases, lanes, 49:].to(device)
    native_velocity = data['next_values'][releases, lanes, columns['rally/outgoing_velocity']].to(device)
    sample_times = (.05, .1, .25, .5)
    evidence = []
    for i, (release, lane) in enumerate(zip(releases.tolist(), lanes.tolist(), strict=True)):
        stop = min(release + 800, len(streams))
        end = release
        for t in range(release, stop):
            if (streams[t] != streams[release] or ticks[t] - ticks[release] != t - release
                    or data['episode'][t, lane] != data['episode'][release, lane]):
                break
            end = t
            if data['terminated'][t, lane] or data['timeout'][t, lane]:
                break
        sampled = {}
        for time in sample_times:
            tick = release + round(time / adapter.dt)
            if tick <= end:
                after = data['next_values'][tick, lane]
                if not (bool(after[ground] > .5) or bool(after[net_contact] > .5)):
                    sampled[time] = data['next_state'][tick, lane, 49:]
        row = data['next_values'][end, lane]
        valid = data['next_valid'][end, lane]
        if bool(row[ground] > .5) and valid[points].bool().all():
            clock_known = bool(data['next_valid'][release, lane, clock])
            elapsed = (float(row[landing_time] - data['next_values'][release, lane, clock])
                       if bool(valid[landing_time]) and clock_known else None)
            evidence.append((i, row[points], bool(row[crossing] > .5),
                             float(row[clearance]) if bool(valid[clearance]) else None, elapsed, sampled))

    def score(ball, supported):
        selected = [row for row in evidence if bool(supported[row[0]])]
        if not selected:
            return dict(native_landings=0, predicted_landings=0)
        indices = torch.tensor([row[0] for row in selected], device=device)
        flight = predict_flight_events(ball[indices, :3], ball[indices, 3:], prior, court,
                                       dt=adapter.dt, sample_times=sample_times)
        landed = flight['landed']
        native_landing = torch.stack([row[1] for row in selected]).to(device)
        native_crossing = torch.tensor([row[2] for row in selected], device=device)
        native_clearance = torch.tensor([float('nan') if row[3] is None else row[3] for row in selected], device=device)
        paired_net = native_clearance.isfinite() & flight['net_clearance'].isfinite()
        durations = torch.tensor([float('nan') if row[4] is None else row[4] for row in selected], device=device)
        paired_time = durations.isfinite() & landed
        traces = {}
        for time, sample in flight['samples'].items():
            ids = [i for i, row in enumerate(selected) if time in row[5]]
            if not ids:
                continue
            native = torch.stack([selected[i][5][time] for i in ids]).to(device)
            predicted_state = torch.cat((sample['position'][ids], sample['velocity'][ids]), dim=-1)
            valid = sample['airborne'][ids] & ~sample['net_contact'][ids]
            error = predicted_state[valid] - native[valid]
            traces[str(time)] = dict(native_samples=len(ids), paired_free_flight_samples=int(valid.sum()),
                position_mae_m=float(error[:, :3].norm(dim=-1).mean()) if len(error) else None,
                velocity_mae_mps=float(error[:, 3:].norm(dim=-1).mean()) if len(error) else None)
        return dict(native_landings=len(selected), predicted_landings=int(landed.sum()),
                    landing_mae_m=float((flight['landing_position'][landed] - native_landing[landed]).norm(dim=-1).mean()) if landed.any() else None,
                    landing_time_mae_s=float((flight['landing_time'][paired_time] - durations[paired_time]).abs().mean()) if paired_time.any() else None,
                    net_crossing_agreement=int((native_crossing == flight['crossed_net']).sum()),
                    net_clearance_cases=int(paired_net.sum()),
                    net_clearance_mae_m=float((flight['net_clearance'][paired_net] - native_clearance[paired_net]).abs().mean()) if paired_net.any() else None,
                    release_speed_mae_mps=float((ball[indices, 3:].norm(dim=-1) - native_velocity[indices].norm(dim=-1)).abs().mean()),
                    free_flight_samples=traces)

    common = captured & body_captured
    shared_landings = [row[0] for row in evidence if bool(common[row[0]])]
    identities = [{'stream_id': int(streams[releases[index]]), 'release_tick': int(ticks[releases[index]]),
                   'lane': int(lanes[index])} for index in shared_landings]
    return dict(common_native_landing_count=len(identities), common_native_landing_ids=identities,
                lead_ticks=8, native_release_to_flight=score(boundaries, common),
                native_robot_contact_to_flight=score(native_body, common),
                predicted_robot_contact_to_flight=score(predicted, common))
