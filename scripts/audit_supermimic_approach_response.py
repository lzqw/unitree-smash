"""Replay matched PhysX commands through the exact saved Approach world model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from audit_supermimic_native_window_response import complete_window, window_signature
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_approach_body_response import native_data
from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation
from quadsmash.rl.supermimic_native_manifold_groups import classify, signatures
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES, perturb_request
from quadsmash.rl.supermimic_outgoing_flight import predict_flight_events
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_release_calibration import _replay_release, release_error_summary
from quadsmash.rl.supermimic_toi_racket_objective import racket_features
from run_supermimic_approach_ball_only import registered_options
from scripts.evaluate_supermimic_fresh_wm_panel import assess as assess_forward
from scripts.supermimic_online_forward_guard import compare_forward


def racket_loading(tensors):
    return ((tensors['racket_force_w'][1:].norm(dim=-1) > .001)
            | (tensors['racket_friction_force_w'][1:].norm(dim=-1) > .001))


def contact_face_speed(trace, start, end, *, loaded=None):
    """Compare racket twist at the measured contact time and point on one world axis."""
    t = trace['tensors']
    if loaded is None:
        loaded = racket_loading(t)
    index = loaded.int().argmax(0)
    lanes = torch.arange(len(index))
    have_point = t['racket_contact_position_valid'][index + 1, lanes]
    center = t['racket_position_w'][index, lanes]
    incoming = t['shuttle_state_w'][index, lanes]
    axis = rotation(t['racket_quaternion_w'][index, lanes])[:, :, 0]
    axis *= torch.where(((incoming[:, :3] - center) * axis).sum(-1, keepdim=True) >= 0, 1., -1.)
    point = t['racket_contact_position_w'][index + 1, lanes].nan_to_num()
    native = (t['racket_linear_velocity_w'][index, lanes]
              + torch.linalg.cross(t['racket_angular_velocity_w'][index, lanes], point - center))
    fraction = index[:, None].to(start) / len(loaded)
    forecast = torch.lerp(start, end, fraction)
    expected = forecast[:, 3:6] + torch.linalg.cross(forecast[:, 9:12], point - forecast[:, :3])
    valid = loaded.any(0) & have_point
    native_speed = ((native - incoming[:, 7:10]) * axis).sum(-1)
    predicted_speed = ((expected - incoming[:, 7:10]) * axis).sum(-1)
    return torch.stack((native_speed, predicted_speed), dim=-1), valid


def native_racket_gap(trace, preview, device):
    t = trace['tensors']
    delta = (t['shuttle_state_w'][..., :3] - t['racket_position_w']).to(device)
    attitude = rotation(t['racket_quaternion_w'].to(device))
    local = (attitude.transpose(-1, -2) @ delta[..., None]).squeeze(-1)
    return preview.racket_gap(local).amin(0)


def replay_branch(model, run, launch, device, *, residual_scale=1., cross_body_channels=False):
    body, ball, contact = model.physical_model, model.ball_model, model.action_ball_model
    raw, spec = run['raw'], run['spec']
    lanes = spec['lanes']
    traces = complete_window(raw, spec, contact.impulse_prior.steps)
    initial = raw['start']['before']['physical']
    native_start = encode_physical(initial)[lanes].to(device)
    previous = initial['final_action'][lanes].to(device)
    feet = initial['foot_contact'][lanes].float().to(device)
    present = initial['ball_present'][lanes].to(device)
    elapsed = initial['elapsed_s'][lanes].to(device)
    native_context = (feet, present, elapsed)
    predicted, oracle_ball = native_start, native_start[:, 49:]
    predicted_racket = racket_features(predicted, body.geometry)
    # Swap measured positions/orientations or measured velocities at both
    # control boundaries. The crossed trajectories are offline diagnostics.
    robot_pose = torch.zeros(49, dtype=torch.bool, device=device)
    robot_pose[:7] = True
    robot_pose[13:31] = True
    crossed_balls = ({'measured_pose': oracle_ball, 'measured_velocity': oracle_ball}
                     if cross_body_channels else {})
    measurements = []
    for control, trace in zip(raw['step_sequence'], traces, strict=True):
        target = control['final_joint_targets'][lanes].to(device)
        native_end = control['next_state'][lanes].to(device)
        predicted_start = predicted
        predicted, feet, present, full = ball_transition(
            body, ball, predicted, target, previous, feet, present, elapsed, launch,
            action_model=contact, return_details=True, impulse_residual_scale=residual_scale)
        oracle, _, _, measured = ball_transition(
            body, ball, torch.cat((native_start[:, :49], oracle_ball), dim=-1),
            target, previous, *native_context,
            launch, action_model=contact, native_robot_endpoint=native_end[:, :49],
            return_details=True, impulse_residual_scale=residual_scale)
        oracle_ball = oracle[:, 49:]
        endpoint_racket = racket_features(predicted, body.geometry)
        tensors = trace['tensors']
        loaded = racket_loading(tensors)
        face_speed, face_supported = contact_face_speed(
            trace, predicted_racket.cpu(), endpoint_racket.cpu(), loaded=loaded)
        native_velocity = tensors['shuttle_state_w'][..., 7:10]
        row = dict(native_ball=native_end[:, 52:], predicted_ball=predicted[:, 52:],
                   measured_body_ball=oracle_ball[:, 3:],
                   native_racket=racket_features(native_end, body.geometry),
                   predicted_racket=endpoint_racket, face_speed=face_speed, face_supported=face_supported,
                   native_load_velocity_change=(native_velocity.diff(dim=0) * loaded[..., None]).sum(0),
                   native_gap=native_racket_gap(trace, contact.impulse_prior.preview, device))
        details = {'predicted': full, 'measured_body': measured}
        for name in crossed_balls:
            observed = robot_pose if name == 'measured_pose' else ~robot_pose
            start = torch.where(observed, native_start[:, :49], predicted_start[:, :49])
            endpoint = torch.where(observed, native_end[:, :49], predicted[:, :49])
            crossed_state, _, _, details[name] = ball_transition(
                body, ball, torch.cat((start, crossed_balls[name]), dim=-1), target, previous,
                *native_context, launch, action_model=contact, native_robot_endpoint=endpoint,
                return_details=True, impulse_residual_scale=residual_scale)
            crossed_balls[name] = crossed_state[:, 49:]
            row[name + '_ball'] = crossed_state[:, 52:]
        for name, detail in details.items():
            impact = detail['impulse_info']
            row[name + '_hit'] = impact['active'] & (impact['object_id'] == 0)
            row[name + '_impact_time'] = impact['time']
            row[name + '_gap'] = impact['min_gap'][:, 0]
            for key in ('racket_part', 'normal_speed', 'normal', 'surface_velocity', 'impulse_dv'):
                row[name + '_' + key] = impact[key]
            residual = (detail['action_model_info'] or {}).get('impulse_residual_dv')
            row[name + '_residual_dv'] = (torch.zeros_like(impact['impulse_dv']) if residual is None
                                          else residual * residual_scale)
        measurements.append(row)
        predicted_racket = endpoint_racket
        native_start = native_end
        previous = control['final_action'][lanes].to(device)
        native_context = (control['foot_contact'][lanes].float().to(device),
                          control['ball_present'][lanes].to(device),
                          control['elapsed_s'][lanes].to(device))
        elapsed = elapsed + body.dt
    result = {key: torch.stack([row[key] for row in measurements]).cpu() for key in measurements[0]}
    local_signatures = [signatures(trace) for trace in traces]
    result['signatures'] = [
        window_signature([row[lane] for row in local_signatures], contact.impulse_prior.steps)
        for lane in range(len(lanes))]
    return result


def assess(branches, lanes, steps):
    def response(field, ticks, indices, *, target=None):
        field_data = branches['plus'][field] - branches['minus'][field]
        native = field if target is None else target
        native_delta = branches['plus'][native] - branches['minus'][native]
        disagreement = branches['nominal_b'][native] - branches['nominal_a'][native]
        return response_metrics(native_delta[ticks, indices], field_data[ticks, indices],
                                disagreement[ticks, indices])

    horizon = len(branches['nominal_a']['native_ball'])
    paths = tuple(name for name in ('predicted', 'measured_body', 'measured_pose', 'measured_velocity')
                  if name + '_hit' in branches['nominal_a'])
    predicted_ball_fields = tuple(name + '_ball' for name in paths)
    labels = list(zip(*(branches[b]['signatures'] for b in BRANCHES), strict=True))
    per_case = []
    for index, (lane, sig) in enumerate(zip(lanes, labels, strict=True)):
        first_hit = min((x[3][0] // steps for x in sig if x[3]), default=None)
        last_hit = max((x[3][-1] // steps for x in sig if x[3]), default=None)
        settled_tick = last_hit + 1 if last_hit is not None and last_hit + 1 < horizon else None
        tick = settled_tick if settled_tick is not None else horizon - 1
        row = dict(lane=lane, group=classify(sig), first_native_load_tick=first_hit,
                   last_native_load_tick=last_hit, sampled_tick=tick, settled_tick=settled_tick)
        for field in ('native_ball', *predicted_ball_fields):
            row[field + '_delta_mps'] = (branches['plus'][field] - branches['minus'][field])[tick, index].tolist()
        nominal = branches['nominal_a']['native_ball'][tick, index]
        low = nominal - branches['minus']['native_ball'][tick, index]
        high = branches['plus']['native_ball'][tick, index] - nominal
        repeated = branches['nominal_b']['native_ball'][tick, index] - nominal
        row['native_left_delta_mps'] = low.tolist()
        row['native_right_delta_mps'] = high.tolist()
        row['native_repeat_delta_mps'] = repeated.tolist()
        measurable = torch.minimum(low.norm(), high.norm()) > max(1e-6, 3 * float(repeated.norm()))
        row['native_one_sided_direction_cosine'] = (
            float(torch.nn.functional.cosine_similarity(low[None], high[None])) if measurable else None)
        for branch, series in branches.items():
            frame = series['signatures'][index][3]
            measured_tick = frame[0] // steps if frame else None
            row[branch + '_native_first_load_tick'] = measured_tick
            row[branch + '_normal_face_speed_mps'] = (
                series['face_speed'][measured_tick, index].tolist()
                if measured_tick is not None and series['face_supported'][measured_tick, index] else None)
            row[branch + '_native_load_velocity_change_mps'] = (
                series['native_load_velocity_change'][:tick + 1, index].sum(0).tolist())
            row[branch + '_minimum_racket_gap_m'] = {
                k: float(series[k + '_gap'][:, index].min()) for k in ('native', 'predicted', 'measured_body')}
            for path in paths:
                hits = series[path + '_hit'][:, index].nonzero().flatten().tolist()
                row[branch + '_' + path + '_load_ticks'] = hits
                row[branch + '_' + path + '_first_impact_in_tick_s'] = (
                    float(series[path + '_impact_time'][hits[0], index]) if hits else None)
                impulse_events = []
                for hit in (load_tick for load_tick in hits if load_tick <= tick):
                    event = {'tick': hit, 'racket_part': int(series[path + '_racket_part'][hit, index])}
                    for field in ('normal_speed', 'normal', 'surface_velocity', 'impulse_dv', 'residual_dv'):
                        event[field] = series[path + '_' + field][hit, index].tolist()
                    impulse_events.append(event)
                row[branch + '_' + path + '_racket_impulse_events'] = impulse_events
        per_case.append(row)
    def summarize(indices):
        ticks = [per_case[i]['sampled_tick'] for i in indices]
        result = {field: response(field, ticks, indices, target='native_ball')
                  for field in predicted_ball_fields}
        def speed_delta(field, left, right):
            return (branches[right][field][ticks, indices].norm(dim=-1)
                    - branches[left][field][ticks, indices].norm(dim=-1))
        native = speed_delta('native_ball', 'minus', 'plus')
        noise = speed_delta('native_ball', 'nominal_a', 'nominal_b')
        supported = native.abs() > torch.maximum(3 * noise.abs(), torch.full_like(noise, 1e-5))
        for field, metrics in result.items():
            modeled = speed_delta(field, 'minus', 'plus')
            metrics['speed_delta_rmse_mps'] = float((modeled - native).square().mean().sqrt())
            metrics['speed_sign_support'] = int(supported.sum())
            metrics['speed_sign_matches'] = int((supported & (modeled * native > 0)).sum())
        return result
    groups = {'all': summarize(list(range(len(per_case))))}
    for group in sorted({row['group'] for row in per_case}):
        groups[group] = summarize([i for i, row in enumerate(per_case) if row['group'] == group])
    finished = [i for i, row in enumerate(per_case) if row['settled_tick'] is not None]
    if finished:
        groups['observed_post_contact_boundary'] = summarize(finished)
    boundaries = {}
    spans = dict(center=slice(0, 3), velocity=slice(3, 6),
                 normal=slice(6, 9), angular_velocity=slice(9, 12))
    for tick in (i for i in (0, 3, 7, 15) if i < horizon):
        nominal = branches['nominal_a']
        native = branches['plus']['native_racket'][tick] - branches['minus']['native_racket'][tick]
        predicted = branches['plus']['predicted_racket'][tick] - branches['minus']['predicted_racket'][tick]
        repeat = branches['nominal_b']['native_racket'][tick] - branches['nominal_a']['native_racket'][tick]
        boundaries[tick + 1] = {name: response_metrics(native[:, span], predicted[:, span], repeat[:, span])
                                for name, span in spans.items()}
        for name, span in spans.items():
            absolute = nominal['predicted_racket'][tick, :, span] - nominal['native_racket'][tick, :, span]
            boundaries[tick + 1][name]['nominal_state_rmse'] = float(absolute.square().mean().sqrt())
    face_rows = [r for r in per_case if all(r[b + '_normal_face_speed_mps'] is not None for b in BRANCHES)]
    normal_response = None
    if face_rows:
        def delta(left, right):
            return (torch.tensor([r[right + '_normal_face_speed_mps'] for r in face_rows])
                    - torch.tensor([r[left + '_normal_face_speed_mps'] for r in face_rows]))
        measured, predicted = delta('minus', 'plus').split(1, dim=-1)
        repeat = delta('nominal_a', 'nominal_b')[:, :1]
        normal_response = response_metrics(measured, predicted, repeat)
        normal_response['scope'] = 'Signed approach speed at native first load; [native racket, predicted racket] per case'
    return dict(contact_groups=dict(Counter(r['group'] for r in per_case)),
                post_contact_ball=groups, racket_by_control_tick=boundaries,
                measured_contact_face_speed_response=normal_response, cases=per_case)


def e1_release_report(model, data, held, launch, string_count, scoring, separation_m):
    cases = []
    for release, lane in data['qualified_release'].nonzero().tolist():
        if lane not in held:
            continue
        first = max(0, release - 16)
        edges = data['new_contact'][first:release + 1, lane].nonzero().flatten()
        if len(edges):
            cases.append((first + int(edges[-1]), release, lane))
    report = {}
    for lead in (1, 4, 8, 16):
        windows = []
        for hit, release, lane in cases:
            start, stop = hit - lead, min(release + 4, len(data['tick']) - 1)
            if start < 0:
                continue
            episode = data['episode'][start, lane]
            while stop > release and data['episode'][stop, lane] != episode:
                stop -= 1
            span = slice(start, stop + 1)
            if ((data['episode'][span, lane] == episode).all()
                    and (data['stream'][span] == data['stream'][start]).all()
                    and (data['tick'][span].diff() == 1).all()):
                windows.append((start, stop, release, lane))
        if not windows:
            continue
        start, stop, release, lanes = torch.tensor(windows).T
        native = data['outgoing_velocity'][release, lanes].to(launch.device)
        predicted, captured = _replay_release(model, data, launch, start, stop, lanes,
                                               string_strips=string_count, separation_m=separation_m)
        pv, nv = predicted[captured, 3:], native[captured]
        values = release_error_summary(pv.norm(dim=-1), nv.norm(dim=-1),
                                       threshold=scoring['minimum_outgoing_speed_mps'], fitted_bias=0.)
        if captured.any():
            values['release_vector_rmse_mps'] = float((pv - nv).square().mean().sqrt())
            values['release_vector_cosine_median'] = float(torch.nn.functional.cosine_similarity(pv, nv).median())
        report[lead] = dict(
            values, eligible=len(windows), captured_lanes=lanes[captured.cpu()].tolist(),
            cases=[dict(lane=int(lane), episode=int(data['episode'][tick, lane]), native=velocity.tolist(),
                        predicted=prediction.tolist() if bool(valid) else None)
                   for lane, tick, velocity, prediction, valid in
                   zip(lanes, release, native.cpu(), predicted[:, 3:].cpu(), captured.cpu(), strict=True)])
    return report


def flight_from_native_release(data, episodes, actor_hash, held, prior, court):
    """Audit ball-only flight from the first recorded boundary after release."""
    terminal = {(row['env_id'], row['episode_index']): row for row in episodes
                if row['behavior_at_terminal']['actor_sha256'] == actor_hash}
    starts, lanes, boundaries, labels = [], [], [], []
    excluded = Counter()
    for tick, lane in data['qualified_release'].nonzero().tolist():
        if lane not in held:
            continue
        episode = int(data['episode'][tick, lane])
        row = terminal.get((lane, episode))
        if row is None or row['rally_fault'] or row['numerical_failures']:
            excluded['unmatched_or_faulted_terminal'] += 1
            continue
        if data['net_contact_at_end'][tick, lane]:
            excluded['net_contact_before_boundary'] += 1
            continue
        flight = row['native_flight']
        before = float(data['ball_clock'][tick, lane])
        boundary = float(data['next_ball_clock'][tick, lane])
        if not before <= flight['release_time_s'] <= boundary:
            raise ValueError(f'Native release {flight["release_time_s"]} outside the ball clock interval '
                             f'[{before}, {boundary}] in lane {lane}')
        starts.append(tick)
        lanes.append(lane)
        boundaries.append(boundary)
        labels.append(flight)
    if not starts:
        raise ValueError('No held-out native release with a compatible terminal observation')

    start, lane = torch.tensor(starts), torch.tensor(lanes)
    boundary = torch.tensor(boundaries, dtype=torch.float64)
    ball = data['next_state'][start, lane, 49:].to(prior.wind)
    sample_times = (.05, .10, .25, .50)
    predicted = predict_flight_events(ball[:, :3], ball[:, 3:], prior, court,
                                      dt=prior.dt, sample_times=sample_times)
    landing_time = torch.tensor([r['first_landing_time_s'] for r in labels], dtype=torch.float64)
    crossing_time = torch.tensor([r['crossing_time_s'] for r in labels], dtype=torch.float64)
    net_contact = torch.tensor([r['shuttle_net_contact'] for r in labels])
    native_landing = landing_time.isfinite() & (landing_time > boundary)
    native_crossing = crossing_time.isfinite() & (crossing_time > boundary)
    predicted_landing = predicted['landed'].cpu()
    predicted_crossing = predicted['crossed_net'].cpu()
    landing_matched = native_landing & ~net_contact & predicted_landing
    crossing_matched = native_crossing & ~net_contact & predicted_crossing
    landing = torch.tensor([r['first_landing_m'] for r in labels])
    clearance = torch.tensor([r['net_margin_m'] for r in labels])
    summary = {
        'cases': len(starts), 'excluded': dict(excluded),
        'observed_ground_events': int(native_landing.sum()), 'observed_crossings': int(native_crossing.sum()),
        'missed_crossings': int((native_crossing & ~predicted_crossing).sum()),
        'post_boundary_net_contact': int(net_contact.sum()), 'predicted_net_contact': int(predicted['net_contact'].sum()),
        'matched_no_net_contact_landings': int(landing_matched.sum()),
        'matched_no_net_contact_crossings': int(crossing_matched.sum()),
    }
    if landing_matched.any():
        error = predicted['landing_position'].cpu() - landing
        summary['landing_xy_rmse_m'] = float(error[landing_matched, :2].square().sum(-1).mean().sqrt())
        delta = predicted['landing_time'].cpu() + boundary - landing_time
        summary['landing_time_mae_s'] = float(delta[landing_matched].abs().mean())
    if crossing_matched.any():
        delta = predicted['crossing_time'].cpu() + boundary - crossing_time
        summary['crossing_time_mae_s'] = float(delta[crossing_matched].abs().mean())
        difference = predicted['net_clearance'].cpu() - clearance
        summary['net_clearance_mae_m'] = float(difference[crossing_matched].abs().mean())

    samples = {}
    for horizon, sample in predicted['samples'].items():
        end = start + round(horizon / prior.dt)
        safe = end.clamp_max(len(data['tick']) - 1)
        support = (native_landing & ~net_contact & (boundary + horizon < landing_time)
                   & sample['airborne'].cpu() & ~sample['net_contact'].cpu()
                   & (end < len(data['tick'])) & (data['tick'][safe] - data['tick'][start] == end - start)
                   & (data['stream'][safe] == data['stream'][start])
                   & (data['episode'][safe, lane] == data['episode'][start, lane]))
        samples[str(horizon)] = {'cases': int(support.sum())}
        if support.any():
            truth = data['next_state'][end[support], lane[support], 49:]
            estimate = torch.cat((sample['position'], sample['velocity']), -1).cpu()[support]
            position, velocity = (estimate - truth).split(3, dim=-1)
            samples[str(horizon)].update(position_rmse_m=float(position.square().sum(-1).mean().sqrt()),
                                         velocity_rmse_mps=float(velocity.square().sum(-1).mean().sqrt()))
    summary.update(samples=samples, scope='Flight from a measured control boundary after release. Event timing '
                   'is evaluated only with a native witness. Trajectory errors use airborne windows without net contact.')
    return summary


def calibration_bias(report):
    bias = {}
    for lead, values in report.items():
        matched = [case for case in values['cases'] if case['predicted'] is not None]
        if matched:
            actual = torch.tensor([case['native'] for case in matched]).norm(dim=-1)
            forecast = torch.tensor([case['predicted'] for case in matched]).norm(dim=-1)
            bias[lead] = dict(matched=len(matched), bias_mps=float((forecast - actual).quantile(.5)))
    return bias


def compare_e1_release_reports(reference, candidate, minimum_speed, *, biases):
    report = {}
    for lead in reference:
        def indexed(rows):
            return {(case['lane'], case['episode']): case
                    for case in rows['cases'] if case['predicted'] is not None}
        original, proposed = indexed(reference[lead]), indexed(candidate[lead])
        common = sorted(original.keys() & proposed.keys())
        if not common:
            continue
        native = torch.tensor([original[i]['native'] for i in common])
        torch.testing.assert_close(native, torch.tensor([proposed[i]['native'] for i in common]))
        paired = {}
        for key, values in (('reference', original), ('candidate', proposed)):
            predicted = torch.tensor([values[i]['predicted'] for i in common])
            bias = biases[key].get(lead, {}).get('bias_mps', 0.)
            summary = release_error_summary(predicted.norm(dim=-1), native.norm(dim=-1),
                                            threshold=minimum_speed, fitted_bias=bias)
            summary['release_vector_rmse_mps'] = float((predicted - native).square().mean().sqrt())
            paired[key] = summary
        report[lead] = dict(paired, common_cases=[list(case) for case in common])
    return report


def command_path_jacobian(model, runs, launch, rows, device, *, residual_scale=1.):
    """Differentiate the midpoint of the two measured full command sequences."""
    before = runs['nominal_a']['raw']['start']['before']['physical']
    lanes = runs['nominal_a']['spec']['lanes']
    selected = [i for i, row in enumerate(rows)
                if row['group'] == 'racket_stable' and row['settled_tick'] is not None]
    if not selected:
        return dict(rows=0)
    native_lanes = [lanes[i] for i in selected]
    stops = torch.tensor([rows[i]['settled_tick'] for i in selected], device=device)
    sequence = {branch: runs[branch]['raw']['step_sequence'] for branch in ('minus', 'plus')}
    body, ball, contact = model.physical_model, model.ball_model, model.action_ball_model
    with torch.enable_grad():
        alpha = torch.full((len(selected),), .5, device=device, requires_grad=True)
        state = encode_physical(before)[native_lanes].to(device)
        previous = before['final_action'][native_lanes].to(device)
        feet = before['foot_contact'][native_lanes].float().to(device)
        present = before['ball_present'][native_lanes].to(device)
        elapsed = before['elapsed_s'][native_lanes].to(device)
        output = state.new_zeros((len(selected), 3))
        for tick in range(int(stops.max()) + 1):
            def interpolate(field):
                minus, plus = (sequence[branch][tick][field][native_lanes].to(device)
                               for branch in ('minus', 'plus'))
                return torch.lerp(minus, plus, alpha[:, None])
            target = interpolate('final_joint_targets')
            state, feet, present = ball_transition(
                body, ball, state, target, previous, feet, present, elapsed, launch[selected],
                action_model=contact, impulse_residual_scale=residual_scale)
            output = torch.where((tick == stops)[:, None], state[:, 52:], output)
            previous = interpolate('final_action')
            elapsed = elapsed + body.dt
        # Each batch row only depends on its own interpolation coordinate.
        derivative = torch.stack([torch.autograd.grad(output[:, j].sum(), alpha, retain_graph=j < 2)[0]
                                  for j in range(3)], dim=-1).detach().cpu()
    native = torch.tensor([rows[i]['native_ball_delta_mps'] for i in selected])
    finite = torch.tensor([rows[i]['predicted_ball_delta_mps'] for i in selected])
    ticks = stops.cpu()
    low, high = (runs[branch]['raw']['step_sequence'] for branch in ('nominal_a', 'nominal_b'))
    noise = torch.stack([high[t]['next_state'][lane, 52:] - low[t]['next_state'][lane, 52:]
                         for t, lane in zip(ticks, native_lanes, strict=True)])
    return dict(rows=len(selected), cases=[dict(lane=lane, model_local_derivative_mps=value.tolist())
                                          for lane, value in zip(native_lanes, derivative, strict=True)],
                compared_to_native=response_metrics(native, derivative, noise),
                compared_to_model_finite=response_metrics(finite, derivative, torch.zeros_like(finite)))


def load_forward_panel(path):
    spec = json.loads(path.read_text())
    artifacts = {}
    for name in ('panel', 'capture_metrics'):
        item = spec[name]
        artifact = path.parent / item['path']
        if sha(artifact) != item['sha256']:
            raise ValueError(f'Changed registered forward panel artifact {name}')
        artifacts[name] = json.loads(artifact.read_text())
    panel, capture = artifacts['panel'], artifacts['capture_metrics']
    native = (path.parent / spec['capture_metrics']['path']).parent / 'native.pt'
    if capture['panel_sha256'] != spec['panel']['sha256'] or sha(native) != capture['data_sha256']:
        raise ValueError('Changed native forward trajectories')
    data = torch.load(native, map_location='cpu', weights_only=True, mmap=True)
    return panel, capture, data


def score_forward_panel(model, forward_panel, device):
    panel, capture, data = forward_panel
    result = assess_forward(data, panel['cases'], None, model, panel['horizons'], device, predictors=('candidate',))
    if result['windows'] != capture['windows']:
        raise ValueError('Changed forward evaluation windows')
    return result['metrics']['candidate']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'registration', 'collection', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--body-candidate', type=Path)
    parser.add_argument('--e1-archive', type=Path)
    parser.add_argument('--flight-from-native-release', action='store_true',
                        help='Compare ball flight from measured separation boundaries using the E1 archive')
    parser.add_argument('--directional-jacobian', action='store_true')
    parser.add_argument('--cross-body-channels', action='store_true',
                        help='Compare native pose and native velocity substitutions independently')
    parser.add_argument('--long-forward-guard', action='store_true',
                        help='Also compare H1–H64 on the registered native panel')
    parser.add_argument('--impulse-residual-scale', default=1., type=float, choices=(0., .5, 1.))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.e1_archive and args.impulse_residual_scale != 1.:
        parser.error('E1 archive calibration uses the recorded residual scale')
    if args.flight_from_native_release and not args.e1_archive:
        parser.error('Native flight comparison requires the E1 archive')
    if args.long_forward_guard and (not args.body_candidate or args.impulse_residual_scale != 1.):
        parser.error('Long forward comparison requires a Body candidate and the recorded residual scale')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    registration = json.loads(args.registration.read_text())['command']
    options = registered_options(registration,
                                 ('archive', 'physical-checkpoint', 'ball-checkpoint', 'task-checkpoint',
                                  'safety-mapping', 'action-ball-checkpoint', 'g07-execution-root'))
    values = {options[i][2:].replace('-', '_'): options[i + 1] for i in range(0, len(options), 2)}
    values['execution_root'] = values.pop('g07_execution_root')
    _, model, _, launch, manifest = load_physical_loop(**values, device=args.device)
    model.load_state_dict(checkpoint['world'], strict=True)
    model.eval()
    parent_hash = tensor_digest(model.state_dict().items())
    if parent_hash != checkpoint['wm_fit_provenance']['world_state_sha256']:
        raise ValueError('Checkpoint World Model digest mismatch')
    status_path = args.collection / 'status.json'
    status = json.loads(status_path.read_text())
    rows = status['completed']
    if (status['status'] != 'COLLECTION_FINISHED' or len(rows) != len(BRANCHES)
            or any(r['exit_code'] for r in rows) or {r['branch'] for r in rows} != set(BRANCHES)):
        raise ValueError('Expected one complete four-branch PhysX collection')
    specs = {row['spec'] for row in rows}
    if len(specs) != 1:
        raise ValueError('Mismatched probe families')
    folder = args.collection / specs.pop()
    for row in rows:
        if sha(folder / row['branch'] / 'native_probe.pt') != row['probe_sha256']:
            raise ValueError('Native probe hash mismatch')
    runs = {b: _load_branch(folder, b) for b in BRANCHES}
    _same_contract(runs)
    reference = runs['nominal_a']
    if reference['task_contract'] not in (manifest['task_contract'], checkpoint['metadata']['task_contract']):
        raise ValueError('Different physical source contract')
    spec = reference['spec']
    lanes = spec['lanes']
    probe_actor_hash = tensor_digest(reference['actor_state'].items())
    if spec.get('probe_actor_sha256', probe_actor_hash) != probe_actor_hash:
        raise ValueError('Unexpected actor for the declared probe')
    for branch, run in runs.items():
        if tensor_digest(run['actor_state'].items()) != probe_actor_hash:
            raise ValueError('Mismatched actor weights across physical branches')
        raw = run['raw']
        torch.testing.assert_close(raw['request_after'], perturb_request(raw['request_before'], spec, branch),
                                   atol=0., rtol=0.)
    if set(lanes) - set(checkpoint['config']['wm_holdout_lanes']):
        raise ValueError('Probe lanes overlap the saved WM training partition')
    long_forward = None
    if args.long_forward_guard:
        guard_path = Path(registered_options(registration, ('wm-forward-guard',))[1])
        panel = load_forward_panel(guard_path)
        initial = score_forward_panel(model, panel, args.device)
    native_e1 = None
    if args.e1_archive:
        original = args.e1_archive.parent / 'evaluation_boundary.json'
        actor = json.loads(original.read_text())['invocation']['start']['actor_sha256']
        phase_task = checkpoint['metadata']['task_contract']
        data, native_hashes = native_data(args.e1_archive, phase_task, actor)
        path = Path(values['execution_root']) / 'configs/full_racket_v4_r3.json'
        string_count = len(string_strips(load_racket(path)))
        separation_m = json.loads(Path('configs/shuttle_physics_v2.json').read_text())['separation_hysteresis_m']
        held = set(checkpoint['config']['wm_holdout_lanes'])
        fitting = set(range(len(launch))) - held
        e1_arguments = (launch, string_count, phase_task['approach_ball_scoring'], separation_m)
        native_e1 = dict(chunks=native_hashes, reference=e1_release_report(model, data, held, *e1_arguments),
                         biases=dict(reference=calibration_bias(
                             e1_release_report(model, data, fitting, *e1_arguments))))
        if args.flight_from_native_release:
            episodes = args.e1_archive.parent / 'native_episodes.jsonl'
            records = [json.loads(line) for line in episodes.read_text().splitlines()]
            court = json.loads(Path('configs/court_net_v4.json').read_text())
            native_e1['flight_from_native_release'] = flight_from_native_release(
                data, records, actor, held, model.action_ball_model.impulse_prior, court)
    parent_probe = None
    if args.body_candidate:
        candidate = torch.load(args.body_candidate, map_location='cpu', weights_only=True)
        if (candidate['parent_world_sha256'] != parent_hash
                or candidate['schema'] != model.physical_model.schema
                or set(candidate['train_lanes']).intersection(checkpoint['config']['wm_holdout_lanes'])):
            raise ValueError('Incompatible Body WM or overlapping candidate fitting lanes')
        if tensor_digest(candidate['model'].items()) != candidate['body_sha256']:
            raise ValueError('Candidate Body WM digest mismatch')
        baseline = {b: replay_branch(model, run, launch[lanes], args.device,
                                     residual_scale=args.impulse_residual_scale,
                                     cross_body_channels=args.cross_body_channels) for b, run in runs.items()}
        parent_probe = assess(baseline, lanes, model.action_ball_model.impulse_prior.steps)
        model.physical_model.load_state_dict(candidate['model'], strict=True)
        if args.long_forward_guard:
            revised = score_forward_panel(model, panel, args.device)
            long_forward = dict(comparison=compare_forward(initial, revised, panel[0]['horizons']),
                                parent=initial, candidate=revised, panel_sha256=panel[1]['panel_sha256'],
                                archive_sha256=panel[1]['data_sha256'], windows=len(panel[1]['windows']))
        if native_e1 is not None:
            native_e1['candidate'] = e1_release_report(model, data, held, *e1_arguments)
            native_e1['biases']['candidate'] = calibration_bias(
                e1_release_report(model, data, fitting, *e1_arguments))
            native_e1['common_predicted_releases'] = compare_e1_release_reports(
                native_e1['reference'], native_e1['candidate'],
                phase_task['approach_ball_scoring']['minimum_outgoing_speed_mps'], biases=native_e1['biases'])
    weight_hash = tensor_digest(model.state_dict().items())
    measurements = {b: replay_branch(model, run, launch[lanes], args.device,
                                     residual_scale=args.impulse_residual_scale,
                                     cross_body_channels=args.cross_body_channels) for b, run in runs.items()}
    report = assess(measurements, lanes, model.action_ball_model.impulse_prior.steps)
    if long_forward is not None:
        report['long_forward'] = long_forward
    if args.directional_jacobian:
        model.requires_grad_(False)
        report['recorded_command_path_jacobian'] = command_path_jacobian(
            model, runs, launch[lanes], report['cases'], args.device, residual_scale=args.impulse_residual_scale)
    report.update(world_sha256=weight_hash, parent_world_sha256=parent_hash, e1_releases=native_e1,
                  parent_probe=parent_probe,
                  body_candidate_sha256=sha(args.body_candidate) if args.body_candidate else None,
                  probe_status_sha256=sha(status_path),
                  checkpoint=str(args.checkpoint.resolve()), source_contract=reference['task_contract'],
                  probe_actor_sha256=probe_actor_hash,
                  impulse_residual_scale=args.impulse_residual_scale,
                  crossed_body_channels=args.cross_body_channels,
                  epsilon_request=spec['epsilon_request'], seed=reference['seed'],
                  probe_policy_execution=reference['policy_execution'],
                  scope='Archived PhysX interventions with executed targets and the current saved WM. '
                        'Ball comparison uses the first complete control boundary after the last observed '
                        'native load, or the terminal probe boundary when that sample is unavailable. '
                        'The measured body condition supplies native control-boundary robot states. '
                        'Native load velocity change sums observed substep velocity differences while the '
                        'racket sensor reports normal load or friction, through the sampled boundary. '
                        'It can include flight and simultaneous forces. Model impulse events through '
                        'the same boundary report pre-integration velocity changes per impact. '
                        'Crossed body channels substitute native position and rotation or native '
                        'velocity at both control boundaries before predicting ball contact. '
                        'Policy execution and checkpoint identity are recorded separately.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
