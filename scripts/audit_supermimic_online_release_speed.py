"""Compare the deployed outgoing-speed surrogate with native first-contact and release records."""
import argparse
import json
from collections import defaultdict
from math import hypot
from pathlib import Path
from statistics import mean

import torch

from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest


@torch.no_grad()
def audit(arm, checkpoint, block, flights, *, baseline=None, robot_endpoint_oracle=False):
    command = json.loads((arm / 'registration.json').read_text())['command']

    def option(name):
        return Path(command[command.index('--' + name) + 1])

    keys = ['archive', 'physical-checkpoint', 'ball-checkpoint', 'task-checkpoint',
            'safety-mapping', 'action-ball-checkpoint']
    _, adapter, _, launch, _ = load_physical_loop(
        **{key.replace('-', '_'): option(key) for key in keys},
        execution_root=option('g07-execution-root'), device='cpu')
    strip_count = len(string_strips(load_racket(option('g07-execution-root')/'configs/full_racket_v4_r3.json')))
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    checkpoint_version = saved['global_counters']['accepted']
    checkpoint_actor = tensor_digest(saved['actor'].items())
    actor_sha = block['actor_sha256']
    episode = block.get('episode_index', 0)
    if flights.get('episode_index', 0) != episode or (baseline and baseline.get('episode_index', 0) != episode):
        raise ValueError('Contact and release evidence must use the same episode index')
    if baseline is None:
        if checkpoint_version != block['policy_version'] or checkpoint_actor != actor_sha:
            raise ValueError('Checkpoint and recorded behavior differ')
    elif (baseline['actor_sha256'] != actor_sha or baseline['policy_version'] != block['policy_version']
          or baseline['source_chunks'] != block['source_chunks'] or baseline['world_updates'] >= saved['world_updates']):
        raise ValueError('The comparison requires the same verified native block and a later WM')
    adapter.load_state_dict(saved['world'], strict=True)
    adapter.eval()
    world_sha = tensor_digest(adapter.state_dict().items())
    updates = saved['world_updates']
    guards = {(g['world_updates'], g['world_sha256']) for g in map(
        json.loads, (arm / 'train/online_forward_guard.jsonl').read_text().splitlines())}
    if (updates, world_sha) not in guards or (baseline and (baseline['world_updates'], baseline['world_sha256']) not in guards):
        raise ValueError('Checkpoint world identity missing from the running guard')
    records = block['first_tick_records']
    wanted = defaultdict(list)
    for i, event in enumerate(records):
        for offset in (-1, 0, 1):
            wanted[event['native_tick'] + offset].append((i, offset))
    samples, releases = {}, {}
    measured_endpoints = {}
    for receipt in block['source_chunks']:
        with read_archive_rows(arm / 'train/native_transitions', receipt, scratch=arm) as chunk:
            for row in chunk:
                tick = row['native_tick']
                if row['actor_state_sha256'] != actor_sha or row['policy_version'] != block['policy_version']:
                    raise ValueError('Mixed behavior in audited window')
                before, after = row['before'], row['after']
                current, following = before['task'], after['task']
                released = (~current['released'] & following['released']
                            & (before['episode_index'] == episode) & (after['episode_index'] == episode))
                for lane in released.nonzero().flatten().tolist():
                    releases[lane] = (tick, following['outgoing_velocity'][lane].clone())
                if tick not in wanted:
                    continue
                phys = before['physical']
                state = encode_physical(phys)
                if robot_endpoint_oracle:
                    next_robot = encode_physical(after['physical'])[:, :49]
                for i, offset in wanted[tick]:
                    lane = records[i]['lane']
                    if before['episode_index'][lane] != episode or after['episode_index'][lane] != episode:
                        continue
                    if offset == 0 and (current['had_contact'][lane] or not following['had_contact'][lane]):
                        raise ValueError('First-contact label and archive disagree')
                    if robot_endpoint_oracle and offset == 0:
                        measured_endpoints[i] = next_robot[lane].clone()
                    fields = (state, row['final_joint_targets'], phys['final_action'],
                              phys['foot_contact'].float(), phys['ball_present'], phys['elapsed_s'],
                              after['physical']['ball_velocity_w'])
                    samples[i, offset] = tuple(field[lane].clone() for field in fields)
    usable = [i for i in range(len(records)) if all((i, k) in samples for k in (-1, 0, 1))]
    if not usable:
        raise ValueError('No complete three-tick contact windows')
    lanes = [records[i]['lane'] for i in usable]

    def batch(k):
        return tuple(torch.stack(field) for field in zip(*(samples[i, k] for i in usable)))

    def step(state, targets, previous, feet, present, elapsed):
        nxt, feet, present, details = ball_transition(
            adapter.physical_model, adapter.ball_model, state, targets, previous, feet,
            present, elapsed, launch[lanes], action_model=adapter.action_ball_model, return_details=True)
        info = details['impulse_info']
        racket_hit = info['active'] & (info['object_id'] == 0)
        return nxt, feet, present, racket_hit, info['racket_part'].reshape(-1), details

    state, _, _, feet, present, _, _ = batch(-1)
    outcomes = [dict(**records[i], first_predicted_offset=None, wm_first_outgoing_mps=None,
                     wm_first_racket_part=None, wm_first_string=None, wm_impact_velocity_mps=None) for i in usable]
    for offset in (-1, 0, 1):
        recorded_state, targets, previous, native_feet, native_present, elapsed, velocity = batch(offset)
        state, feet, present, hit, part, details = step(state, targets, previous, feet, present, elapsed)
        toi_velocity = details.get('impact_outgoing_velocity')
        impulse = details['impulse_info']
        action_info = details.get('action_model_info') or {}
        if offset == 0:
            boundary = (recorded_state, targets, previous, native_feet, native_present, elapsed)
            one, _, _, one_hit, one_part, _ = step(*boundary)
            if robot_endpoint_oracle:
                endpoints = torch.stack([measured_endpoints[i] for i in usable])
                def replace_robot(_module, _inputs, output):
                    physical, logits, available = output
                    return torch.cat((endpoints, physical[:, 49:]), -1), logits, available
                with adapter.physical_model.register_forward_hook(replace_robot):
                    oracle, _, _, oracle_hit, oracle_part, _ = step(*boundary)
                if not torch.equal(oracle[:, :49], endpoints):
                    raise ValueError('Measured robot endpoint did not reach the ball transition')
        for j, out in enumerate(outcomes):
            if hit[j] and out['first_predicted_offset'] is None:
                first_part = int(part[j])
                out.update(first_predicted_offset=offset, wm_first_outgoing_mps=float(state[j, -3:].norm()),
                           wm_first_racket_part=first_part, wm_first_string=0 <= first_part < strip_count,
                           wm_impact_velocity_mps=None if toi_velocity is None else toi_velocity[j].tolist())
                if toi_velocity is not None and 'analytic_tangent_dv' in action_info and 'learned_racket_dv' in action_info:
                    normal = impulse['pre_velocity'][j] + impulse['impulse_dv'][j]
                    tangent = action_info['analytic_tangent_dv'][j]
                    residual = action_info['learned_racket_dv'][j]
                    torch.testing.assert_close(normal + (tangent + residual), toi_velocity[j],
                                               rtol=1e-5, atol=1e-6)
                    out['wm_impact_components_mps'] = dict(after_normal=normal.tolist(),
                                                          tangent_dv=tangent.tolist(), residual_dv=residual.tolist())
            if offset == 0:
                out.update(common_boundary_wm_mps=float(one[j, -3:].norm()),
                           common_boundary_native_mps=float(velocity[j].norm()), one_step_hit=bool(one_hit[j]))
                if robot_endpoint_oracle:
                    out.update(common_boundary_oracle_mps=float(oracle[j, -3:].norm()),
                               oracle_hit=bool(oracle_hit[j]), oracle_part=int(oracle_part[j]),
                               one_step_part=int(one_part[j]))
    flips = {f['lane']: f for f in flights['target_score_flips']
             if not f['gained'] and f['online']['failed_conditions'] == ['release_speed']}
    for out in outcomes:
        lane = out['lane']
        tick, velocity = releases.get(lane, (None, None))
        speed = float(velocity.norm()) if velocity is not None else None
        out.update(native_release_tick=tick, native_latched_mps=speed,
                   native_release_velocity_mps=None if velocity is None else velocity.tolist(),
                   sole_speed_loss=lane in flips)
        if lane in flips and (speed is None or abs(speed - flips[lane]['online']['release_speed_mps']) > 1e-5):
            raise ValueError('Native scoring speed does not reproduce the flight report')

    def summarize(rows):
        released = [r for r in rows if r['native_latched_mps'] is not None]
        matched = [r for r in released if r['first_predicted_offset'] is not None]
        return dict(cases=len(rows), native_released=len(released), predicted_racket_with_release=len(matched),
                    native_below_5=sum(r['native_latched_mps'] < 5 for r in released),
                    wm_above_native_below=sum(r['wm_first_outgoing_mps'] >= 5 > r['native_latched_mps'] for r in matched),
                    wm_below_native_above=sum(r['wm_first_outgoing_mps'] < 5 <= r['native_latched_mps'] for r in matched),
                    mean_signed_speed_error_mps=mean(r['wm_first_outgoing_mps']-r['native_latched_mps'] for r in matched) if matched else None)
    strings = [r for r in outcomes if r['online_first_tick_string'] and r.get('online_frame_first') is False]
    result = dict(episode_index=episode, policy_version=block['policy_version'], actor_sha256=actor_sha, world_updates=updates,
                checkpoint_policy_version=checkpoint_version, checkpoint_actor_sha256=checkpoint_actor,
                world_sha256=world_sha, source_chunks=block['source_chunks'], native_first_contacts=len(records),
                scope='Three recorded commands starting one tick before native first contact. Predicted body, ball and feet roll freely. Native scoring latches the last touch velocity. Common-boundary numbers use one step from the native first-contact start.',
                all_windows=summarize(outcomes), string_first=summarize([r for r in outcomes if r['online_first_tick_string']]),
                verified_string_first=summarize(strings),
                sole_speed_losses=summarize([r for r in outcomes if r['sole_speed_loss']]), cases=outcomes)

    held = set(saved['config']['wm_holdout_lanes'])
    result['release_clock'] = dict(
        held=release_clock_errors([r for r in outcomes if r['lane'] in held]),
        train=release_clock_errors([r for r in outcomes if r['lane'] not in held]),
    )
    result['impact_components'] = dict(
        held=impact_component_errors([r for r in outcomes if r['lane'] in held]),
        train=impact_component_errors([r for r in outcomes if r['lane'] not in held]),
    )
    if baseline is not None:
        result['paired_refresh'] = compare_refresh(baseline, outcomes, held)
    return result


def impact_component_errors(rows):
    """Velocity at one predicted TOI compared with the native latched release on the same lane."""
    released = [r for r in rows if r['native_release_velocity_mps'] is not None]
    matched = [r for r in released if r.get('wm_impact_components_mps') is not None]
    result = dict(first_contacts=len(rows), released=len(released), component_cases=len(matched))
    if not matched:
        return result
    terms = [r['wm_impact_components_mps'] for r in matched]
    normal = torch.tensor([t['after_normal'] for t in terms], dtype=torch.float64)
    tangent = torch.tensor([t['tangent_dv'] for t in terms], dtype=torch.float64)
    final = torch.tensor([r['wm_impact_velocity_mps'] for r in matched], dtype=torch.float64)
    native = torch.tensor([r['native_release_velocity_mps'] for r in matched], dtype=torch.float64)
    native_speed = native.norm(dim=-1)
    for key, velocity in [('normal', normal), ('normal_plus_tangent', normal + tangent), ('full', final)]:
        speed = velocity.norm(dim=-1)
        result[key] = dict(component_rmse_mps=float((velocity - native).square().mean().sqrt()),
                           speed_mae_mps=float((speed - native_speed).abs().mean()),
                           false_above_5=int(((speed >= 5) & (native_speed < 5)).sum()),
                           false_below_5=int(((speed < 5) & (native_speed >= 5)).sum()))
    return result


def release_clock_errors(rows):
    released = [r for r in rows if r['native_latched_mps'] is not None]
    matched = [r for r in released if r['wm_impact_velocity_mps'] is not None]

    def score(values):
        errors = [v-r['native_latched_mps'] for v, r in zip(values, matched, strict=True)]
        return dict(mae_mps=mean(map(abs, errors)) if errors else None,
                    bias_mps=mean(errors) if errors else None,
                    false_above=sum(v >= 5 > r['native_latched_mps'] for v, r in zip(values, matched, strict=True)),
                    false_below=sum(v < 5 <= r['native_latched_mps'] for v, r in zip(values, matched, strict=True)))
    return dict(first_contacts=len(rows), released=len(released), common_predictions=len(matched),
                control_end=score([r['wm_first_outgoing_mps'] for r in matched]),
                impact=score([hypot(*r['wm_impact_velocity_mps']) for r in matched]))


def compare_refresh(baseline, outcomes, holdout):
    prior = {(r['lane'], r['native_tick']): r for r in baseline['cases']}
    if len(prior) != len(outcomes) or set(prior) != {(r['lane'], r['native_tick']) for r in outcomes}:
        raise ValueError('A WM refresh comparison requires identical first-contact cases')
    for row in outcomes:
        old = prior[row['lane'], row['native_tick']]
        for k in ('native_release_tick', 'native_latched_mps', 'common_boundary_native_mps', 'sole_speed_loss'):
            if old[k] != row[k]:
                raise ValueError(f'Changed native label {k}')

    def cohort(rows):
        matched = [(prior[r['lane'], r['native_tick']], r) for r in rows]
        released = [(a,b) for a,b in matched if b['native_latched_mps'] is not None]
        both = [(a,b) for a,b in released if all(r['wm_first_outgoing_mps'] is not None for r in (a,b))]
        def scores(version):
            pairs = [p[version] for p in both]
            errors = [r['wm_first_outgoing_mps'] - r['native_latched_mps'] for r in pairs]
            return dict(predicted_hits=sum(p[version]['wm_first_outgoing_mps'] is not None for p in released),
                        common_hit_mae_mps=mean(map(abs, errors)) if errors else None,
                        common_hit_bias_mps=mean(errors) if errors else None,
                        common_hit_false_above=sum(r['wm_first_outgoing_mps'] >= 5 > r['native_latched_mps'] for r in pairs),
                        common_hit_false_below=sum(r['wm_first_outgoing_mps'] < 5 <= r['native_latched_mps'] for r in pairs))
        return dict(cases=len(rows), released=len(released), matched_hits=len(both), before=scores(0), after=scores(1))

    return dict(baseline_world_updates=baseline['world_updates'], baseline_world_sha256=baseline['world_sha256'],
                all=cohort(outcomes), excluded_lanes=cohort([r for r in outcomes if r['lane'] in holdout]),
                training_lanes=cohort([r for r in outcomes if r['lane'] not in holdout]),
                sole_speed_losses=cohort([r for r in outcomes if r['sole_speed_loss']]))


def common_release_scalars(baseline, result, excluded_lanes):
    held = {r['lane']: r for r in baseline['cases'] if r['lane'] in excluded_lanes}
    after = {r['lane']: r for r in result['cases'] if r['lane'] in excluded_lanes}
    if held.keys() != after.keys():
        raise ValueError('Different held-out first-contact lanes')
    common = []
    for lane, prior in held.items():
        current = after[lane]
        if any(prior[k] != current[k] for k in ('native_tick', 'native_release_tick', 'native_release_velocity_mps')):
            raise ValueError('Different native release targets')
        if (prior['native_release_velocity_mps'] is not None
                and all(r['wm_impact_velocity_mps'] is not None for r in (prior, current))):
            common.append(lane)
    released = [r for r in held.values() if r['native_release_velocity_mps'] is not None]
    metrics = dict(common_hits=len(common), native_releases=len(released))
    if released:
        metrics['native/first_contact_step_end_vs_release_speed_mae_mps'] = mean(
            abs(r['common_boundary_native_mps'] - r['native_latched_mps']) for r in released)
        metrics['native/first_contact_step_end_vs_release_threshold_flips'] = sum(
            (r['common_boundary_native_mps'] >= 5) != (r['native_latched_mps'] >= 5) for r in released)
    for stage, rows in [('before', held), ('after', after)]:
        metrics[stage+'/predicted_hits_with_release'] = sum(r['wm_impact_velocity_mps'] is not None
                    and r['native_release_velocity_mps'] is not None for r in rows.values())
        for name, value in release_clock_errors([rows[i] for i in common])['impact'].items():
            if value is not None:
                metrics[stage+'/impact/'+name] = value
    components = impact_component_errors([after[i] for i in common])
    metrics['after/component_cases'] = components['component_cases']
    for component in ('normal', 'normal_plus_tangent', 'full'):
        for name, value in components.get(component, {}).items():
            metrics['after/components/'+component+'/'+name] = value
    return metrics


def publish_refresh(arm, result, baseline, directory):
    """Log the two WMs on the same native contact panel at its first post-refresh proposal."""
    from torch.utils.tensorboard import SummaryWriter
    from scripts.report_supermimic_research_curve import arm_origin, axes
    if baseline is None or any(result[k] != baseline[k] for k in ('source_chunks', 'actor_sha256', 'episode_index')):
        raise ValueError('Expected matching native releases before and after the refresh')
    pair = result['paired_refresh']
    if (pair['baseline_world_sha256'], pair['baseline_world_updates']) != (baseline['world_sha256'], baseline['world_updates']):
        raise ValueError('Baseline differs from the paired refresh audit')
    last_contact = max(c['native_tick'] for c in result['cases'])
    with (arm/'train/training_history.jsonl').open() as stream:
        row = next((r for r in map(json.loads, stream) if r['world_updates'] == result['world_updates']
                    and r['model_runtime_state_sha256'] == result['world_sha256']
                    and r['native_ticks'] > last_contact), None)
    if row is None:
        raise ValueError('No recorded training proposal used this refreshed WM after the measured contacts')
    origin = arm_origin(json.loads((arm/'registration.json').read_text()))
    manifest = json.loads((arm/'train/native_transitions/manifest.json').read_text())
    positions = axes(row, origin, manifest['num_envs'])
    config = json.loads((arm/'train/config.json').read_text())
    metrics = common_release_scalars(baseline, result, set(config['wm_holdout_lanes']))
    with SummaryWriter(str(directory)) as writer:
        for axis, step in positions.items():
            for name, value in metrics.items():
                writer.add_scalar(f'held_release/{axis}/{name}', value, step)
    return positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('arm', 'checkpoint', 'block', 'flights', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--baseline-audit', type=Path, help='Replay exactly the same commands with a later guarded WM')
    parser.add_argument('--tensorboard', type=Path, help='Log the paired native release panel on the three training axes')
    parser.add_argument('--robot-endpoint-oracle', action='store_true',
                        help='Repeat the first-contact step with the measured 49-channel robot endpoint')
    args = parser.parse_args()
    if args.tensorboard and not args.baseline_audit:
        parser.error('--tensorboard requires --baseline-audit')
    torch.set_num_threads(2)
    baseline = json.loads(args.baseline_audit.read_text()) if args.baseline_audit else None
    result = audit(args.arm, args.checkpoint, json.loads(args.block.read_text()), json.loads(args.flights.read_text()),
                   baseline=baseline, robot_endpoint_oracle=args.robot_endpoint_oracle)
    if args.tensorboard:
        result['axes'] = publish_refresh(args.arm, result, baseline, args.tensorboard)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print({key: result[key] for key in ('all_windows', 'string_first', 'sole_speed_losses')})
