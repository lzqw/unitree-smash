"""Compare the deployed outgoing-speed surrogate with native first-contact and release records."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest


@torch.no_grad()
def audit(arm, checkpoint, block, flights, *, baseline=None):
    command = json.loads((arm / 'registration.json').read_text())['command']

    def option(name):
        return Path(command[command.index('--' + name) + 1])

    keys = ['archive', 'physical-checkpoint', 'ball-checkpoint', 'task-checkpoint',
            'safety-mapping', 'action-ball-checkpoint']
    _, adapter, _, launch, _ = load_physical_loop(
        **{key.replace('-', '_'): option(key) for key in keys},
        execution_root=option('g07-execution-root'), device='cpu')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    checkpoint_version = saved['global_counters']['accepted']
    checkpoint_actor = tensor_digest(saved['actor'].items())
    actor_sha = block['actor_sha256']
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
    for receipt in block['source_chunks']:
        with read_archive_rows(arm / 'train/native_transitions', receipt, scratch=arm) as chunk:
            for row in chunk:
                tick = row['native_tick']
                if row['actor_state_sha256'] != actor_sha or row['policy_version'] != block['policy_version']:
                    raise ValueError('Mixed behavior in audited window')
                before, after = row['before'], row['after']
                current, following = before['task'], after['task']
                released = (~current['released'] & following['released'] & (before['episode_index'] == 0))
                for lane in released.nonzero().flatten().tolist():
                    releases[lane] = (tick, float(following['outgoing_velocity'][lane].norm()))
                if tick not in wanted:
                    continue
                phys = before['physical']
                state = encode_physical(phys)
                for i, offset in wanted[tick]:
                    lane = records[i]['lane']
                    if before['episode_index'][lane] != 0 or after['episode_index'][lane] != 0:
                        continue
                    if offset == 0 and (current['had_contact'][lane] or not following['had_contact'][lane]):
                        raise ValueError('First-contact label and archive disagree')
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
        return nxt, feet, present, info['active'] & (info['object_id'] == 0)

    state, _, _, feet, present, _, _ = batch(-1)
    outcomes = [dict(**records[i], first_predicted_offset=None, wm_first_outgoing_mps=None) for i in usable]
    for offset in (-1, 0, 1):
        recorded_state, targets, previous, native_feet, native_present, elapsed, velocity = batch(offset)
        state, feet, present, hit = step(state, targets, previous, feet, present, elapsed)
        if offset == 0:
            one, _, _, one_hit = step(recorded_state, targets, previous, native_feet, native_present, elapsed)
        for j, out in enumerate(outcomes):
            if hit[j] and out['first_predicted_offset'] is None:
                out.update(first_predicted_offset=offset, wm_first_outgoing_mps=float(state[j, -3:].norm()))
            if offset == 0:
                out.update(common_boundary_wm_mps=float(one[j, -3:].norm()),
                           common_boundary_native_mps=float(velocity[j].norm()), one_step_hit=bool(one_hit[j]))
    flips = {f['lane']: f for f in flights['target_score_flips']
             if not f['gained'] and f['online']['failed_conditions'] == ['release_speed']}
    for out in outcomes:
        lane = out['lane']
        tick, speed = releases.get(lane, (None, None))
        out.update(native_release_tick=tick, native_latched_mps=speed, sole_speed_loss=lane in flips)
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
    result = dict(policy_version=block['policy_version'], actor_sha256=actor_sha, world_updates=updates,
                checkpoint_policy_version=checkpoint_version, checkpoint_actor_sha256=checkpoint_actor,
                world_sha256=world_sha, source_chunks=block['source_chunks'], native_first_contacts=len(records),
                scope='Three recorded commands starting one tick before native first contact. Predicted body, ball and feet roll freely. Native scoring latches the last touch velocity. Common-boundary numbers use one step from the native first-contact start.',
                all_windows=summarize(outcomes), string_first=summarize([r for r in outcomes if r['online_first_tick_string']]),
                sole_speed_losses=summarize([r for r in outcomes if r['sole_speed_loss']]), cases=outcomes)

    if baseline is not None:
        holdout = set(json.loads((arm / 'train/config.json').read_text())['wm_holdout_lanes'])
        result['paired_refresh'] = compare_refresh(baseline, outcomes, holdout)
    return result


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('arm', 'checkpoint', 'block', 'flights', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--baseline-audit', type=Path, help='Replay exactly the same commands with a later guarded WM')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(args.arm, args.checkpoint, json.loads(args.block.read_text()), json.loads(args.flights.read_text()),
                   baseline=json.loads(args.baseline_audit.read_text()) if args.baseline_audit else None)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print({key: result[key] for key in ('all_windows', 'string_first', 'sole_speed_losses')})
