"""Publish measured research-arm curves, with inherited interactions removed."""
import argparse
import json
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def arm_origin(registration):
    import torch
    from run_supermimic_research_scale import digest
    cmd = registration['command']
    path = Path(registration.get('initial_checkpoint', cmd[cmd.index('--resume-training') + 1]))
    if digest(path) != registration['initial']['checkpoint_sha256']:
        raise ValueError('Initial training checkpoint changed')
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    return ck['global_counters']['accepted'], ck['native_ticks']


def axes(row, origin, lanes):
    retained = row['accepted_id'] - origin[0]
    steps = (row['native_ticks'] - origin[1]) * lanes
    if retained < 0 or steps < 0 or row['native_transitions'] != row['native_ticks'] * lanes:
        raise ValueError('Invalid research-arm counters')
    return dict(retained_updates=retained, new_native_env_steps=steps,
                training_wall_seconds=int(row['research_elapsed_s']))


def scalars(row):
    head = row['block']
    metrics = {f'{head}/{key}': row[key] for key in (
        'objective_before', 'objective_after', 'final_command_max_change', 'task_gradient_norm',
        'tracking_gradient_norm', 'effective_gate_lanes', 'critic_loss', 'critic_loss_before_proposal', 'contact_aux_weight',
        'native_collect_seconds', 'bptt_seconds', 'world_update_seconds') if key in row}
    sampling = row.get('actor_start_sampling', {})
    for family, count in sampling.get('actual_by_family', {}).items():
        metrics[f'{head}/start_lanes/{family}'] = count
    coverage = row.get('reference_coverage', {})
    if 'predicted_first_margin_sum_m' in coverage:
        n = sum(coverage[k] for k in ('predicted_first_string_lanes', 'predicted_first_frame_lanes'))
        if n:
            metrics['approach/predicted_first_margin_mm'] = 1000 * coverage['predicted_first_margin_sum_m'] / n
            metrics['approach/predicted_first_frame_fraction'] = coverage['predicted_first_frame_lanes'] / n
    if head == 'approach' and row.get('contact_aux_weight', 0) > 0:
        for key in ('outgoing_impact_lanes', 'task_tracking_cosine'):
            if key in row:
                metrics['approach/'+key] = row[key]
        if all(key in row for key in ('task_gradient_norm','tracking_gradient_norm','toi_racket_weight')):
            outgoing_norm = row['contact_aux_weight'] * row['task_gradient_norm']
            tracking_norm = row['toi_racket_weight'] * row['tracking_gradient_norm']
            metrics['approach/weighted_outgoing_gradient_norm'] = outgoing_norm
            if tracking_norm > 0:
                metrics['approach/outgoing_to_tracking_gradient_ratio'] = outgoing_norm / tracking_norm
    metrics[head+'/accepted'] = int(row['update_accepted'])
    metrics['gpu/peak_allocated_GiB'] = row['peak_gpu_bytes'] / 2**30
    guard = row.get('fixed_native_forward')
    if guard:
        for group in ('body', 'flight', 'contact'):
            checks = guard[group]
            for item in checks.get('comparisons', checks.get('checks', [])):
                if item.get('candidate') is not None:
                    path = '/'.join(map(str, (group, item.get('horizon', 'event'), item['metric'])))
                    metrics['world/'+path] = item['candidate']
    reports = row.get('world_update_reports', [])
    fits = [r['contact_supervision'] for r in reports if r.get('contact_supervision')]
    if fits:
        metrics['world/contact_supervision_batches'] = len(fits)
        metrics['world/contact_optimizer_steps'] = sum(int(r['updated']) for r in fits)
        metrics['world/sampled_contact_events'] = sum(r['measured_events'] for r in fits)
    latest = [r['sampling']['latest_policy_windows'] for r in reports if 'latest_policy_windows' in r.get('sampling', {})]
    if latest:
        for key in ('requested', 'sampled'):
            metrics['world/latest_policy_windows_' + key] = sum(entry[key] for entry in latest)
    if reports and all('online_fit_by_policy' in r.get('sampling', {}) for r in reports):
        # Collection and all WM updates precede this block's first Actor proposal.
        behavior = str(row['accepted_id'] - int(row['update_accepted']))
        coverage = [r['sampling']['online_fit_by_policy'] for r in reports]
        metrics['world/online_transition_uses'] = sum(
            group['transition_uses'] for batch in coverage for group in batch.values())
        for key in ('transition_uses', 'racket_edge_uses'):
            metrics['world/collection_policy_' + key] = sum(batch.get(behavior, {}).get(key, 0) for batch in coverage)
        metrics['world/collection_policy_contact_fit_edge_uses'] = sum(
            batch.get(behavior, {}).get('racket_edge_uses', 0) for batch, report in zip(coverage, reports)
            if report.get('contact_supervision', {}).get('updated'))
    return metrics


def read_new(path, offset):
    if not path.exists():
        return [], offset
    rows = []
    with path.open('rb') as stream:
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b'\n'):
                break
            rows.append(json.loads(line))
            offset += len(line)
    return rows, offset


class FirstEpisodeReturns:
    """Match the first completed online episode per initial lane to its frozen TRAIN128 case."""
    def __init__(self, arm, registration, manifest):
        from scripts.compare_supermimic_fixed_stages import load_run
        path = Path(registration['baseline_path'])
        rows, identity, actor = load_run(path.parent)
        if (actor != registration['initial']['actor_sha256'] or identity['seed'] != manifest['seed']
                or identity['task'] != manifest['task_contract'] or identity['execution'] != manifest['policy_execution']):
            raise ValueError('Online return comparison has incompatible baseline identity')
        self.baseline = {(row['case_id'], row['env_id']): row for row in rows}
        self.path = arm/'train/native_episodes.jsonl'
        self.offset = 0
        self.seen = set()
        self.online_total = self.initial_total = 0.
        self.duration_pairs = self.duration_delta = 0
        self.early_after_timeout = self.timeout_after_early = 0
        self.flips = dict(ball_side_complete_except_recovery=0, physical_recovery_success=0, complete5=0)

    def update(self, writer):
        rows, self.offset = read_new(self.path, self.offset)
        for row in rows:
            if row.get('episode_index') != 0 or row.get('scope') != 'mixed-policy monitoring, not frozen score':
                continue
            key = (row['case_id'], row['env_id'])
            if key in self.seen:
                continue
            baseline = self.baseline[key]
            if row['scoring_schema'] != baseline['scoring_schema']:
                raise ValueError('Different native scoring rules for the same case')
            self.seen.add(key)
            n = len(self.seen)
            self.online_total += row['native_episode_return']
            self.initial_total += baseline['native_episode_return']
            for metric in self.flips:
                self.flips[metric] += int(row[metric]) - int(baseline[metric])
            metrics = dict(online_mean=self.online_total/n, initial_same_cases_mean=self.initial_total/n,
                           mean_difference=(self.online_total-self.initial_total)/n,
                           latest_case_difference=row['native_episode_return']-baseline['native_episode_return'],
                           **{name+'_net_cases': count for name,count in self.flips.items()})
            if all(r.get('native_episode_length_steps') is not None and
                   isinstance(r.get('episode_timeout'), bool) for r in (row, baseline)):
                self.duration_pairs += 1
                self.duration_delta += row['native_episode_length_steps'] - baseline['native_episode_length_steps']
                self.early_after_timeout += baseline['episode_timeout'] and not row['episode_timeout']
                self.timeout_after_early += row['episode_timeout'] and not baseline['episode_timeout']
                metrics.update(duration_pairs=self.duration_pairs,
                               mean_episode_length_delta_ticks=self.duration_delta/self.duration_pairs,
                               baseline_timeout_to_online_early=self.early_after_timeout,
                               baseline_early_to_online_timeout=self.timeout_after_early)
            values = [Summary.Value(tag='first_completed_mixed_cases/'+k,simple_value=v) for k,v in metrics.items()]
            writer.add_event(Event(wall_time=time.time(),step=n,summary=Summary(value=values)))
        if rows:
            writer.flush()


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--tensorboard', type=Path, required=True)
    parser.add_argument('--follow-pid', type=int)
    parser.add_argument('--online-returns-only', action='store_true')
    args = parser.parse_args()
    owner = psutil.Process(args.follow_pid) if args.follow_pid else None
    arm = args.arm_dir.resolve()
    reg = json.loads((arm/'registration.json').read_text())
    manifest_path = arm/'train/native_transitions/manifest.json'
    while owner and not manifest_path.exists():
        if not owner.is_running() or owner.status() == psutil.STATUS_ZOMBIE:
            return
        time.sleep(5)
    manifest = json.loads(manifest_path.read_text())
    reference_path = Path(reg['baseline_path']) if reg.get('baseline_path') else None
    matched = (FirstEpisodeReturns(arm,reg,manifest) if reference_path is not None
               and reference_path.name == 'episodes_summary.json' and (reference_path.parent/'completed.json').exists()
               else None)
    if args.online_returns_only and matched is None:
        raise ValueError('A complete registered frozen baseline is required for online return comparison')
    origin = None if args.online_returns_only else arm_origin(reg)
    from scripts.report_supermimic_native_contact_counts import NativeContactCounts
    contacts = None if args.online_returns_only else NativeContactCounts(arm)
    writer = EventFileWriter(str(args.tensorboard))
    offset = 0
    heads = dict(reg.get('previous_heads', {'recovery': 0, 'approach': 0}))
    attempts = reg.get('previous_proposals', 0)
    try:
        while True:
            live = owner and owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE
            if matched is not None:
                matched.update(writer)
            rows = []
            if contacts is not None:
                contacts.update(writer)
                rows, offset = read_new(arm/'train/training_history.jsonl', offset)
            for row in rows:
                pos = axes(row, origin, manifest['num_envs'])
                metrics = scalars(row)
                heads[row['block']] += int(row['update_accepted'])
                attempts += 1
                if sum(heads.values()) != pos['retained_updates']:
                    raise ValueError('Recorded proposals do not account for every retained step')
                if row['update_accepted'] and pos['retained_updates'] == 16:
                    from scripts.report_supermimic_first_pair_budget import write_budget
                    budget = write_budget(arm)
                    if budget is not None:
                        metrics['timing/first_pair_wall_seconds'] = budget['first_pair_wall_seconds']
                        metrics['timing/estimated_remaining_hours_after_first_pair'] = budget['estimated_remaining_hours_after_first_pair']
                for axis, step in pos.items():
                    values = [Summary.Value(tag=axis+'/'+k, simple_value=v) for k,v in metrics.items()]
                    writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))
            if rows:
                writer.flush()
                status = dict(arm=reg['arm'], **pos, accepted=dict(heads), proposals=attempts,
                              inherited_native_ticks=origin[1], inherited_updates=origin[0],
                              note='Training wall time starts after initial replay reconstruction.')
                tmp = arm/'learning_curve.json.tmp'
                tmp.write_text(json.dumps(status, indent=2)+'\n')
                tmp.replace(arm/'learning_curve.json')
            if not live:
                break
            time.sleep(10)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
