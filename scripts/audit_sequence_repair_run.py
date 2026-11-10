#!/usr/bin/env python3
"""Read-only audit of one completed native run against saved TRAIN128 evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_run(root):
    metrics = read(root / 'metrics.json')
    completed = read(root / 'completed.json')
    if metrics['episodes'] != 128 or metrics['actual_control_env_steps'] != 102400:
        raise ValueError('Full original128 × 800 control ticks required')
    if metrics['distinct_TRAIN_indices'] != list(range(128)):
        raise ValueError('Original lane mapping required')
    if (root / 'error.txt').exists():
        raise ValueError('Run contains an error; cannot treat as completed evaluation')
    rows = [json.loads(line) for line in (root / 'pure_control_diagnostics.jsonl').open()]
    return dict(root=root, metrics=metrics, completed=completed, rows=rows,
                episodes={x['source_training_index']: x for x in read(root / 'episodes.json')},
                outcomes={x['training_index']: x for x in metrics['complete5_case_outcomes']})


def diagnostics(run, fixed):
    rows = run['rows']
    first = {}
    for row in rows:
        first.setdefault(row['lane'], row)
    # Frame timestamps are original simulation timestamps, at 16000 Hz.
    collision = {}
    for event in read(run['root'] / 'first_nonfoot_contact_details.json'):
        for lane in event['first_event_env_ids']:
            collision.setdefault(lane, event['physics_frame'] / 16000.)
    paired = ground_audit(run, first)
    counts = Counter()
    entry = Counter()
    active_repair = defaultdict(bool)
    per = defaultdict(lambda: dict(repair_attempts=0, repair_successes=0,
                                  repaired_execution_ticks=0, pre_collision_fallback_ticks=0,
                                  longest_pre_collision_fallback_ticks=0))
    streak = defaultdict(int)
    paired_streak = defaultdict(int)
    failed_steps = Counter()
    time_s = []
    for row in rows:
        lane = row['lane']; d = row['diagnostic']; plan = d.get('plan', {})
        counts['owned_ticks'] += 1
        time_s.append(d.get('wall_seconds', 0.))
        counts['fallback_ticks'] += int(not d['solver_accepted'])
        if d.get('replanned'):
            counts['replans'] += 1
            active_repair[lane] = bool(plan.get('sequence_repaired'))
            attempted = bool(plan.get('repair_attempted') or plan.get('sequence_repaired'))
            counts['repair_attempts'] += attempted
            counts['repair_successes'] += active_repair[lane]
            per[lane]['repair_attempts'] += attempted
            per[lane]['repair_successes'] += active_repair[lane]
            original = plan.get('original_failure', plan)
            rejection = original.get('rejection')
            if rejection:
                if rejection.startswith('long_horizon'):
                    stage = 'first_prediction' if original.get('failed_step') == 0 else 'future_prediction'
                    failed_steps[str(original.get('failed_step'))] += 1
                elif 'Independent interval' in rejection:
                    stage = 'independent_interval_check'
                else:
                    stage = 'single_step_qp_or_other'
                entry[stage + ':' + rejection] += 1
        if d['solver_accepted'] and active_repair[lane]:
            counts['repaired_execution_ticks'] += 1
            per[lane]['repaired_execution_ticks'] += 1
        if row['tick'] * .01 < collision.get(lane, 8.):
            streak[lane] = streak[lane] + 1 if not d['solver_accepted'] else 0
            per[lane]['pre_collision_fallback_ticks'] += int(not d['solver_accepted'])
            per[lane]['longest_pre_collision_fallback_ticks'] = max(
                per[lane]['longest_pre_collision_fallback_ticks'], streak[lane])
        cutoff = paired[lane]['first_post_takeover_nonfoot_force_s']
        if row['tick'] * .01 < (8. if cutoff is None else cutoff):
            paired_streak[lane] = paired_streak[lane] + 1 if not d['solver_accepted'] else 0
            per[lane]['pre_paired_nonfoot_fallback_ticks'] = per[lane].get('pre_paired_nonfoot_fallback_ticks', 0) + int(not d['solver_accepted'])
            per[lane]['longest_pre_paired_nonfoot_fallback_ticks'] = max(
                per[lane].get('longest_pre_paired_nonfoot_fallback_ticks', 0), paired_streak[lane])
    with np.load(run['root'] / 'pure_control_commands.npz') as z:
        commands = {k: z[k] for k in z.files}
    for lane in fixed:
        info = per[lane]
        info['first_nonfoot_event_s'] = collision.get(lane)
        info['first_nonfoot_event_scope'] = 'historical first-event logger; use paired audit separately for head-inclusive contacts'
        info['first_paired_nonfoot_force_s'] = paired[lane]['first_post_takeover_nonfoot_force_s']
        info['release_s'] = first.get(lane, {}).get('phase', {}).get('release_time_s')
        e = run['episodes'][lane]
        info['contact_flight_recovery'] = e['contact_flight_recovery']
        info['conditional_recovery'] = run['outcomes'][lane]['recovery']
        cutoff = collision.get(lane, 8.)
        ids = np.flatnonzero((commands['lane'] == lane) & (commands['time_s'] < cutoff))[-10:]
        if len(ids):
            info['last_10_control_samples_before_first_event'] = dict(
                times_s=commands['time_s'][ids], root_height_m=commands['q'][ids, 2],
                angular_speed_rad_s=np.linalg.norm(commands['v'][ids, 3:6], axis=1),
                measured_foot_fz_n=commands['ground_force'][ids, :, 2])
    return dict(counts=dict(counts), failure_entries=dict(entry),
                original_failed_nodes=dict(failed_steps),
                command_wall_s=dict(sum=float(sum(time_s)), max=float(max(time_s)),
                                    median=float(np.median(time_s)), p95=float(np.quantile(time_s, .95))),
                cases=dict(per)), first, commands


def prefix_check(base, candidate, first_a, first_b, cmd_a, cmd_b):
    per = {}
    with np.load(base['root'] / 'sequence_dispatch.npz') as a, np.load(candidate['root'] / 'sequence_dispatch.npz') as b:
        dispatch_a = a['final']; dispatch_b = b['final']
        for lane in range(128):
            ta = first_a.get(lane, {}).get('tick', 800)
            tb = first_b.get(lane, {}).get('tick', 800)
            stop = min(ta, tb)
            per[lane] = dict(base_takeover_tick=ta, candidate_takeover_tick=tb,
                             takeover_tick_equal=ta == tb,
                             prefix_max_command_delta=float(np.max(abs(dispatch_a[:stop, lane] - dispatch_b[:stop, lane]), initial=0.)))
            for name in ['contact_time_s', 'release_time_s', 'first_string_time_s', 'outgoing_velocity_mps']:
                x = np.asarray(base['episodes'][lane][name]); y = np.asarray(candidate['episodes'][lane][name])
                per[lane][name + '_equal'] = bool(np.array_equal(x, y, equal_nan=True))
            ia = np.flatnonzero(cmd_a['lane'] == lane); ib = np.flatnonzero(cmd_b['lane'] == lane)
            if len(ia) and len(ib):
                for field in ['q', 'v', 'previous']:
                    per[lane]['takeover_' + field + '_max_delta'] = float(np.max(abs(cmd_a[field][ia[0]] - cmd_b[field][ib[0]])))
    return per


def ground_audit(run, first):
    body_names = read(run['root'] / 'pure_control_ground_forces.json')['body_names']
    feet = {'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'}
    with np.load(run['root'] / 'pure_control_ground_forces.npz') as z:
        peaks = z['interval_peak_normal_n']
        frames = z['interval_first_frame_over1n']
    with np.load(run['root'] / 'sequence_dispatch.npz') as z:
        first_episode = z['first_episode']
    result = {}
    for lane, row in first.items():
        mask = (np.arange(len(peaks)) >= row['tick']) & first_episode[:, lane]
        sub = peaks[mask, lane]
        nonfeet = [i for i, name in enumerate(body_names) if name not in feet]
        touched = [i for i in nonfeet if np.any(sub[:, i] > 1.)]
        hits = frames[mask, lane][:, nonfeet]
        valid_hits = hits[hits >= row['tick'] * 160]
        result[lane] = dict(
            scope='post_takeover up to episode end; diagnostic >1N paired normal force; original metric unchanged',
            nonfoot_bodies_over1n=[body_names[i] for i in touched],
            head_contact=any('head' in body_names[i].lower() for i in touched),
            first_post_takeover_nonfoot_force_s=float(valid_hits.min() / 16000.) if len(valid_hits) else None,
            peak_nonfoot_normal_n=float(np.max(sub[:, nonfeet], initial=0.)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a = load_run(args.baseline); b = load_run(args.candidate)
    fixed = {i for i, row in a['outcomes'].items() if row['air']}
    old = {i for i in fixed if a['outcomes'][i]['recovery']}
    new = {i for i in fixed if b['outcomes'][i]['recovery']}
    if len(fixed) != 74 or len(old) != 52:
        raise ValueError('Expected frozen 9cb7983 52/74 baseline')
    ad, af, ac = diagnostics(a, fixed); bd, bf, bc = diagnostics(b, fixed)
    out = dict(
        fixed74=sorted(fixed), baseline_recovery=len(old), candidate_recovery=len(new),
        retained=sorted(old & new), lost=sorted(old-new), rescued=sorted(new-old), net=len(new)-len(old),
        candidate_scope='Executed candidate has known first-command and mutable-history contract defects; not a valid single-factor result',
        native_metrics={name:{k:r['metrics'][k] for k in ['complete5','airborne_contacts','conditional_recovery_count',
                        'crossed_net','landed_target','numerical_failures','actual_control_env_steps']}
                        for name,r in [('baseline',a),('candidate',b)]},
        diagnostics={'baseline':ad,'candidate':bd},
        prefix=prefix_check(a,b,af,bf,ac,bc),
        ground_audit={'baseline':ground_audit(a,af),'candidate':ground_audit(b,bf)},
        inputs={'baseline':read(args.baseline/'INPUTS.json'),'candidate':read(args.candidate/'INPUTS.json')},
        source_evidence={name:{f:sha(r['root']/f) for f in ['metrics.json','episodes.json','completed.json','pure_control_diagnostics.jsonl']}
                         for name,r in [('baseline',a),('candidate',b)]},
        limitations=['No dense predicted-state trace; cannot quantify prediction/actual error at each repaired node.',
                     'Native aggregate solver_accepted includes cached-plan execution; explicit repair counts are reported separately.',
                     'Conditional recovery includes rule faults; stable physical recovery is reported separately.',
                     'CPU contract tests are not physics success evidence.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(finite(out),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps({k:out[k] for k in ['baseline_recovery','candidate_recovery','retained','lost','rescued','net']}))


if __name__ == '__main__':
    main()
