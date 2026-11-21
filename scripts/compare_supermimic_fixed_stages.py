#!/usr/bin/env python3
"""Attribute policy changes using complete native runs at identical initial states."""
import argparse
import json
import numpy as np
from statistics import median
from pathlib import Path

from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation, paired_flips, verify_baseline
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def read(folder, key):
    return json.loads((folder/(key+'.json')).read_text())


def load_run(folder):
    command = read(folder, 'startup')['argv']
    if command.count('--seed') != 1:
        raise ValueError('Fixed comparison requires an explicit seed')
    seed = int(command[command.index('--seed')+1])
    completed = read(folder, 'completed')
    metrics = read(folder, 'metrics')
    rows = read(folder, 'episodes_summary')
    if (completed['control_ticks'] != 800 or completed['episodes'] != 128
            or not metrics['actor_parameters_unchanged'] or metrics['num_envs'] != 128
            or any(metrics['optimizer_updates'].values()) or len(rows) != 128):
        raise ValueError('Incomplete frozen 128-case native evaluation')
    identity = dict(task=read(folder,'task_contract'), physics=read(folder,'physics'), seed=seed,
                    initials=read(folder,'initial_states'), execution=read(folder,'policy_execution')['contract'],
                    scoring=metrics['scoring_schema'], assignments=sorted((r['case_id'],r['env_id']) for r in rows))
    if (len({r['case_id'] for r in rows}) != 128 or {r['env_id'] for r in rows} != set(range(128))
            or metrics['policy_execution'] != identity['execution'] or metrics['task_contract'] != identity['task']):
        raise ValueError('Inconsistent native evaluation metadata')
    return rows, identity, metrics['actor_state_sha256']


def load_training_baseline(path, actor_hash, execution, task):
    """Accept the per-row training export or the complete frozen evaluator bundle."""
    if path.name == 'episodes_summary.json' and (path.parent/'completed.json').is_file():
        rows, identity, recorded_actor = load_run(path.parent)
        if recorded_actor != actor_hash or identity['execution'] != execution or identity['task'] != task:
            raise ValueError('Frozen evaluation differs from the loaded actor, execution or task')
        summary = summarize_native_episodes(rows)
        metrics = read(path.parent, 'metrics')['native_episode_metrics']
        for key in ('completed_episodes','qualified_airborne_string_contact','ball_side_complete_except_recovery',
                    'complete5','conditional_recovery_count','native_episode_return_mean'):
            if summary[key] != metrics[key]:
                raise ValueError(f'Baseline episode scores disagree with the frozen evaluation for {key}')
        rows = [dict({'actor_sha256':recorded_actor,'policy_execution':identity['execution']}, **row) for row in rows]
    else:
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get('records',rows.get('episodes'))
    if not isinstance(rows,list):
        raise ValueError('Expected a list of paired baseline episode records')
    verify_baseline(rows,actor_hash,execution)
    return rows


def restore_training_comparison(path, current_actor, execution, task, *, continuation=None):
    """Restore the zero-update comparator without loading it into the live policy."""
    import torch
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest

    actor = current_actor
    context = None
    saved_path = continuation['config'].get('paired_evaluation_baseline') if continuation else None
    if saved_path:
        context = continuation.get('paired_evaluation_context')
        if context is not None:
            if digest(path) != context['baseline_sha256']:
                raise ValueError('Paired baseline scores changed during the schedule')
            actor = context['baseline_actor']
        else:
            if Path(saved_path).resolve() != path.resolve() and digest(saved_path) != digest(path):
                raise ValueError('Different baseline supplied for an ongoing paired comparison')
            bundle = path.name == 'episodes_summary.json' and (path.parent/'completed.json').is_file()
            if bundle:
                _, _, expected = load_run(path.parent)
            else:
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    payload = payload.get('records', payload.get('episodes'))
                if not isinstance(payload,list) or not payload:
                    raise ValueError('Missing paired baseline rows')
                expected = payload[0].get('actor_sha256')
            # Legacy checkpoints may still hold the baseline as a block or native reference.
            for candidate in (current_actor, *continuation.get('actor_references', {}).values()):
                if tensor_digest(candidate.items()) == expected:
                    actor = candidate
                    break
            else:
                if not bundle:
                    raise ValueError('Legacy checkpoint has no matching baseline actor reference')
                argv = read(path.parent, 'startup')['argv']
                if argv.count('--checkpoint') != 1:
                    raise ValueError('Legacy paired continuation lacks the original evaluated actor')
                source = Path(argv[argv.index('--checkpoint')+1])
                if digest(source) != read(path.parent, 'metrics')['evaluated_checkpoint_sha256']:
                    raise ValueError('Original paired evaluation checkpoint changed')
                import zipfile
                actor = torch.load(source, map_location='cpu', weights_only=False, mmap=zipfile.is_zipfile(source))['actor']
    sha = tensor_digest(actor.items())
    rows = load_training_baseline(path, sha, execution, task)
    snapshots = {'pre-update': rows}
    actors = {'pre-update': {k: v.detach().cpu().clone() for k, v in actor.items()}}
    if context is not None and context.get('previous') is not None:
        previous = context['previous']
        stage = previous['stage']
        previous_actor = previous['actor']
        if stage == 'pre-update':
            raise ValueError('Invalid previous paired refresh stage')
        verify_baseline(previous['records'], tensor_digest(previous_actor.items()), execution)
        if {r['case_id'] for r in rows} != {r['case_id'] for r in previous['records']}:
            raise ValueError('Paired refresh changed the fixed cases')
        snapshots[stage] = previous['records']
        actors[stage] = previous_actor
    return snapshots, actors, sha


def training_comparison_context(snapshots, actors, baseline_sha256):
    if not snapshots:
        return None
    stage = next(reversed(snapshots))
    previous = (dict(stage=stage, records=snapshots[stage], actor=actors[stage])
                if stage != 'pre-update' else None)
    return dict(baseline_sha256=baseline_sha256, baseline_actor=actors['pre-update'], previous=previous)


def paired_task_outcomes(result):
    """Use one matched before/after difference per case for every estimate."""
    rows = result['case_deltas']
    if len(rows) != 128 or len({row['case_id'] for row in rows}) != 128:
        raise ValueError('Expected 128 independent matched case IDs')
    changes = np.array([row['return_delta'] for row in rows], dtype=np.float64)
    if not np.isfinite(changes).all():
        raise ValueError('Nonfinite paired native return difference')
    seed = 882072
    means = np.random.default_rng(seed).choice(changes, size=(20000, len(rows))).mean(axis=1)
    return dict(binary={key: paired_flips(rows, key) for key in ('contact', 'ball_side', 'complete5')},
                native_return=dict(mean=float(changes.mean()), median=float(np.median(changes)),
                                   descriptive_paired_case_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                                   seed=seed, resamples=len(means)))


def first_episode_failure_flips(before, after, physics_dt):
    """Track body-ground changes and qualified-string/frame-first switches on matched cases."""
    if before.keys() != after.keys():
        raise ValueError('Failure comparison requires identical case IDs')

    def frame_first(row):
        return (row['had_contact'] and row['frame_hit']
                and (not row['string_hit'] or row['first_string_time_s'] > row['contact_time_s'] + physics_dt / 2))

    def switched(string_case, frame_case):
        return (string_case['qualified_airborne_string_contact']
                and not frame_case['qualified_airborne_string_contact'] and frame_first(frame_case))

    return dict(new_body_ground=[k for k in before if after[k]['body_ground_seen'] and not before[k]['body_ground_seen']],
                avoided_body_ground=[k for k in before if before[k]['body_ground_seen'] and not after[k]['body_ground_seen']],
                qualified_to_frame_first=[k for k in before if switched(before[k], after[k])],
                frame_first_to_qualified=[k for k in before if switched(after[k], before[k])])


def discordant_landing_evidence(before, after, paired):
    """Join first-episode native flight measurements to the officially scored recovery flips."""
    paths = [folder / 'episodes.jsonl' for folder in (before, after)]
    if not all(path.is_file() for path in paths):
        return dict(available=False, reason='Missing per-episode native flight records')
    stages = []
    for path in paths:
        records = {}
        for line in path.read_text().splitlines():
            episode = json.loads(line)
            if episode.get('episode_index') != 0:
                continue
            key = episode['case_id']
            if key in records:
                raise ValueError('Duplicate first episode in native landing diagnostics')
            records[key] = episode
        stages.append(records)
    if any(set(rows) != set(paired['case_ids']) for rows in stages):
        raise ValueError('Raw landing measurements and official evaluation have different cases')
    result = dict(available=True, raw_sha256=[digest(p) for p in paths])
    required = {'body_ground_seen', 'had_contact', 'frame_hit', 'string_hit', 'first_string_time_s',
                'contact_time_s', 'qualified_airborne_string_contact'}
    if all(required.issubset(row) for stage in stages for row in stage.values()):
        result['failure_flips'] = first_episode_failure_flips(*stages, read(before, 'task_contract')['physics_dt_s'])
    fields = ('body_landing', 'stable_recovery', 'qualified_contact_landing', 'recovery_s',
              'terminal_root_height_m', 'maximum_body_ground_force_n', 'contact_flight_landing_s', 'rally_contact_time_s')
    for category in ('improved_cases', 'regressed_cases'):
        cases = []
        for key in paired['paired_recovery'][category]:
            rows = [stage[key].get('contact_flight_recovery', {}) for stage in stages]
            if not all(r.get('bound') and all(r.get(k) is not None for k in fields) for r in rows):
                cases.append(dict(case_id=key, available=False))
                continue
            cases.append(dict(case_id=key, available=True,
                              before={k: rows[0][k] for k in fields}, after={k: rows[1][k] for k in fields}))
        known = [r for r in cases if r['available']]
        summary = {}
        if known:
            summary = dict(new_body_landings=sum(r['after']['body_landing'] and not r['before']['body_landing'] for r in known),
                           avoided_body_landings=sum(r['before']['body_landing'] and not r['after']['body_landing'] for r in known),
                           max_touchdown_time_change_s=max(abs(r['after']['contact_flight_landing_s']-r['before']['contact_flight_landing_s']) for r in known),
                           stable_time_median_s={stage:median(r[stage]['recovery_s'] for r in known) for stage in ('before','after')})
        result[category] = dict(cases=cases, observed=len(known), **summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('before', 'after', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--expected-before', required=True)
    parser.add_argument('--expected-after', required=True)
    a = parser.parse_args()
    pre, first, old = load_run(a.before)
    post, second, new = load_run(a.after)
    mismatched = [k for k in first if first[k] != second[k]]
    if mismatched or (old,new) != (a.expected_before,a.expected_after):
        raise ValueError(f'Incompatible frozen comparisons: {mismatched}, actual actors {old} and {new}')
    result = paired_evaluation(pre,post,stage=a.stage)
    result['paired_task_outcomes'] = paired_task_outcomes(result)
    result['landing_diagnostics'] = discordant_landing_evidence(a.before, a.after, result)
    result.update(native_sources=[dict(actor_sha256=h, seed=first['seed'], path=str(folder),
                                       metrics_sha256=digest(folder/'metrics.json'),
                                       episodes_sha256=digest(folder/'episodes_summary.json'))
                                  for folder,h in ((a.before,old),(a.after,new))],
                  identical_initial_states=True)
    with a.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
