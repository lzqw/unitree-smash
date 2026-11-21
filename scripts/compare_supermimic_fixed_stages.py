#!/usr/bin/env python3
"""Attribute policy changes using complete native runs at identical initial states."""
import argparse
import json
from statistics import median
from pathlib import Path

from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation, verify_baseline
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
