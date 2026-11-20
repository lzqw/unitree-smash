#!/usr/bin/env python3
"""Attribute policy changes using complete native runs at identical initial states."""
import argparse
import json
from pathlib import Path

from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation
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
    result.update(native_sources=[dict(actor_sha256=h, seed=first['seed'], path=str(folder),
                                       metrics_sha256=digest(folder/'metrics.json'),
                                       episodes_sha256=digest(folder/'episodes_summary.json'))
                                  for folder,h in ((a.before,old),(a.after,new))],
                  identical_initial_states=True)
    with a.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
