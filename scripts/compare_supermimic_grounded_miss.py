"""Compare a causal grounded-miss Recovery gate on a previously scored frozen Actor."""
import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.compare_supermimic_fixed_stages import load_run, paired_task_outcomes, read
from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation
from quadsmash.rl.supermimic_policy_execution_v1 import POSTHIT, POSTHIT_OR_GROUNDED_MISS, contract


def compare(baseline, diagnostic):
    before, base_id, base_actor = load_run(baseline)
    after, diag_id, diag_actor = load_run(diagnostic)
    b, d = (read(path, 'metrics') for path in (baseline, diagnostic))
    if (base_id.pop('execution') != contract(POSTHIT)
            or diag_id.pop('execution') != contract(POSTHIT_OR_GROUNDED_MISS)
            or d.get('checkpoint_policy_execution') != contract(POSTHIT)):
        raise ValueError('Unexpected grounded-miss comparison contracts')
    if (base_id != diag_id or base_actor != diag_actor
            or b['evaluated_checkpoint_sha256'] != d['evaluated_checkpoint_sha256']
            or b['numerical_failures'] or d['numerical_failures']):
        raise ValueError('Gate comparison requires identical frozen inputs, weights, and valid native scores')
    paired = paired_evaluation(before, after, stage='frozen-grounded-miss')
    prior = {row['env_id']: row for row in before}
    current = {row['env_id']: row for row in after}
    baseline_events = {r['env_id']: r for r in map(json.loads, (baseline/'episodes.jsonl').read_text().splitlines())}
    if baseline_events.keys() != prior.keys() or any(e['episode_index'] != 0 or e['case_id'] != prior[i]['case_id']
                                                     for i, e in baseline_events.items()):
        raise ValueError('Incomplete first-episode baseline events')
    ticks, onset = d['first_episode_extra_gate_ticks'], d['first_episode_extra_gate_onset']
    if len(ticks) != 128 or len(onset) != 128 or any(
            not (0 <= n <= 800 and ((n == 0 and t == -1) or (n > 0 and 0 <= t <= 800 - n)))
            for n, t in zip(ticks, onset)):
        raise ValueError('Invalid first-episode gate measurements')
    rows = []
    for i, a in sorted(prior.items()):
        c, e = current[i], baseline_events[i]
        rows.append(dict(case_id=a['case_id'], lane=i, gate_ticks=ticks[i], first_gate_tick=onset[i],
                         baseline_grounded_miss=bool(e['launched'] and not e['in_play'] and e['ground_seen']
                                                     and not (e['had_contact'] or e['frame_hit'] or e['string_hit'])),
                         return_delta=c['native_episode_return']-a['native_episode_return'],
                         body_ground_before=a['body_ground_seen'], body_ground_after=c['body_ground_seen'],
                         physical_recovery_before=a['physical_recovery_success'],
                         physical_recovery_after=c['physical_recovery_success']))

    def subset(selected):
        return dict(cases=len(selected), mean_return_delta=mean(r['return_delta'] for r in selected) if selected else None,
                    max_abs_return_delta=max((abs(r['return_delta']) for r in selected), default=None),
                    avoided_body_ground=[r['case_id'] for r in selected if r['body_ground_before'] and not r['body_ground_after']],
                    new_body_ground=[r['case_id'] for r in selected if r['body_ground_after'] and not r['body_ground_before']],
                    physical_recovery_net=sum(int(r['physical_recovery_after'])-int(r['physical_recovery_before']) for r in selected))

    fields = ('native_episode_return_mean', 'body_ground_seen', 'qualified_airborne_string_contact',
              'conditional_recovery_count', 'ball_side_complete_except_recovery', 'complete5')
    return dict(scope='TRAIN128, zero weight updates, only the grounded-miss execution gate changed',
                actor_sha256=base_actor, checkpoint_sha256=b['evaluated_checkpoint_sha256'], seed=base_id['seed'],
                before={k:b[k] for k in fields}, after={k:d[k] for k in fields},
                paired=paired_task_outcomes(paired), all_cases=subset(rows),
                baseline_grounded_misses=subset([r for r in rows if r['baseline_grounded_miss']]),
                gate_active=subset([r for r in rows if r['gate_ticks']]), gate_inactive=subset([r for r in rows if not r['gate_ticks']]),
                case_deltas=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'diagnostic', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.diagnostic)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
