"""Compare complete native stance trials without conflating local and task success.

Reports case membership, measured contact events, terminal sampled stability
and the first changed command/state. Does not claim that numerical differences
alone establish causality or that sampled stability is formal acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(preparation, support):
    plan = read(preparation/'plan.json')
    run = Path(plan['output'])
    if read(run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('A partial native run is not a performance comparison')
    audit = read(preparation/'native_audit.json')
    if not audit['passed_dispatch_integrity']:
        raise ValueError('Native final commands have not been verified')
    evidence = read(support)
    for filename,digest in evidence['source_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('Changed support evidence '+filename)
    if evidence['source_sha256'].get(str((preparation/'native_audit.json').resolve())) != sha(preparation/'native_audit.json'):
        raise ValueError('Support evidence describes another native run')
    with np.load(run/'pure_control_commands.npz') as data:
        arrays = {k:data[k].copy() for k in ['tick', 'lane', 'q', 'v', 'final']}
    records = [json.loads(line) for line in (run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(records) != len(arrays['tick']):
        raise ValueError('Misaligned commands and diagnostics')
    by_case = {}
    for i,row in enumerate(records):
        if (row['tick'], row['lane']) != (int(arrays['tick'][i]), int(arrays['lane'][i])):
            raise ValueError('Misaligned native control interval')
        case = by_case.setdefault(row['lane'], {})
        if row['tick'] in case:
            raise ValueError('Repeated native control interval')
        case[row['tick']] = dict(q=arrays['q'][i], v=arrays['v'][i], command=arrays['final'][i],
            stance=bool(row['diagnostic'].get('terminal_stance_active')),
            foot_soft_target=bool(row['diagnostic'].get('reference', {}).get('terminal_foot_soft_target')),
            source=row['diagnostic']['source'])
    return dict(run=run, configuration=read(preparation/'controller.json'), audit=audit,
        support=evidence, cases=by_case, support_cases={r['case']:r for r in evidence['cases']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['old-preparation', 'new-preparation', 'old-support', 'new-support', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh comparison output required')
    old = load(args.old_preparation, args.old_support)
    new = load(args.new_preparation, args.new_support)
    if old['configuration']['interface'] != new['configuration']['interface']:
        raise ValueError('Physical command interfaces differ')
    if sha(old['configuration']['model_reference_library']) != sha(new['configuration']['model_reference_library']):
        raise ValueError('Reference libraries differ; this is not the intended stance-only comparison')
    def success(data):
        return set(data['support']['summary']['clean_and_final_sampled_stable_08s_through_tick799'])
    old_ok, new_ok = success(old), success(new)
    results = []
    for case in sorted(set(old['cases']) | set(new['cases'])):
        left, right = old['cases'].get(case, {}), new['cases'].get(case, {})
        ticks = sorted(set(left) & set(right))
        stance_ticks = [t for t in ticks if left[t]['stance'] or right[t]['stance']]
        first_stance = min(stance_ticks, default=None)
        first_action, first_state = None, None
        before = dict(command_rad=0., q=0., v=0.)
        for tick in ticks:
            a, b = left[tick], right[tick]
            du = float(np.max(abs(a['command']-b['command'])))
            dq = float(np.max(abs(a['q']-b['q'])))
            dv = float(np.max(abs(a['v']-b['v'])))
            if first_action is None and du > 2e-6:
                first_action = dict(tick=tick, maximum_command_difference_rad=du,
                    old_source=a['source'], new_source=b['source'],
                    old_stance=a['stance'], new_stance=b['stance'])
            if first_state is None and (dq > 2e-6 or dv > 2e-5):
                first_state = dict(tick=tick, maximum_q_difference=dq, maximum_v_difference=dv)
            if first_stance is None or tick < first_stance:
                before['command_rad'] = max(before['command_rad'], du)
                before['q'] = max(before['q'], dq)
                before['v'] = max(before['v'], dv)
        def outcome(data):
            row = data['support_cases'].get(case)
            if row is None:
                return dict(control_owned=False)
            def event(name):
                r = row[name]
                return {k:r[k] for k in ['body', 'time_s', 'physics_frame']} if r else None
            return dict(control_owned=True, clean=row['no_recorded_nonfoot_ground_force'],
                final_stable_s=row['final_sampled_stable_s'], longest_stable_s=row['longest_sampled_stable_s'],
                last_control_tick=row['last_control_tick'],
                first_head_over1n=event('first_head_after_takeover'),
                first_other_over1n=event('first_other_nonfoot_after_takeover'))
        results.append(dict(case=case, old=outcome(old), new=outcome(new),
            common_observations=len(ticks), only_old_ticks=sorted(set(left)-set(right)),
            only_new_ticks=sorted(set(right)-set(left)), first_stance_tick_either=first_stance,
            first_command_difference=first_action, first_state_difference=first_state,
            maximum_difference_before_either_stance=before,
            old_clean_stable=case in old_ok, new_clean_stable=case in new_ok))
    files = [Path(__file__), args.old_support, args.new_support]
    for prep,data in [(args.old_preparation, old), (args.new_preparation, new)]:
        files += [prep/name for name in ['plan.json', 'controller.json', 'native_audit.json']]
        files += [data['run']/name for name in ['job.json', 'metrics.json',
            'pure_control_commands.npz', 'pure_control_diagnostics.jsonl']]
    result = dict(summary=dict(old_clean_stable=sorted(old_ok), new_clean_stable=sorted(new_ok),
        retained=sorted(old_ok & new_ok), lost=sorted(old_ok-new_ok), added=sorted(new_ok-old_ok),
        old_historical_complete5=old['audit']['complete5'], new_historical_complete5=new['audit']['complete5'],
        action_difference_before_stance=[r['case'] for r in results if
            r['maximum_difference_before_either_stance']['command_rad'] > 2e-6]),
        cases=results, source_sha256={str(f.resolve()):sha(f) for f in files},
        contact_threshold_note='Any positive nonfoot ground force prevents clean; 1 N only dates diagnostic events.',
        difference_thresholds=dict(command_rad=2e-6, q=2e-6, v=2e-5),
        numerical_differences_not_causal_proof=True, formal_acceptance=False, training=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
