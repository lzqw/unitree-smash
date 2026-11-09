"""Locate first post-release head contacts within a completed native control chain.

Command-source association is not causal attribution. The >1 N timestamps are
diagnostic events, not a replacement for the strict any-positive-force audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'support-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists() or read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh audit of completed native evidence required')
    support = read(args.support_audit)
    for path, digest in support['source_sha256'].items():
        if sha(path) != digest: raise ValueError('Changed support evidence '+path)
    records = {r['case']: r for r in support['cases'] if r['first_head_after_takeover']}
    histories = {case: [] for case in records}
    for line in (args.run/'pure_control_diagnostics.jsonl').open():
        row = json.loads(line); case = row['lane']
        if case not in records: continue
        end = records[case]['first_head_after_takeover']['preceding_control_observation']['tick']
        if records[case]['takeover_tick'] <= row['tick'] <= end:
            histories[case].append(row)
    output = []
    for case, record in records.items():
        head = record['first_head_after_takeover']; observation = head['preceding_control_observation']
        history = histories[case]
        if not history or history[-1]['tick'] != observation['tick']:
            raise ValueError('Missing last pre-contact control observation')
        last = history[-1]; diag = last['diagnostic']
        if diag['source'] != observation['command_source']:
            raise ValueError('Support audit and native command source disagree')
        recent = [r for r in history if r['tick'] >= last['tick']-9]
        fallback = [r['tick'] for r in history if r['diagnostic']['source']=='uncertified_posture_fallback']
        ground = last['phase']['first_ground_time_s']
        nominal_end = record['nominal_reference_end_tick']
        output.append(dict(case=case, takeover_tick=record['takeover_tick'],
            initial_reference_matched=record['reference_source_case'] is not None,
            nominal_reference_end_tick=nominal_end, first_head_time_s=head['time_s'],
            head_after_reference_end=(head['time_s']>=nominal_end*.01 if nominal_end is not None else None),
            first_ground_control_time_s=ground,
            head_delay_from_ground_control_s=(head['time_s']-ground if ground is not None else None),
            last_precontact_tick=last['tick'], last_precontact_source=diag['source'],
            solver_accepted=diag['solver_accepted'], solver_status=diag['solver_status'],
            rejection=diag.get('rejection'), head_lower=diag.get('head_lower'),
            last_precontact_phase=last['phase']['phase'],
            ground_contacts=last['phase']['ground_contacts'],
            observation=observation, first_fallback_tick=min(fallback, default=None),
            recent_control_steps=len(recent),
            last_100ms_command_source_counts=dict(Counter(r['diagnostic']['source'] for r in recent)),
            last_100ms_solver_status_counts=dict(Counter(r['diagnostic']['solver_status'] for r in recent))))
    groups = {}
    for matched in [False, True]:
        rows = [r for r in output if r['initial_reference_matched']==matched]
        total = sum((r['reference_source_case'] is not None)==matched for r in support['cases'])
        groups['matched' if matched else 'unmatched'] = dict(owned_cases=total, head_contact_cases=len(rows),
            head_after_nominal_reference_end=sum(r['head_after_reference_end'] is True for r in rows),
            last_precontact_sources=dict(Counter(r['last_precontact_source'] for r in rows)))
    files = [Path(__file__), args.support_audit, args.run/'job.json', args.run/'pure_control_diagnostics.jsonl']
    result = dict(scope=__doc__, groups=groups, cases=output,
        total_head_contact_cases=len(output),
        last_precontact_sources=dict(Counter(r['last_precontact_source'] for r in output)),
        timing_resolution_note='Last command observation precedes physics-frame contact; ground time is control sampled',
        causal_attribution=False, physics_execution=False, acceptance_changed=False,
        source_sha256={str(p.resolve()):sha(p) for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(args.output)
    print(json.dumps(dict(groups=groups, total=len(output), sources=result['last_precontact_sources'])))


if __name__ == '__main__': main()
