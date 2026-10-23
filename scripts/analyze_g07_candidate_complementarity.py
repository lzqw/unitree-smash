#!/usr/bin/env python3
"""Analyze developer candidates; oracle labels are never a deployed selector."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new analysis file')
    manifest = json.loads((a.input/'eval_manifest.json').read_text())
    if manifest.get('formal_evaluation_eligible'):
        raise ValueError('Formal holdout outcomes cannot be used for developer selection')
    rows = json.loads((a.input/'episodes.json').read_text())
    candidates = len(manifest['candidate_checkpoints'])
    cases = manifest['source_cases_per_candidate']
    lookup = {(r['candidate_id'], r['source_scenario']): r for r in rows}
    if len(lookup) != candidates*cases or len(rows) != len(lookup):
        raise ValueError('Missing or duplicated candidate/scenario outcomes')
    for candidate in range(candidates):
        if {r['source_scenario'] for r in rows if r['candidate_id']==candidate} != set(range(cases)):
            raise ValueError('Candidate did not cover every developer scenario')
    free = lambda r: not r['robot_net_contact'] and not r['numerical_failures']
    full = lambda r: bool(r['complete_smash'] and r['ground_init'] and free(r))
    recovery = lambda r: bool(r['airborne_contact'] and r['recovery_success'] and free(r))
    details = []
    for case in range(cases):
        batch = [lookup[candidate, case] for candidate in range(candidates)]
        groups = {r['development_group'] for r in batch}
        if len(groups) != 1:
            raise ValueError('Candidate case/reset mapping differs')
        details.append(dict(source_scenario=case, group=groups.pop(),
            complete_candidates=[i for i, r in enumerate(batch) if full(r)],
            airborne_recovery_candidates=[i for i, r in enumerate(batch) if recovery(r)]))
    groups = {}
    for group in sorted({r['group'] for r in details}):
        batch = [r for r in details if r['group']==group]
        groups[group] = dict(cases=len(batch),
            oracle_any_complete_rate=sum(bool(r['complete_candidates']) for r in batch)/len(batch),
            oracle_any_airborne_recovery_fraction=sum(bool(r['airborne_recovery_candidates']) for r in batch)/len(batch),
            fixed_complete_rates=[sum(full(lookup[c, r['source_scenario']]) for r in batch)/len(batch)
                                  for c in range(candidates)])
    result = dict(kind='Developer complementarity analysis only; outcome oracle is not an executable policy',
        input=str(a.input), candidate_count=candidates, cases_per_candidate=cases, groups=groups,
        interpretation='Oracle chooses using completed outcomes. Any deployable gate must learn from public observations and be independently evaluated.',
        cases=details)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(candidate_count=candidates, groups=groups)), flush=True)


if __name__ == '__main__':
    main()
