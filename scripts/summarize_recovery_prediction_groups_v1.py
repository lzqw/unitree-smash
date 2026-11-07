#!/usr/bin/env python3
"""Stratify archived one-step predictions; this is not a causal experiment.

Only actual applied commands sampled every 20 ticks are represented. Require
the complete prediction interval to precede the first recorded nonfoot event.
Report per-case statistics as well as pooled ticks to expose duration bias.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ('leg_position_rad', 'arm_position_rad',
           'world_linear_velocity_mps', 'world_angular_velocity_rad_s')


def stats(rows):
    common = [r for r in rows if all(r['models'][m]['ok']
                                   for m in ('frozen', 'refreshed'))]
    result = dict(samples=len(rows), paired_successful_samples=len(common),
                  cases=sorted({r['case'] for r in rows}), models={})
    for name in ('frozen', 'refreshed'):
        values = [r['models'][name] for r in common]
        normal = 'normal_force_violation_n' if name == 'frozen' else 'normal_force_n'
        friction = 'friction_violation_n' if name == 'frozen' else 'friction_n'
        result['models'][name] = dict(
            failed=sum(not r['models'][name]['ok'] for r in rows),
            p95={k: float(np.quantile([v[k] for v in values], .95))
                 if values else None for k in METRICS},
            negative_normal_force_samples=sum(v['diagnostics'][normal] > 1e-6 for v in values),
            friction_cone_violation_samples=sum(v['diagnostics'][friction] > 1e-6 for v in values))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    sources = [a.directory / f for f in (
        'relinearized_prediction.json', 'acceptance_timing.json', 'recovery_onset_v2.json')]
    prediction, timing, onset = [json.loads(f.read_text()) for f in sources]
    if Path(prediction['source']).resolve() != Path(timing['source']).resolve():
        raise ValueError('Prediction and contact evidence must refer to the same native run')
    events = {r['case']: r['first_observed_nonfoot_s'] for r in timing['cases']}
    cases = {r['case']: r for r in onset['cases']}
    if len(cases) != 53:
        raise ValueError('Expected original 53 ball-qualified cases')
    records = []
    for r in prediction['records']:
        if r['case'] not in cases:
            continue
        event = events[r['case']]
        r = dict(r, group=cases[r['case']]['group'],
                 before_nonfoot=event is None or .01 * (r['tick'] + 1) < event)
        first = cases[r['case']]['first_applied_qp_tick']
        r['early'] = first is not None and first <= r['tick'] < first + 40
        records.append(r)
    groups = {}
    for group in ('retained', 'lost', 'rescued', 'unresolved'):
        selected = [r for r in records if r['group'] == group]
        valid = [r for r in selected if r['before_nonfoot']]
        groups[group] = dict(
            case_ids=sorted(i for i, c in cases.items() if c['group'] == group),
            all=stats(selected), before_nonfoot=stats(valid),
            early_before_nonfoot=stats([r for r in valid if r['early']]),
            force_flag_unchanged=stats([r for r in valid if not r['force_flag_changed']]),
            force_flag_changed=stats([r for r in valid if r['force_flag_changed']]))
    per_case = []
    for case, info in sorted(cases.items()):
        selected = [r for r in records if r['case'] == case and r['before_nonfoot']]
        per_case.append(dict(case=case, group=info['group'], before_nonfoot=stats(selected)))
    for group, data in groups.items():
        data['case_balanced_median_of_case_p95'] = {m: {
            k: float(np.median(v)) if v else None for k in METRICS
            for v in [[c['before_nonfoot']['models'][m]['p95'][k] for c in per_case
                       if c['group'] == group and c['before_nonfoot']['paired_successful_samples']]]}
            for m in ('frozen', 'refreshed')}
    output = dict(scope=__doc__, native_new_execution=False,
        source_run=prediction['source'], groups=groups, cases=per_case,
        limitations=[
            'Observational outcome groups; no causal attribution or controller promotion.',
            'Sampling every 200 ms misses brief impacts and excludes rejected-QP commands.',
            'Both models retain the initial contact set for 10 ms; no new collision/slip model.',
            'Computable predictions may violate unilateral force or friction; counts are retained.',
            'First nonfoot event is the recorded sensor-threshold crossing, not exact collision onset.',
            'An unchanged thresholded force flag does not imply unchanged physical contact.'],
        source_sha256={str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sources + [Path(__file__)]})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({g: {k: d[k] for k in ('case_ids', 'before_nonfoot', 'early_before_nonfoot')}
                      for g, d in groups.items()}))


if __name__ == '__main__':
    main()
