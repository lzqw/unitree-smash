#!/usr/bin/env python3
"""Check executed command-cost selection and count its actual native use.

This checks branch provenance and command mapping, not a counterfactual benefit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def validate_selection(row, weight):
    s = row['command_cost_selection']
    if s['declared_weight'] != weight or s['same_control_tick'] != row['control_tick']:
        raise ValueError('Different weight or control tick')
    fallback = s['baseline_fallback_evaluated']
    expected = weight > 0 and not s['command_cost_horizon_accepted']
    # With no support/risk solve, no alternative plan is available.
    if fallback and (not expected or s['selected'] != 'baseline' or s['selected_weight'] != 0):
        raise ValueError('Invalid fallback selection')
    if not fallback and (s['selected'] != 'command_cost' or s['selected_weight'] != weight):
        raise ValueError('Invalid candidate selection')
    expected_accepted = s['baseline_horizon_accepted'] if fallback else s['command_cost_horizon_accepted']
    if bool(row['accepted']) != bool(expected_accepted):
        raise ValueError('Native record differs from selected plan acceptance')
    if s['future_student_policy_predicted'] or s['full_horizon_acceptance_is_native_success']:
        raise ValueError('Unsupported predictive or safety claim')
    times = [s['candidate_runtime_s'], s['total_runtime_s']]
    if fallback:
        times.append(s['baseline_runtime_s'])
    elif s['baseline_runtime_s'] is not None:
        raise ValueError('Unexpected fallback runtime')
    if not all(math.isfinite(t) and t >= 0 for t in times):
        raise ValueError('Invalid runtime')
    inner = s['candidate_runtime_s'] + (s['baseline_runtime_s'] or 0.)
    if s['total_runtime_s'] + 1e-8 < inner:
        raise ValueError('Reported time excludes a candidate solve')
    if abs(row['runtime_ms'] - 1000 * s['total_runtime_s']) > 1e-5:
        raise ValueError('Native runtime omits candidate or fallback')
    return s['selected']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'command-audit', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve prior evidence')
    read = lambda n: json.loads((a.run / n).read_text())
    inputs, job = read('INPUTS.json'), read('job.json')
    audit = json.loads(a.command_audit.read_text())
    if (job['status'] != 'SUCCEEDED' or not inputs['landing_mpc_minimum_command_cost'] or
            not audit['passed'] or Path(audit['run']).resolve() != a.run.resolve()):
        raise ValueError('Matching completed native command audit required')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    for path, digest in audit['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed audited evidence')
    records = read('landing_mpc_records.json')
    totals, applied, cases = Counter(), Counter(), {}
    for r in records:
        mode = validate_selection(r, inputs['landing_mpc_command_change_weight'])
        totals[mode] += 1
        case = cases.setdefault(r['case'], Counter())
        case[mode] += 1
        if r.get('applied_first_step'):
            applied[mode] += 1
            case['applied_' + mode] += 1
            case['applied_full_horizon_accepted'] += bool(r['accepted'])
    sources = [Path(__file__), a.command_audit] + [a.run / n for n in
        ['INPUTS.json', 'job.json', 'metrics.json', 'landing_mpc_records.json']]
    result = dict(passed=bool(records), scope=__doc__, run=str(a.run.resolve()),
                  weight=inputs['landing_mpc_command_change_weight'], records=len(records),
                  selected=dict(totals), applied=dict(applied), cases=cases,
                  complete5=read('metrics.json')['complete5'],
                  source_sha256={str(f.resolve()): sha(f) for f in sources})
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'source_sha256']}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
