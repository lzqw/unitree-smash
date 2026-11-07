#!/usr/bin/env python3
"""Compact verified constraint-conflict and native-prediction evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    names = ['constraint_conflicts_with_state.json', 'controller35_active_set_audit.json',
             'controller35_arm_feasibility_audit.json', 'controller35_native_prediction_error.json']
    inputs = {name:json.loads((a.directory/name).read_text()) for name in names}
    for report in inputs.values():
        for name, expected in report['source_sha256'].items():
            if sha(name) != expected:
                raise ValueError('Audited source changed: ' + name)
    conflict, before, after, prediction = (inputs[name] for name in names)
    sources = {str(Path(report['source']).resolve()) for report in inputs.values()}
    if len(sources) != 1:
        raise ValueError('Different native source runs')
    old = {(r['case'],r['tick']):r for r in before['records']}
    new = {(r['case'],r['tick']):r for r in after['records']}
    if old.keys() != new.keys():
        raise ValueError('Different recorded-state panels')
    transitions = {}
    for key in ['candidate_first_step_accepted','candidate_horizon_accepted']:
        transitions[key] = dict(before=sum(r[key] for r in old.values()), after=sum(r[key] for r in new.values()),
            gained=[list(k) for k in old if new[k][key] and not old[k][key]],
            lost=[list(k) for k in old if old[k][key] and not new[k][key]])
    summaries = {}
    for label, predicate in [('all',lambda r:True), ('applied',lambda r:r['applied']),
            ('applied_unchanged_contact',lambda r:r['applied'] and not r['observed_contact_change']),
            ('rescued_cases',lambda r:r['case'] in [3,8,58,61,67,85,95,115])]:
        rows = [r for r in prediction['records'] if predicate(r)]
        summaries[label] = dict(samples=len(rows),models={})
        for model in ['held_initial_torque','implicit_16','implicit_160']:
            valid = [r['models'][model] for r in rows if r['models'][model]['ok']]
            fields = ['arm_position_error_max_rad','leg_position_error_max_rad','arm_velocity_error_max_rad_s',
                      'leg_velocity_error_max_rad_s','base_position_error_m','base_linear_velocity_error_mps',
                      'base_angular_velocity_error_rad_s']
            summaries[label]['models'][model] = dict(successful_predictions=len(valid),
                errors={k:dict(median=float(np.median([r[k] for r in valid])),
                    p95=float(np.quantile([r[k] for r in valid],.95)),maximum=max(r[k] for r in valid)) for k in fields})
    rejected = []
    for row in conflict['records']:
        if not row['free_arm']['accepted']:
            q = np.asarray(row['joint_position_rad']);limits = np.asarray(row['joint_limits_rad'])
            margin = np.minimum(q-limits[:,0],limits[:,1]-q)
            i = int(np.argmin(margin))
            rejected.append(dict(case=row['case'],tick=row['tick'],joint=row['joint_order'][i],
                joint_position_rad=float(q[i]),joint_velocity_rad_s=row['joint_velocity_rad_s'][i],
                limits_rad=limits[i].tolist(),nearest_limit_margin_rad=float(margin[i]),
                interpretation='Near-bound state; frozen-contact/unsaturated model conflict, not proof of hardware impossibility.'))
    output = dict(schema='interval-arm-feasibility-evidence-v1', native_new_execution=False,
        source=next(iter(sources)),constraint_conflicts=conflict['summary'],
        recorded_panel=transitions,free_arm_still_rejected_states=rejected,
        new_first_step_later_failures=[dict(case=k[0],tick=k[1],failed_stage=new[k]['candidate_failed_stage'],
            rejection=new[k]['rejection']) for k in old if new[k]['candidate_first_step_accepted'] and not old[k]['candidate_first_step_accepted']],
        native_one_step_prediction=summaries,
        observed_contact_changed_samples=prediction['contact_changed_samples'],
        scope='No constraint-family-removal LP is executable. Only arm preservation equality can be removed in the candidate; original physical limits remain. One-step errors do not establish long-horizon prediction accuracy.',
        source_sha256={str((a.directory/name).resolve()):sha(a.directory/name) for name in names})
    output['source_sha256'][str(Path(__file__).resolve())] = sha(__file__)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(conflicts=conflict['summary'],transitions=transitions)))


if __name__ == '__main__':
    main()
