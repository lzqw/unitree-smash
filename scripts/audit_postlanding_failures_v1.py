#!/usr/bin/env python3
"""Separate failed first support from later collapse in finished native runs.

Uses actual first nonfoot force-threshold events and original episode labels.
The output is observed timing and anatomy, not a counterfactual control claim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run, sha


def audit(source):
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(ROOT)
    summary, _ = read_run(source, protocol)
    rows = {r['source_training_index']: r for r in json.loads((source / 'episodes.json').read_text())}
    inputs = json.loads((source / 'INPUTS.json').read_text())
    env_to_index = dict(enumerate(inputs['training_indices']))
    metadata = json.loads((source / 'contact_telemetry_metadata.json').read_text())
    events = json.loads((source / 'first_nonfoot_contact_details.json').read_text())
    first = {}
    for event in events:
        for env_id in event['first_event_env_ids']:
            index = env_to_index[env_id]
            if index in first:
                raise ValueError('Expected only one first nonfoot event per original episode')
            first[index] = (env_id, event)
    if metadata['physics_dt_s'] != protocol['physics_dt_s']:
        raise ValueError('Force event timebase differs from the original protocol')
    # On this evaluator all lanes start together at real clock zero. The
    # metric frame is the actual native substep, with first-episode selection.
    selected = [s for s in summary['stages'] if s['ball5_except_recovery'] and not s['complete5']]
    cases = []
    for stage in selected:
        index = stage['training_index']
        row = rows[index]
        recovery = row['contact_flight_recovery']
        observed = None
        if index in first:
            env_id, event = first[index]
            patches = [dict(body=p['body'], normal_force_sum_n=sum(p['normal_force_n']))
                       for sensor in event['sensors'] for p in sensor['native_patch_contacts']
                       if p['env_id'] == env_id and sum(p['normal_force_n']) > 0]
            time_s = event['physics_frame'] * metadata['physics_dt_s']
            observed = dict(time_s=time_s,
                            delay_after_qualified_landing_s=time_s - recovery['contact_flight_landing_s']
                                if recovery['qualified_contact_landing'] else None,
                            native_patches=patches,
                            root_link_twist_world=event['root_link_twist_world_all_lanes'][env_id])
        if recovery['body_landing'] and observed is None:
            raise ValueError('Nonfoot failure has no matching native first-event record')
        cases.append(dict(training_index=index, ball_conditions_passed=True,
                          qualified_contact_landing=recovery['qualified_contact_landing'],
                          stable_recovery=recovery['stable_recovery'], longest_stable_s=recovery['recovery_s'],
                          nonfoot_seen=recovery['body_landing'], landing_s=recovery['contact_flight_landing_s'],
                          first_nonfoot=observed))
    delays = [r['first_nonfoot']['delay_after_qualified_landing_s'] for r in cases
              if r['first_nonfoot'] is not None and r['first_nonfoot']['delay_after_qualified_landing_s'] is not None]
    bodies = Counter(p['body'] for r in cases if r['first_nonfoot'] is not None
                     for p in r['first_nonfoot']['native_patches'])
    return dict(schema='native-ball-ok-postlanding-failure-timing-v1',
                source=str(source.resolve()), source_policy_sha256=summary['policy_sha256'],
                source_hashes={n: sha(source / n) for n in ['INPUTS.json', 'episodes.json',
                    'first_nonfoot_contact_details.json', 'contact_telemetry_metadata.json']},
                ball5_except_recovery=summary['ball5_except_recovery'], complete5=summary['complete5'],
                ball_ok_recovery_fail=len(cases), qualified_contact_landings=sum(r['qualified_contact_landing'] for r in cases),
                first_nonfoot_cases=sum(r['first_nonfoot'] is not None for r in cases),
                no_nonfoot_but_insufficient_stability=sum(not r['nonfoot_seen'] and not r['stable_recovery'] for r in cases),
                first_nonfoot_delay_after_landing_s_quantiles=dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                    np.quantile(delays, [0, .25, .5, .75, 1]).tolist())) if delays else None,
                positive_first_native_patch_body_case_counts=dict(bodies), cases=cases,
                formal_acceptance=False, generalization_test=False,
                scope='Original fixed128 TRAIN. First nonfoot force-threshold event observed after real ground execution; '
                      'event timing is not proof of a single control cause. No task or physics changes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audit')
    result = audit(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'source_hashes']}))


if __name__ == '__main__':
    main()
