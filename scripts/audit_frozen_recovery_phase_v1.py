#!/usr/bin/env python3
"""Classify frozen episodes by observed contact and recovery phase.

The report separates post-hit recovery failures from episodes that never had a
qualified airborne contact. It is an outcome audit, not a causal claim and
does not alter the evaluator or task definition.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_complete_demonstrations_v1 import classify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_phase(row, label):
    outcome = label['original_outcomes']
    recovery = row.get('contact_flight_recovery', {})
    if not outcome['airborne_contact']:
        return 'no_qualified_airborne_contact'
    if outcome['conditional_recovery']:
        return 'airborne_contact_and_stable_recovery'
    if outcome['fault']:
        return 'airborne_contact_recovered_body_but_rule_fault'
    if recovery.get('body_landing'):
        return 'airborne_contact_nonfoot_landing'
    if recovery.get('qualified_contact_landing') and not recovery.get('stable_recovery'):
        return 'airborne_contact_no_stable_window'
    if not recovery.get('qualified_contact_landing'):
        return 'airborne_contact_no_qualified_foot_landing'
    return 'airborne_contact_other_recovery_failure'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Require a fresh output path')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['_root'] = str(ROOT)
    rows_path = args.source / 'episodes.json'
    rows = json.loads(rows_path.read_text())
    counts = Counter()
    speed_positive = Counter()
    details = []
    for row in rows:
        label = classify(row, protocol, 5.0)
        phase = classify_phase(row, label)
        counts[phase] += 1
        speed = label['outgoing_speed_mps']
        ball_positive = bool(
            speed is not None and speed >= 5.0 and
            all(value for name, value in label['criteria_except_speed'].items()
                if name != 'same_flight_stable_recovery'))
        if ball_positive:
            speed_positive[phase] += 1
        details.append(dict(
            source_training_index=row['source_training_index'],
            env_id=row['env_id'], episode_index=row['episode_index'],
            phase=phase, ball_positive_5mps=ball_positive,
            outgoing_speed_mps=speed,
            airborne_contact=label['original_outcomes']['airborne_contact'],
            conditional_recovery=label['original_outcomes']['conditional_recovery'],
            qualified_contact_landing=row.get('contact_flight_recovery', {}).get('qualified_contact_landing'),
            stable_recovery=row.get('contact_flight_recovery', {}).get('stable_recovery'),
            body_landing=row.get('contact_flight_recovery', {}).get('body_landing'),
            recovery_s=row.get('contact_flight_recovery', {}).get('recovery_s'),
            rejection_reasons=label['rejection_reasons']))
    result = dict(
        schema='frozen-recovery-phase-audit-v1',
        source=str(args.source.resolve()), episodes=len(rows),
        counts=dict(counts), ball_positive_5mps_by_phase=dict(speed_positive),
        details=details,
        source_sha256={str(rows_path.resolve()): sha(rows_path),
                       str(protocol_path): sha(protocol_path),
                       str(Path(__file__).resolve()): sha(Path(__file__))},
        causal_effect_estimated=False, formal_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['episodes', 'counts', 'ball_positive_5mps_by_phase']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
