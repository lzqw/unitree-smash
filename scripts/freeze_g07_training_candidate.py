#!/usr/bin/env python3
"""Freeze a development candidate using a prospective rule, never formal labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--selection-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.training_output
    job = json.loads((source/'job.json').read_text())
    evaluation = json.loads((source/'eval_manifest.json').read_text())
    receipt = json.loads((source/'completed.json').read_text())
    pool = json.loads(args.candidate_manifest.read_text())
    plan = json.loads(args.selection_plan.read_text())
    if job['status'] != 'SUCCEEDED' or evaluation['formal_evaluation_eligible']:
        raise ValueError('Selection requires completed development simulation only')
    if receipt['episodes'] != evaluation['episodes']:
        raise ValueError('Incomplete source episodes')
    if args.output.exists():
        raise ValueError('Use a new frozen output directory')
    if plan['selection_rule']['exclude_candidate_ids'] != [0]:
        raise ValueError('This predeclared study excludes the unchanged reference')
    ranked = []
    for row in json.loads((source/'candidate_metrics.json').read_text()):
        index = row['candidate_id']
        if index == 0 or row['metrics']['numerical_failures']:
            continue
        groups = row['groups']
        if set(groups) != {'training_a', 'training_b'}:
            raise ValueError('Only the independently drawn noisy training groups are permitted')
        full = min(g['complete_smash_success_rate'] for g in groups.values())
        air = min(g['airborne_contact_rate'] for g in groups.values())
        recovery = min(g['recovery_given_airborne_contact'] for g in groups.values())
        score = min(full/.5, air/.7, recovery/.85)
        ranked.append(dict(candidate_id=index, name=pool['candidates'][index]['name'],
                           score=score, minimum_complete=full, minimum_airborne=air,
                           minimum_recovery=recovery, training_thresholds_met=score >= 1.,
                           robot_net_contact=row['metrics']['robot_net_contact'], groups=groups))
    if not ranked:
        raise ValueError('No finite new candidate to freeze')
    ranked.sort(key=lambda row: (-row['score'], -row['minimum_complete'],
                                -row['minimum_recovery'], row['robot_net_contact'], row['candidate_id']))
    selected = ranked[0]
    candidate = pool['candidates'][selected['candidate_id']]
    checkpoint = Path(candidate['path'])
    actual = digest(checkpoint)
    expected = evaluation['candidate_checkpoints'][selected['candidate_id']]['sha256']
    if actual != candidate['sha256'] or actual != expected:
        raise ValueError('Candidate checkpoint changed since the measured run')
    args.output.mkdir(parents=True)
    shutil.copyfile(checkpoint, args.output/'policy.pt')
    result = dict(kind='Best new training candidate; independent acceptance not established',
                  frozen_at=datetime.now(timezone.utc).isoformat(), selected=selected,
                  ranking=ranked, checkpoint_sha256=actual, source_checkpoint=str(checkpoint),
                  source_training_output=str(source), source_job_sha256=digest(source/'job.json'),
                  selection_plan=str(args.selection_plan), selection_plan_sha256=digest(args.selection_plan),
                  formal_scores_used=False, formal_acceptance=False)
    (args.output/'FROZEN.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
