"""Reconcile audited speed losses with native terminal labels."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from run_supermimic_approach_v2 import dump


def labelled_speed_risk(audit, labels):
    risk = []
    for row in audit:
        label = labels.get(tuple(row['episode']), {})
        speed = label.get('outgoing_speed_mps')
        if row['outcome'] == 'low_speed':
            group = 'low_speed'
        elif label.get('ball_side_complete_except_recovery') and speed is not None and 4.5 <= speed < 4.75:
            group = 'marginal_success'
        else:
            continue
        modeled = row['release_speed_mps']
        risk.append(dict(row, risk_group=group, case_id=label.get('case_id'), actual_speed_mps=speed,
                         signed_error_mps=modeled - speed if modeled is not None and speed is not None else None))
    return risk


def summarize(rows):
    speeds = [row['signed_error_mps'] for row in rows if row['signed_error_mps'] is not None]
    return dict(rows=len(rows), distinct_episodes=len({tuple(row['episode']) for row in rows}),
                with_native_speed=len(speeds), median_signed_error_mps=median(speeds) if speeds else None,
                predicted_missing_release=sum(not row['predicted_release'] for row in rows),
                speed_loss_inactive=sum(not row['speed_loss_active'] for row in rows),
                speed_loss_below_0_001=sum(row['speed_loss'] is not None and row['speed_loss'] < .001 for row in rows))


def audit_history(history, labels):
    attempts = []
    total_proposals = 0
    proposals_with_logging = 0
    for row in history:
        total_proposals += 1
        proposals_with_logging += bool(row.get('proposal_outcome_audit') or row.get('guard_outcome_audit_before'))
        proposal = labelled_speed_risk(row.get('proposal_outcome_audit', ()), labels)
        guard = labelled_speed_risk(row.get('guard_outcome_audit_before', ()), labels)
        if not (proposal or guard):
            continue
        retained = []
        merit = None
        if row['update_accepted']:
            trial = next(trial for trial in row['backtrack_trials'] if trial['alpha'] == row['backtrack_alpha'])
            retained = labelled_speed_risk(trial.get('guard_outcome_audit', ()), labels)
            merit = trial['baseline_release_flight']
        after = {tuple(case['episode']): case for case in retained}
        paired = []
        for before in guard:
            candidate = after.get(tuple(before['episode']))
            if candidate is not None:
                old_speed, new_speed = before['release_speed_mps'], candidate['release_speed_mps']
                old_terms, new_terms = before.get('flight_cost_terms'), candidate.get('flight_cost_terms')
                term_delta = ({name: new_terms[name] - value for name, value in old_terms.items()}
                              if old_terms is not None and new_terms is not None else None)
                paired.append(dict(before=before, after=candidate,
                    lost_release=before['predicted_release'] and not candidate['predicted_release'],
                    speed_delta_mps=new_speed - old_speed if old_speed is not None and new_speed is not None else None,
                    flight_cost_delta=term_delta))
        attempts.append(dict(attempt=row['attempt_id'], update_accepted=row['update_accepted'],
                             proposal=proposal, guard=guard, accepted_guard_pairs=paired, flight_merit=merit))
    groups = ('low_speed', 'marginal_success')
    metrics = {group: {part: summarize([case for row in attempts for case in row[part] if case['risk_group'] == group])
                       for part in ('proposal', 'guard')} for group in groups}
    accepted_pairs = [(pair, row['flight_merit']) for row in attempts for pair in row['accepted_guard_pairs']]
    risk_tradeoffs = {}
    for group in groups:
        paired = [(pair, merit) for pair, merit in accepted_pairs if pair['before']['risk_group'] == group]
        speed_changes = [pair['speed_delta_mps'] for pair, _ in paired if pair['speed_delta_mps'] is not None]
        risk_tradeoffs[group] = dict(
            paired_starts=len(paired), unique_episodes=len({tuple(pair['before']['episode']) for pair, _ in paired}),
            modeled_speed_decreased=sum(change < 0 for change in speed_changes),
            modeled_speed_delta_median_mps=median(speed_changes) if speed_changes else None,
            speed_loss_increased_with_improved_aggregate_flight=sum(
                pair['flight_cost_delta'] is not None and pair['flight_cost_delta']['speed'] > 0
                and merit is not None and merit.get('after') is not None and merit['after'] < merit['before']
                for pair, merit in paired),
        )
    return dict(total_proposals=total_proposals, proposals_with_logging=proposals_with_logging,
                attempts_with_risk=len(attempts), trials_with_accepted_guard_pairs=sum(
        bool(row['accepted_guard_pairs']) for row in attempts), metrics=metrics,
                accepted_guard_speed_tradeoffs=risk_tradeoffs, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True,
                        help='Supply banks in chronological order. Later banks supersede overlapping labels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels = {}
    legacy_checkpoints = []
    for path in args.checkpoint:
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        events = state['actor_start_bank'].get('resolved_events')
        if events is None:
            legacy_checkpoints.append(str(path))
        else:
            labels.update((tuple(key), label) for key, (_, label) in events.items())
        del state
    if not labels:
        parser.error('None of the supplied checkpoints contains resolved native episode labels')
    with args.history.open() as stream:
        result = audit_history(map(json.loads, stream), labels)
    result['label_source'] = dict(resolved_episodes=len(labels), legacy_checkpoints_without_labels=legacy_checkpoints)
    dump(args.output, result)


if __name__ == '__main__':
    main()
