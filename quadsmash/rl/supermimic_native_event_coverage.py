"""Measure independent native cases settled during Approach collection."""
import math


COVERAGE_KEYS = ('settled_cases', 'qualified_string', 'low_speed', 'frame',
                 'near_edge_under_10_mm', 'marginal_success')


def coverage_targets(receipt):
    counts = receipt.get('counts') if isinstance(receipt, dict) else None
    if (not isinstance(counts, dict) or set(counts) != set(COVERAGE_KEYS)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or not counts['settled_cases'] or not counts['qualified_string']
            or counts['qualified_string'] + counts['frame'] > counts['settled_cases']
            or any(counts[key] > counts['qualified_string'] for key in
                   ('low_speed', 'near_edge_under_10_mm', 'marginal_success'))):
        raise ValueError('Expected measured, internally consistent Approach native event counts')
    return counts


def coverage_deficits(counts, targets):
    return {key: targets[key] - counts[key] for key in COVERAGE_KEYS if counts[key] < targets[key]}


def native_event_coverage(episodes, holdout_lanes, seconds, *, indexed_episodes=None):
    """Count settled cases available for training-side labels in one stream."""
    distinct = {}
    for row in episodes:
        key = (row['env_id'], row['episode_index'])
        if key in distinct:
            raise ValueError(f'Duplicate settled native episode {key}')
        distinct[key] = row
    training = {
        row['case_id']: row for episode_key, row in distinct.items()
        if row['env_id'] not in holdout_lanes and row.get('arm_identification_episode') is False
        and not row['numerical_failures']
        and (indexed_episodes is None or episode_key in indexed_episodes)
    }
    qualified = [row for row in training.values() if row['qualified_airborne_string_contact']]
    margins = [row['native_contact'].get('signed_string_margin_m') for row in qualified]
    known_margins = [value for value in margins if value is not None and math.isfinite(value)]
    counts = dict(
        settled_cases=len(training),
        qualified_string=len(qualified),
        low_speed=sum('speed_threshold' in row['rejection_reasons'] for row in qualified),
        frame=sum(row['native_contact']['frame_hit'] and not row['native_contact']['string_hit']
                  for row in training.values()),
        near_edge_under_10_mm=sum(value < .010 for value in known_margins),
        marginal_success=sum(row['ball_side_complete_except_recovery']
                             and row['outgoing_speed_mps'] is not None
                             and 4.5 <= row['outgoing_speed_mps'] < 4.75 for row in qualified),
    )
    scope = ('settled training-side cases with indexed Actor starts' if indexed_episodes is not None
             else 'settled training-side cases, before checking start snapshot availability')
    return dict(scope=scope, settled_episodes=len(distinct), training_case_ids=sorted(training),
                missing_qualified_margins=len(margins) - len(known_margins),
                counts=counts, per_collection_hour={key: count * 3600 / seconds if seconds else None
                                                   for key, count in counts.items()})
