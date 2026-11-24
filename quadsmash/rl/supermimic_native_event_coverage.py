"""Measure independent native cases settled during Approach collection."""
import math


COVERAGE_KEYS = ('settled_cases', 'qualified_string', 'low_speed', 'frame',
                 'near_edge_under_10_mm', 'marginal_success')


def coverage_targets(receipt):
    counts = receipt.get('counts') if isinstance(receipt, dict) else None
    if (not isinstance(counts, dict) or set(counts) != set(COVERAGE_KEYS)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or not counts['settled_cases'] or not counts['qualified_string']
            or max(counts['qualified_string'], counts['frame']) > counts['settled_cases']
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
    training = [
        row for episode_key, row in distinct.items()
        if row['env_id'] not in holdout_lanes and row.get('arm_identification_episode') is False
        and not row['numerical_failures']
        and (indexed_episodes is None or episode_key in indexed_episodes)
    ]
    # Later attempts on a case can change the outcome while an earlier indexed
    # start remains available. Count a case once within each observed category.
    training_ids = {row['case_id'] for row in training}
    qualified = [row for row in training if row['qualified_airborne_string_contact']]
    qualified_ids = {row['case_id'] for row in qualified}
    margins = [(row['case_id'], row['native_contact'].get('signed_string_margin_m')) for row in qualified]
    known_margins = [(case_id, value) for case_id, value in margins if value is not None and math.isfinite(value)]
    counts = dict(
        settled_cases=len(training_ids),
        qualified_string=len(qualified_ids),
        low_speed=len({row['case_id'] for row in qualified if 'speed_threshold' in row['rejection_reasons']}),
        frame=len({row['case_id'] for row in training
                   if row['native_contact']['frame_hit'] and not row['native_contact']['string_hit']}),
        near_edge_under_10_mm=len({case_id for case_id, value in known_margins if value < .010}),
        marginal_success=len({row['case_id'] for row in qualified if row['ball_side_complete_except_recovery']
                              and row['outgoing_speed_mps'] is not None
                              and 4.5 <= row['outgoing_speed_mps'] < 4.75}),
    )
    scope = ('settled training-side cases with indexed Actor starts' if indexed_episodes is not None
             else 'settled training-side cases, before checking start snapshot availability')
    return dict(scope=scope, settled_episodes=len(distinct), training_case_ids=sorted(training_ids),
                missing_qualified_margins=len(qualified_ids - {case_id for case_id, _ in known_margins}),
                counts=counts, per_collection_hour={key: count * 3600 / seconds if seconds else None
                                                   for key, count in counts.items()})
