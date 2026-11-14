"""Historical independent task scoring, applied only to complete native evidence.

No new success definition. Raw environment flags remain available separately.
This module does not import Isaac or mutate the input record.
"""
from copy import deepcopy


def attach_reset_jump_records(rallies, jumps):
    """Join records produced by ONE reset call, never search older episodes."""
    by_lane = {int(row['env_id']): row for row in jumps}
    lanes = [int(row['env_id']) for row in rallies]
    if len(by_lane) != len(jumps) or len(set(lanes)) != len(lanes):
        raise ValueError('Duplicate lane within one reset batch')
    if set(by_lane) != set(lanes):
        raise ValueError('Reset rally/jump evidence lanes differ')
    for row in rallies:
        row['jump'] = deepcopy(by_lane[int(row['env_id'])])
        row['jump']['episode_index'] = int(row['episode_index'])


def score_terminal_episode(row, protocol):
    from quadsmash.rl.supermimic_task_contract import summarize_episode
    from scripts.export_complete_demonstrations_v1 import classify

    if 'jump' not in row:
        raise ValueError('Missing terminal jump evidence; cannot reconstruct from reset state')
    if ('episode_index' in row['jump']
            and row['jump']['episode_index'] != row['episode_index']):
        raise ValueError('Jump evidence belongs to another episode')
    result = summarize_episode(row)
    label = classify(row, protocol, 5.)
    original = label['original_outcomes']
    result['raw_environment_flags'] = {
        name: row[name] for name in ('qualified_airborne_string_contact',
            'ball_side_complete_except_recovery', 'recovery_success', 'complete5')}
    result.update(
        scoring_schema='historical-independent-complete5-v1',
        qualified_airborne_string_contact=bool(original['airborne_contact']),
        # Historical conditional recovery also excludes rule faults. Keep the
        # physical recovery flag separately; do not silently call both recovery.
        recovery_success=bool(original['conditional_recovery']),
        physical_recovery_success=bool(row['recovery_success']),
        ball_side_complete_except_recovery=bool(
            label['outgoing_speed_mps'] is not None and label['outgoing_speed_mps'] >= 5.
            and all(v for k, v in label['criteria_except_speed'].items()
                    if k != 'same_flight_stable_recovery')),
        complete5=bool(label['curriculum_complete']),
        rally_fault=bool(original['fault']),
        criteria_except_speed=label['criteria_except_speed'],
        rejection_reasons=label['rejection_reasons'])
    return result
