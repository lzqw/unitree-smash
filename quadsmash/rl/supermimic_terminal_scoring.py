"""Historical independent task scoring, applied only to complete native evidence.

No new success definition. Raw environment flags remain available separately.
This module does not import Isaac or mutate the input record.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


STRICT_SCORING = 'historical-independent-complete5-v1'
RELAXED_SCORING = 'relaxed-complete5-v1'
SCORING_SCHEMAS = frozenset((STRICT_SCORING, RELAXED_SCORING))


def relaxed_scoring_protocol():
    path = Path(__file__).resolve().parents[2] / 'configs/evaluation_protocol_v3_relaxed.json'
    return json.loads(path.read_text())


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
    # Keep the case identity attached to the scored record.  The frozen
    # return monitor uses it to reject lane duplication or case-set drift
    # across refreshes.
    result['case_id'] = row.get('case_id', f"ENV{int(row['env_id'])}")
    label = classify(row, protocol, 5.)
    original = label['original_outcomes']
    result['raw_environment_flags'] = {
        name: row[name] for name in ('qualified_airborne_string_contact',
            'ball_side_complete_except_recovery', 'recovery_success', 'complete5')}
    result.update(
        scoring_schema=STRICT_SCORING,
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
    relaxed = protocol.get('relaxed_scoring')
    if relaxed is not None:
        if relaxed['scoring_schema'] != RELAXED_SCORING:
            raise ValueError('Unknown relaxed scoring schema')
        evidence = row['relaxed_recovery']
        result['strict_observed'] = {
            'complete5': True if result['complete5'] else (None if row['terminated_success'] else False),
            'recovery_success': True if result['recovery_success'] else (None if row['terminated_success'] else False),
            'censored_by_success_termination': bool(row['terminated_success']),
        }
        criteria = dict(label['criteria_except_speed'])
        del criteria['released_downward_forward']
        velocity = row['outgoing_velocity_mps']
        criteria['released_forward'] = bool(row['released'] and all(map(math.isfinite, velocity)) and velocity[0] > 0.)
        physical_recovery = bool(evidence['success'] and row['recovery_success'])
        criteria['same_flight_stable_recovery'] = physical_recovery
        speed_ok = label['outgoing_speed_mps'] is not None and label['outgoing_speed_mps'] >= relaxed['minimum_outgoing_speed_mps']
        ball_ok = bool(speed_ok and all(value for key, value in criteria.items() if key != 'same_flight_stable_recovery'))
        result.update(scoring_schema=RELAXED_SCORING, relaxed_recovery=evidence,
            recovery_success=physical_recovery and not original['fault'],
            physical_recovery_success=physical_recovery,
            ball_side_complete_except_recovery=ball_ok, complete5=ball_ok and physical_recovery,
            criteria_except_speed=criteria, terminated_success=bool(row['terminated_success']),
            rejection_reasons=[key for key, value in criteria.items() if not value] + ([] if speed_ok else ['speed_threshold']))
        if 'relaxed_survival_one_second' in row:
            result['relaxed_survival_one_second'] = row['relaxed_survival_one_second']
    return result
