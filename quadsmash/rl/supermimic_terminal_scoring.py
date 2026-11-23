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
APPROACH_BALL_SCORING = 'approach_ball_only_v1'
SCORING_SCHEMAS = frozenset((STRICT_SCORING, RELAXED_SCORING, APPROACH_BALL_SCORING))


def relaxed_scoring_protocol():
    path = Path(__file__).resolve().parents[2] / 'configs/evaluation_protocol_v3_relaxed.json'
    return json.loads(path.read_text())


def approach_ball_scoring_protocol():
    path = Path(__file__).resolve().parents[2] / 'configs/evaluation_protocol_v4_approach_ball_only.json'
    return json.loads(path.read_text())


def score_approach_ball_episode(row, protocol, result):
    from quadsmash.evaluation_v4 import FAULTS

    rules = protocol['approach_ball_scoring']
    if rules['scoring_schema'] != APPROACH_BALL_SCORING:
        raise ValueError('Unknown Approach-only scoring schema')
    terminal = row['approach_ball_terminal']
    court = json.loads((Path(protocol['_root']) / 'configs/court_net_v4.json').read_text())
    contact, velocity, landing = (row[k] for k in ('contact_point_m', 'outgoing_velocity_mps', 'first_landing_m'))
    hit, release, net, ground = (row[k] for k in
        ('contact_time_s', 'release_time_s', 'crossing_time_s', 'first_landing_time_s'))
    tolerance = protocol['physics_dt_s']
    airborne = bool(row['legal_initial_contact'] and row['string_hit'] and row['airborne_string_hit']
                    and all(map(math.isfinite, contact)) and contact[0] < court['net_plane_x_m'])
    faulted = any(row[k] for k in FAULTS)
    speed = math.hypot(*velocity) if all(map(math.isfinite, velocity)) and row['released'] else None
    criteria = dict(
        ground_single_launch=bool(row['ground_init'] and row['active_rally_launch_count'] == 1),
        legal_airborne_strings=airborne, no_rule_fault=not faulted,
        released_forward=speed is not None and velocity[0] > 0,
        actual_net_and_target=bool(row['crossed_net'] and row['landed_target'] and all(map(math.isfinite, landing))
            and court['target_x_limits_m'][0] <= landing[0] <= court['target_x_limits_m'][1]
            and court['opponent_x_limits_m'][0] <= landing[0] <= court['opponent_x_limits_m'][1]
            and abs(landing[1]) <= min(court['target_y_half_width_m'], court['singles_width_m'] / 2)),
        event_order=bool(all(map(math.isfinite, (hit, release, net, ground)))
            and hit >= protocol['preparation_seconds'] - tolerance and release >= hit
            and net + tolerance >= hit and ground + tolerance >= max(net, release)),
        no_numerical_failure=not terminal['numerical_abort'] and row['numerical_failures'] == 0,
        ball_terminal=terminal['ball_terminal'])
    speed_ok = speed is not None and speed >= rules['minimum_outgoing_speed_mps']
    ball_ok = all(criteria.values()) and speed_ok
    tail = terminal['recovery_tail_observed'] and criteria['no_numerical_failure']
    return dict(result, scoring_schema=APPROACH_BALL_SCORING, approach_ball_terminal=terminal,
                qualified_airborne_string_contact=airborne, rally_fault=faulted,
                ball_side_complete_except_recovery=ball_ok, outgoing_speed_mps=speed,
                contact_time_s=hit if math.isfinite(hit) else None,
                recovery_success=bool(airborne and row['recovery_success'] and not faulted) if tail else None,
                physical_recovery_success=bool(row['recovery_success']) if tail else None,
                complete5=bool(ball_ok and row['recovery_success']) if tail else None,
                criteria_except_speed=criteria,
                rejection_reasons=[k for k, passed in criteria.items() if not passed]
                + ([] if speed_ok else ['speed_threshold']))


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
    if protocol.get('approach_ball_scoring') is not None:
        return score_approach_ball_episode(row, protocol, result)
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
