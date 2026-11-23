"""Pair frozen ball-only scores from identical case and initial-state panels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median

from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING


def load_panel(directory):
    completion = json.loads((directory / 'completed.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    initial = json.loads((directory / 'initial_states.json').read_text())
    rows = json.loads((directory / 'episodes_summary.json').read_text())
    cases = {row['case_id']: row for row in rows}
    if (len(cases) != 128 or len(cases) != len(rows)
            or metrics['scoring_schema'] != APPROACH_BALL_SCORING
            or any(row['scoring_schema'] != APPROACH_BALL_SCORING or row['episode_index'] != 0 for row in rows)):
        raise ValueError('Expected all first episodes under the ball-only scoring contract')
    return metrics, cases, initial, completion


def native_records(directory):
    def read_jsonl(path):
        if not path.is_file():
            return []
        with path.open() as stream:
            return [json.loads(row) for row in stream]

    raw = read_jsonl(directory / 'episodes.jsonl')
    motion_file = directory / 'native_contact_motion.json'
    if motion_file.is_file():
        motion = json.loads(motion_file.read_text())
    else:
        motion = [row['first_approach_contact_motion'] for row in raw if row.get('first_approach_contact_motion')]
    return raw, motion


def native_breakdown(raw, motion, cases):
    panel_episodes = {(row['env_id'], row['episode_index']) for row in cases.values()}
    raw = [row for row in raw if (row['env_id'], row['episode_index']) in panel_episodes]
    qualified = [row for row in cases.values() if row['qualified_airborne_string_contact']]
    speed = sorted(row['outgoing_speed_mps'] for row in qualified if row['outgoing_speed_mps'] is not None)
    qualified_episodes = {(row['env_id'], row['episode_index']) for row in qualified}
    motion = [row for row in motion if row['first_racket_contact']
              and (row['env_id'], row['episode_index']) in qualified_episodes]
    arm = [math.hypot(*row['arm_contact_velocity_mps']) for row in motion if 'arm_contact_velocity_mps' in row]
    onset = [row['arm_gate_onset_tick'] for row in motion if row.get('arm_gate_onset_tick') is not None]
    effective = [row['normal_relative_velocity_at_toi'] for row in motion
                 if math.isfinite(row.get('normal_relative_velocity_at_toi', math.nan))]
    components = {}
    for field in ('root_transport_mps', 'root_rotation_at_contact_mps', 'arm_contact_velocity_mps'):
        along_normal = [sum(v * n for v, n in zip(row[field], row['incoming_oriented_face_normal'], strict=True))
                        for row in motion if field in row and 'incoming_oriented_face_normal' in row]
        along_normal = [value for value in along_normal if math.isfinite(value)]
        components[field] = dict(cases=len(along_normal), mean_mps=mean(along_normal) if along_normal else None)
    return dict(qualified_releases_measured=len(speed),
                qualified_release_speed_mps=(dict(minimum=speed[0], median=median(speed), average=mean(speed),
                    maximum=speed[-1]) if speed else None),
                qualified_below_speed_threshold_rate=(sum('speed_threshold' in row['rejection_reasons']
                    and row['outgoing_speed_mps'] is not None for row in qualified) / len(speed) if speed else None),
                cleared_net=sum(row['crossed_net'] for row in raw) if raw else None,
                target_landings=sum(row['landed_target'] for row in raw) if raw else None,
                frame_without_strings=sum(row['frame_hit'] and not row['string_hit'] for row in raw) if raw else None,
                no_racket_contact=sum(not row['had_contact'] for row in raw) if raw else None,
                matched_contact_motion=len(motion), arm_gate_onset_median_tick=median(onset) if onset else None,
                arm_contact_speed_median_mps=median(arm) if arm else None,
                normal_relative_speed_median_mps=median(effective) if effective else None,
                components_along_incoming_face_normal=components)


def paired_contact_motion(before, after, cases_before, cases_after, common_releases):
    def first_contacts(rows):
        contacts = {}
        for row in rows:
            if row['first_racket_contact']:
                identity = (row['env_id'], row['episode_index'])
                if identity in contacts:
                    raise ValueError(f'Duplicate first contact for lane and episode {identity}')
                contacts[identity] = row
        return contacts

    def projection(vector, normal):
        return sum(v * n for v, n in zip(vector, normal, strict=True))

    base, revised = first_contacts(before), first_contacts(after)
    changes = []
    for identity in common_releases:
        a, b = cases_before[identity], cases_after[identity]
        old = base.get((a['env_id'], a['episode_index']))
        new = revised.get((b['env_id'], b['episode_index']))
        if old is None or new is None or not all(
                row['string_contact'] and row['legal_initial_contact'] for row in (old, new)):
            continue
        normal = new.get('incoming_oriented_face_normal')
        row = dict(case_id=identity, speed_delta_mps=b['outgoing_speed_mps'] - a['outgoing_speed_mps'],
                   first_contact_time_delta_ms=1000 * (new['time_s'] - old['time_s']))
        if 'arm_contact_velocity_mps' in old and 'arm_contact_velocity_mps' in new:
            row['arm_contact_speed_delta_mps'] = (math.hypot(*new['arm_contact_velocity_mps'])
                                                  - math.hypot(*old['arm_contact_velocity_mps']))
        if normal is not None:
            for component in ('point_velocity_mps', 'root_transport_mps', 'root_rotation_at_contact_mps',
                              'arm_contact_velocity_mps'):
                if component in old and component in new:
                    row[f'{component}_delta_on_after_face'] = (projection(new[component], normal)
                                                                - projection(old[component], normal))
        if 'normal_relative_velocity_at_toi' in old and 'normal_relative_velocity_at_toi' in new:
            row['normal_relative_velocity_delta_mps'] = (new['normal_relative_velocity_at_toi']
                                                          - old['normal_relative_velocity_at_toi'])
        if any(not math.isfinite(value) for name, value in row.items() if name != 'case_id'):
            raise ValueError(f'Nonfinite paired contact motion for {identity}')
        changes.append(row)

    samples = {}
    for row in changes:
        for name, value in row.items():
            if name != 'case_id':
                samples.setdefault(name, []).append(value)
    return dict(matched_first_legal_string_contacts=len(changes),
                projection_axis='After-policy face normal oriented toward the incoming shuttle',
                summary={name: dict(cases=len(values), mean=mean(values), median=median(values))
                         for name, values in samples.items()}, cases=changes)


def compare(before, after, *, allow_early_arm_change=False):
    start, a, init_a, _ = load_panel(before)
    end, b, init_b, _ = load_panel(after)
    if a.keys() != b.keys() or init_a != init_b or start['task_contract'] != end['task_contract']:
        raise ValueError('Case, initial, physics or scoring contracts differ')
    for name in ('physics.json', 'actuator_contract.json'):
        if json.loads((before / name).read_text()) != json.loads((after / name).read_text()):
            raise ValueError(f'Paired evaluation requires identical {name}')
    execution_a, execution_b = (x['policy_execution'] for x in (start, end))
    if allow_early_arm_change:
        if start['actor_state_sha256'] != end['actor_state_sha256']:
            raise ValueError('Zero-update gate comparison requires identical Actor weights')
        execution_a, execution_b = ({k:v for k,v in x.items() if k != 'arm_preparation_seconds'}
                                    for x in (execution_a, execution_b))
    if execution_a != execution_b:
        raise ValueError('Unexpected policy execution change')
    key = 'ball_side_complete_except_recovery'
    gains = [case for case in a if not a[case][key] and b[case][key]]
    losses = [case for case in a if a[case][key] and not b[case][key]]
    common_releases = [case for case in a if all(
        rows[case]['qualified_airborne_string_contact'] and rows[case]['outgoing_speed_mps'] is not None
        for rows in (a, b))]
    before_speeds = [a[case]['outgoing_speed_mps'] for case in common_releases]
    after_speeds = [b[case]['outgoing_speed_mps'] for case in common_releases]
    speed_changes = [after_speed - before_speed
                     for before_speed, after_speed in zip(before_speeds, after_speeds, strict=True)]
    flips = len(gains) + len(losses)
    probability = min(1., 2 * sum(math.comb(flips, i) for i in range(min(len(gains), len(losses)) + 1)) / 2**flips)
    raw_before, motion_before = native_records(before)
    raw_after, motion_after = native_records(after)

    def outcomes(raw, motion, rows, metrics):
        failures = Counter()
        for row in rows.values():
            if not row[key]:
                failures.update(row['rejection_reasons'])
        count = sum(row[key] for row in rows.values())
        qualified = sum(row['qualified_airborne_string_contact'] for row in rows.values())
        wall = metrics.get('elapsed_s')
        return dict(ball_success=count, ball_success_rate=count / len(rows), qualified_contacts=qualified,
                    success_given_qualified=count / qualified if qualified else None,
                    qualified_low_speed=sum(row['qualified_airborne_string_contact']
                                            and row['outgoing_speed_mps'] is not None
                                            and 'speed_threshold' in row['rejection_reasons'] for row in rows.values()),
                    failure_reasons=dict(failures), wall_seconds=wall,
                    independent_qualified_contacts_per_minute=60 * qualified / wall if wall else None,
                    completed_ball_episodes_per_hour=3600 * len(rows) / wall if wall else None,
                    **native_breakdown(raw, motion, rows))

    return dict(scoring_schema=APPROACH_BALL_SCORING, cases=len(a),
                before=outcomes(raw_before, motion_before, a, start),
                after=outcomes(raw_after, motion_after, b, end), improved_cases=gains, regressed_cases=losses,
                paired_contact_motion=paired_contact_motion(motion_before, motion_after, a, b, common_releases),
                paired_qualified_release_speed=dict(cases=len(common_releases),
                    mean_delta_mps=mean(speed_changes) if speed_changes else None,
                    median_delta_mps=median(speed_changes) if speed_changes else None,
                    below_threshold_to_pass=[case for case in common_releases
                        if 'speed_threshold' in a[case]['rejection_reasons'] and 'speed_threshold' not in b[case]['rejection_reasons']],
                    pass_to_below_threshold=[case for case in common_releases
                        if 'speed_threshold' not in a[case]['rejection_reasons'] and 'speed_threshold' in b[case]['rejection_reasons']]),
                paired_exact_two_sided_p=probability)


def compare_learning(control_before, control_after, early_before, early_after):
    gates = compare(control_before, early_before, allow_early_arm_change=True)
    control = compare(control_before, control_after)
    early = compare(early_before, early_after)
    panels = [load_panel(path)[1] for path in (control_before, control_after, early_before, early_after)]
    key = 'ball_side_complete_except_recovery'
    case_changes = Counter()
    for identity in panels[0]:
        c0, c1, e0, e1 = (int(panel[identity][key]) for panel in panels)
        case_changes[(e1 - e0) - (c1 - c0)] += 1
    return dict(gate_effect_at_zero=gates, E0_learning=control, E1_learning=early,
        E0_success_rate_delta=control['after']['ball_success_rate'] - control['before']['ball_success_rate'],
        E1_success_rate_delta=early['after']['ball_success_rate'] - early['before']['ball_success_rate'],
        E1_minus_E0_learning_rate_delta=sum(change * n for change, n in case_changes.items()) / len(panels[0]),
        case_level_E1_minus_E0_change_counts=dict(case_changes))


def compare_revision(before, original_after, revised_after):
    """Identify which baseline successes a same-gate revision restores or loses."""
    original = compare(before, original_after)
    revised = compare(before, revised_after)
    paired = compare(original_after, revised_after)
    original_gains = set(original['improved_cases'])
    original_losses = set(original['regressed_cases'])
    revised_gains = set(revised['improved_cases'])
    revised_losses = set(revised['regressed_cases'])
    _, original_cases, _, _ = load_panel(original_after)
    _, revised_cases, _, _ = load_panel(revised_after)
    recovered = sorted(original_losses - revised_losses)
    new_losses = sorted(revised_losses - original_losses)
    return dict(original_vs_a0=original, revised_vs_a0=revised, revised_vs_original=paired,
                recovered_baseline_successes=recovered,
                persistent_baseline_losses=sorted(original_losses & revised_losses),
                newly_lost_baseline_successes=new_losses,
                retained_original_gains=sorted(original_gains & revised_gains),
                lost_original_gains=sorted(original_gains - revised_gains),
                additional_gains_over_a0=sorted(revised_gains - original_gains),
                recovered_failure_reasons=dict(Counter(
                    reason for case in recovered for reason in original_cases[case]['rejection_reasons'])),
                new_loss_failure_reasons=dict(Counter(
                    reason for case in new_losses for reason in revised_cases[case]['rejection_reasons'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-early-arm-change', action='store_true')
    parser.add_argument('--control-before', type=Path)
    parser.add_argument('--control-after', type=Path)
    parser.add_argument('--original-after', type=Path,
                        help='Compare two post-training variants against the same zero-update arm')
    args = parser.parse_args()
    if (args.control_before is None) != (args.control_after is None):
        parser.error('Supply both control panels to compare learned changes')
    if args.original_after is not None:
        if args.control_before is not None or args.allow_early_arm_change:
            parser.error('A revision comparison requires one fixed execution contract')
        report = compare_revision(args.before, args.original_after, args.after)
    elif args.control_before is not None:
        if args.allow_early_arm_change:
            parser.error('Each learning arm must retain its own A0 gate')
        report = compare_learning(args.control_before, args.control_after, args.before, args.after)
    else:
        report = compare(args.before, args.after, allow_early_arm_change=args.allow_early_arm_change)
    staged = args.output.with_suffix('.tmp')
    staged.write_text(json.dumps(report, indent=2) + '\n')
    staged.replace(args.output)


if __name__ == '__main__':
    main()
