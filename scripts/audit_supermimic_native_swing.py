"""Join measured first contacts to causal native execution windows."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING


def command_frame(row, lane):
    memory = row['executor_after']['tensors']
    return dict(
        native_tick=row['native_tick'],
        residual_request=row['requested_action'][lane].tolist(),
        mixed_before_limiter=memory['controller/mixer/last_requested'][lane].tolist(),
        mixed_after_limiter=memory['controller/mixer/last_actual'][lane].tolist(),
        executed_normalized=row['final_action'][lane].tolist(),
        executed_joint_target_rad=row['final_joint_targets'][lane].tolist(),
        arm_joint_velocity_before_rad_s=row['before']['physical']['joint_velocity'][lane, -6:].tolist(),
        arm_intervention=bool(row['arm_identification_active'][lane]),
        actor_state_sha256=row['actor_state_sha256'],
    )


def assert_close(actual, expected, label, *, tolerance=1e-6):
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise ValueError(f'Native swing evidence differs at {label}')


def checked_motion(motion, frame):
    for key, target in (('residual_request', 'residual_request'),
                        ('final_normalized_action', 'executed_normalized'),
                        ('commanded_joint_position_rad', 'executed_joint_target_rad')):
        for actual, expected in zip(motion[key], frame[target], strict=True):
            assert_close(actual, expected, key)
    face = motion['incoming_oriented_face_normal']
    terms = {name: sum(v*n for v,n in zip(motion[name], face, strict=True))
             for name in ('root_transport_mps', 'root_rotation_at_contact_mps',
                          'arm_contact_velocity_mps', 'point_velocity_mps', 'incoming_ball_velocity_mps')}
    assert_close(sum(terms[k] for k in ('root_transport_mps','root_rotation_at_contact_mps','arm_contact_velocity_mps')),
                 terms['point_velocity_mps'], 'point velocity decomposition', tolerance=5e-6)
    assert_close(terms['point_velocity_mps'] - terms['incoming_ball_velocity_mps'],
                 motion['normal_relative_velocity_at_toi'], 'normal relative velocity', tolerance=5e-6)
    return terms


def summarize(strokes, dt, minimum_speed):
    motion = [stroke['contact_projection_mps'] for stroke in strokes]
    if not motion:
        return dict(clips=0, independent_cases=0)

    def limiter_delta(stroke):
        final = stroke['precontact_commands'][-1]
        return max(abs(a-b) for a,b in zip(final['mixed_before_limiter'][-6:],
                                           final['mixed_after_limiter'][-6:], strict=True))
    gate_leads = [(s['contact_elapsed_tick']-s['arm_gate_onset_tick'])*dt for s in strokes
                  if s['arm_gate_onset_tick'] is not None]
    return dict(
        clips=len(strokes), independent_cases=len({stroke['case_id'] for stroke in strokes}),
        native_low_speed=sum(s['native_release_speed_mps'] is not None
                             and s['native_release_speed_mps'] < minimum_speed for s in strokes),
        mean_gate_lead_s=mean(gate_leads) if gate_leads else None,
        mean_limiter_delta_arm_normalized=mean(map(limiter_delta, strokes)),
        mean_face_projected_velocity_mps={key: mean(row[key] for row in motion) for key in motion[0]},
    )


def audit_run(run, scratch, width):
    archive = run/'native_transitions'
    manifest_bytes = (archive/'manifest.json').read_bytes()
    episode_bytes = (run/'native_episodes.jsonl').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['pending_transition'] or 'approach_ball_scoring' not in manifest['task_contract']:
        raise ValueError(f'Invalid native ball-only archive at {run}')
    holdout = set(json.loads((run/'config.json').read_text())['wm_holdout_lanes'])
    labeled = [json.loads(line) for line in episode_bytes.splitlines()]
    rows = [row for row in labeled if row['scoring_schema']==APPROACH_BALL_SCORING
            and not row['numerical_failures'] and row.get('first_approach_contact_motion')]
    remaining = {(row['env_id'], row['episode_index']): row for row in rows}
    if len(remaining) != len(rows):
        raise ValueError('Duplicate terminal first-contact identities')

    windows = {key: deque(maxlen=width) for key in remaining}
    strokes = []
    for chunk in manifest['chunks']:
        if not remaining:
            break
        with read_archive_rows(archive, chunk, scratch=scratch) as transitions:
            for transition in transitions:
                before, after = transition['before'], transition['after']
                for lane, episode in list(remaining):
                    if int(before['episode_index'][lane]) != episode:
                        continue
                    key = lane, episode
                    window = windows[key]
                    frame = command_frame(transition, lane)
                    if window and frame['native_tick'] != window[-1]['native_tick']+1:
                        raise ValueError('Native swing window crossed a gap in the archive')
                    window.append(frame)
                    if before['task']['had_contact'][lane] or not after['task']['had_contact'][lane]:
                        continue
                    episode_row = remaining.pop(key)
                    motion = episode_row['first_approach_contact_motion']
                    if not motion['first_racket_contact'] or (motion['env_id'], motion['episode_index']) != key:
                        raise ValueError('Joined contact event belongs to a different episode')
                    projected = checked_motion(motion, frame)
                    strokes.append(dict(
                        case_id=episode_row['case_id'], episode_index=episode, env_id=lane,
                        wm_holdout_lane=lane in holdout, arm_intervention=episode_row['arm_identification_episode'],
                        qualified_airborne_string_contact=episode_row['qualified_airborne_string_contact'],
                        native_release_speed_mps=episode_row['outgoing_speed_mps'],
                        arm_gate_onset_tick=motion['arm_gate_onset_tick'],
                        contact_elapsed_tick=int(before['elapsed_tick'][lane]),
                        arm_joint_velocity_at_toi_rad_s=motion['arm_joint_velocity_at_toi'],
                        contact_projection_mps=projected, precontact_commands=list(window),
                    ))
    if remaining:
        raise ValueError(f'Missing archived first contacts at {run}: {sorted(remaining)}')
    dt = manifest['control_dt_s']
    minimum_speed = manifest['task_contract']['approach_ball_scoring']['minimum_outgoing_speed_mps']
    if not strokes or any(len(stroke['precontact_commands']) != width for stroke in strokes):
        raise ValueError(f'Incomplete requested swing windows at {run}')
    natural = [stroke for stroke in strokes if not stroke['wm_holdout_lane'] and not stroke['arm_intervention']
               and stroke['qualified_airborne_string_contact']]
    return dict(
        source=str(run), native_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        episode_records_sha256=hashlib.sha256(episode_bytes).hexdigest(), control_dt_s=dt,
        all_contact_episodes=summarize(strokes, dt, minimum_speed),
        natural_qualified_training_subset=summarize(natural, dt, minimum_speed),
        strokes=strokes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-ticks', type=int, default=8)
    args = parser.parse_args()
    if args.window_ticks < 1:
        parser.error('Window length must be positive')
    torch.set_num_threads(2)
    result = dict(schema='supermimic-native-swing-audit-v1',
                  scope='Recorded training episodes and their first racket contacts',
                  window_ticks=args.window_ticks,
                  runs=[audit_run(run, args.scratch, args.window_ticks) for run in args.run])
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
