"""Index measured contact-to-reset clips in the ball-only native training archives."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING


def index_training_run(run, scratch):
    archive = run/'native_transitions'
    manifest_bytes = (archive/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    episode_bytes = (run/'native_episodes.jsonl').read_bytes()
    episodes = [json.loads(line) for line in episode_bytes.splitlines()]
    if (not episodes or any(row['scoring_schema'] != APPROACH_BALL_SCORING for row in episodes)
            or 'approach_ball_scoring' not in manifest['task_contract']):
        raise ValueError(f'Expected a ball-only training archive at {run}')
    eligible = [row for row in episodes if row['approach_ball_terminal']['recovery_tail_observed']
                and not row['approach_ball_terminal']['numerical_abort'] and not row['numerical_failures']]
    clips = {(row['env_id'], row['episode_index']): dict(case_id=row['case_id'], env_id=row['env_id'],
                episode_index=row['episode_index'], arm_intervention=row['arm_identification_episode'],
                ball_completed=row['ball_side_complete_except_recovery'],
                physical_recovery_success=row['physical_recovery_success'], contact=None, release=None,
                terminal=None, control_steps=0, chunks=[]) for row in eligible}
    if len(clips) != len(eligible):
        raise ValueError(f'Duplicate native episode in {run}')

    scanned = 0
    last_tick = None
    previous_ticks = {}
    for chunk in manifest['chunks']:
        with read_archive_rows(archive, chunk, scratch=scratch) as rows:
            for offset, row in enumerate(rows):
                tick = row['native_tick']
                if last_tick is not None and tick != last_tick + 1:
                    raise ValueError(f'Noncontiguous native archive at tick {tick}')
                last_tick = tick
                scanned += 1
                before, after = row['before'], row['after']
                pointer = dict(native_tick=tick, file=chunk['file'], row_offset=offset)
                for identity, clip in clips.items():
                    lane, episode = identity
                    if clip['terminal'] is not None or int(before['episode_index'][lane]) != episode:
                        continue
                    if clip['contact'] is None:
                        if bool(before['task']['had_contact'][lane]) or not bool(after['task']['had_contact'][lane]):
                            continue
                        clip['contact'] = pointer
                    if identity in previous_ticks and tick != previous_ticks[identity] + 1:
                        raise ValueError(f'Noncontiguous recovery tail for {identity}')
                    previous_ticks[identity] = tick
                    if int(after['episode_index'][lane]) != episode:
                        raise ValueError(f'Recovery sample was captured after reset for {identity}')
                    if clip['release'] is None and (not bool(before['task']['released'][lane])
                                                    and bool(after['task']['released'][lane])):
                        clip['release'] = pointer
                    for key in ('root_position_w', 'root_quaternion_w', 'joint_position', 'joint_velocity'):
                        if not torch.isfinite(after['physical'][key][lane]).all():
                            raise ValueError(f'Nonfinite body state for {identity} at tick {tick}')
                    if not (row['executor_before']['tensors'] and row['executor_after']['tensors']
                            and torch.isfinite(row['final_joint_targets'][lane]).all()):
                        raise ValueError(f'Missing actuator history for {identity} at tick {tick}')
                    clip['control_steps'] += 1
                    if not clip['chunks'] or clip['chunks'][-1]['file'] != chunk['file']:
                        clip['chunks'].append(dict(file=chunk['file'], sha256=chunk['sha256']))
                    if bool(row['terminated'][lane] or row['timeout'][lane]):
                        clip['terminal'] = pointer
    if scanned != manifest['control_ticks']:
        raise ValueError(f'Native transition count changed in {run}')
    missing = [identity for identity, clip in clips.items() if clip['terminal'] is None]
    if missing:
        raise ValueError(f'No complete pre-reset native clip for {missing} in {run}')
    return dict(training_run=str(run), archive=str(archive), control_dt_s=manifest['control_dt_s'],
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                episode_records_sha256=hashlib.sha256(episode_bytes).hexdigest(),
                completed_episode_records=len(episodes), clips=list(clips.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runs = [index_training_run(path.resolve(), args.scratch) for path in args.run]
    report = dict(schema='supermimic-stage2-recovery-tail-index-v1',
                  scope='Measured training episodes with a completed recovery tail; duplicate cases across runs are repeated observations.',
                  clips=sum(len(run['clips']) for run in runs),
                  clips_with_release=sum(clip['release'] is not None for run in runs for clip in run['clips']), runs=runs)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
