"""Run eight bounded Approach blocks with measured native feedback at each boundary."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from run_supermimic_approach_ball_only import registered_options, train_command, verify_training_milestone
from run_supermimic_approach_v2 import dump, quick_check, run


def execute(args):
    from run_supermimic_approach_hybrid_boundary import evaluate_boundary

    registered = json.loads(args.registration.read_text())['command']
    execution = registered_options(registered, ('g07-execution-root',))[1]
    source, holdout = args.prepared, args.baseline_panel
    best = quick_check(args.baseline, holdout, execution, 882072) | {'directory': holdout}
    excluded = list(map(int, registered_options(registered, ('wm-holdout-lanes',))[1].split(',')))
    args.output.mkdir(exist_ok=False)
    replay_cache = args.output.parent / '.native_replay_cache'
    (args.output / '.native_replay_cache').symlink_to(replay_cache.resolve(strict=True), target_is_directory=True)
    status = args.output / 'status.json'
    records, tape = [], None
    retained = 40
    completed_updates = 0
    try:
        for block in range(1, 9):
            free_gib = shutil.disk_usage(args.output).free / 2**30
            if free_gib < 22:
                raise RuntimeError(f'Hybrid disk reserve exhausted: {free_gib:.1f} GiB available')
            folder = args.output / f'block_{block:02d}'
            target = retained + 8
            command = train_command(registered, source, folder, True, stop_at=target, fresh=False,
                                    prioritize_flight=True, start_lanes=32, guard_lanes=16, e1_v2=True,
                                    labeled_start_checkpoint=args.history, labeled_snapshot_reserve=8)
            command.extend(('--approach-hybrid', '--wm-action-collection', str(args.action_collection)))
            for lane in (10, 28, 52, 78):
                command.extend(('--approach-directed-validation-lane', str(lane)))
            if tape is not None:
                command.extend(('--hybrid-native-data', str(tape)))
            dump(status, dict(status='training', block=block, completed_updates=completed_updates,
                              retained=retained, command=command, records=records, free_gib=free_gib))
            run(command, folder.with_suffix('.log'), 90)
            candidate, receipt = verify_training_milestone(folder, target, source)
            if receipt['actor_updates'] != 8:
                raise ValueError('A hybrid block must contain eight actual Approach updates')
            completed_updates += receipt['actor_updates']
            dump(status, dict(status='native_boundary', block=block, completed_updates=completed_updates,
                              checkpoint=str(candidate), records=records))
            panel, tape = evaluate_boundary(
                candidate, args.output / f'native_{block:02d}', execution, best,
                args.exploration_source, block, excluded)
            gain = panel['success'] - best['success']
            retained_actor = gain >= 0 and (gain > 0 or panel['qualified'] >= best['qualified'])
            if retained_actor:
                source, best, retained = candidate, panel, target
            records.append(dict(block=block, accepted_gradient_updates=8, candidate=str(candidate),
                                native=panel | {'directory': str(panel['directory'])}, native_gain=gain, retained_actor=retained_actor,
                                teacher_data=str(tape), retained=retained))
            # Intermediate proposal snapshot duplicates resumable block-end state.
            for name in ('pre_block.pt', 'hybrid_latest_accepted.pt'):
                (folder / name).unlink(missing_ok=True)
            dump(status, dict(status='complete' if block == 8 else 'between_blocks',
                              completed_updates=completed_updates, retained=retained, selected=str(source), records=records))
    except BaseException as error:
        dump(status, dict(status='stopped', completed_updates=completed_updates, retained=retained,
                          selected=str(source), records=records, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'registration', 'prepared', 'baseline', 'baseline-panel', 'history',
                 'action-collection', 'exploration-source', 'gpu-lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with args.gpu_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    main()
