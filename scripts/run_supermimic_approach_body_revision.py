"""Run one bounded A32+8 Approach experiment and its paired native checks."""
import argparse
import hashlib
import json
from pathlib import Path
from subprocess import CalledProcessError
from time import perf_counter

from compare_supermimic_approach_ball_only import compare
from run_supermimic_approach_ball_only import (
    evaluate_command,
    registered_options,
    train_command,
    verify_frozen_milestone,
    verify_training_milestone,
)
from run_supermimic_approach_v2 import block_timing, dump, paired_quick_result, quick_check, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'source', 'history', 'body', 'quick-baseline', 'development-baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=882072)
    args = parser.parse_args()
    registered = json.loads(args.registration.read_text())['command']
    execution = registered_options(registered, ('g07-execution-root',))[1]
    source = args.source.resolve(strict=True)
    source_metrics = json.loads((args.development_baseline / 'metrics.json').read_text())
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source_metrics['evaluated_checkpoint_sha256']:
            raise ValueError('The fixed128 development baseline must use the exact A32 checkpoint')
    if not (args.quick_baseline / 'completed.json').is_file():
        raise ValueError('The paired TRAIN64 A32 baseline must already be complete')
    args.output.mkdir()
    shared_replay_cache = args.output.parent / '.native_replay_cache'
    if shared_replay_cache.is_dir():
        # The trainer checks both immutable archive and packed-payload digests.
        (args.output / '.native_replay_cache').symlink_to(shared_replay_cache.resolve(), target_is_directory=True)
    status = args.output / 'status.json'
    folder = args.output / 'train'
    baseline = quick_check(source, args.quick_baseline, execution, args.seed)
    command = train_command(
        registered, source, folder, True, stop_at=40, fresh=False, prioritize_flight=True,
        start_lanes=32, e1_v2=True, guard_lanes=16, labeled_start_checkpoint=args.history.resolve(strict=True),
        labeled_snapshot_reserve=8,
    )
    command.extend(('--approach-body-overlay', str(args.body.resolve(strict=True))))
    dump(status, dict(status='training', source=str(source), body=str(args.body), baseline=baseline))
    started = perf_counter()
    try:
        run(command, args.output / 'train.log', 150)
    except CalledProcessError as error:
        incomplete = folder / 'block_incomplete.json'
        detail = json.loads(incomplete.read_text()) if incomplete.is_file() else {}
        dump(status, dict(status='training_stopped', error=str(error), incomplete=detail))
        raise
    checkpoint, invocation = verify_training_milestone(folder, 40, source)
    if invocation['actor_updates'] != 8:
        raise ValueError('Expected exactly eight new accepted updates')
    timing = block_timing(folder, perf_counter() - started)
    dump(args.output / 'timing.json', timing)
    dump(status, dict(status='quick_check', checkpoint=str(checkpoint), baseline=baseline))
    quick_dir = args.output / 'A040_quick64'
    result = quick_check(checkpoint, quick_dir, execution, args.seed)
    pairing = paired_quick_result(baseline, result, args.quick_baseline, quick_dir)
    run_fixed128 = (not pairing['regressed'] and result['success'] >= baseline['success']
                    and result['frame'] <= baseline['frame'] and result['low'] <= baseline['low'])
    outcome = dict(checkpoint=str(checkpoint), native_before=baseline, native_after=result,
                   paired=pairing, fixed128_warranted=run_fixed128)
    dump(status, dict(status='fixed128' if run_fixed128 else 'quick_check_complete', **outcome))
    if run_fixed128:
        panel = args.output / 'A040_fixed128'
        run(evaluate_command(checkpoint, panel, True, execution, seed=args.seed, a0=False),
            panel.with_suffix('.log'), 40)
        outcome['fixed128'] = verify_frozen_milestone(panel, checkpoint, invocation)
        outcome['fixed128']['paired_against_A32'] = compare(args.development_baseline, panel)
        dump(status, dict(status='fixed128_complete', **outcome))


if __name__ == '__main__':
    main()
