"""Run one prepared 32+16 Approach speed-risk block, then freeze and check TRAIN64."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from run_supermimic_approach_ball_only import registered_options, train_command, verify_training_milestone
from run_supermimic_approach_v2 import dump, paired_quick_result, quick_check, run


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'source', 'baseline-panel', 'history-checkpoint', 'risk-history', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepared', type=Path, help='Reuse an existing verified prepared block')
    parser.add_argument('--reuse-collected', type=Path, help='Recover preparation without repeating completed native collection')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--directed-collection', type=Path)
    parser.add_argument('--directed-guard-history', type=Path)
    parser.add_argument('--directed-validation-lane', action='append', type=int, default=[])
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    execution = registered_options(read(args.registration)['command'], ('g07-execution-root',))[1]
    registered = read(args.registration)['command']
    if not (args.baseline_panel / 'completed.json').is_file():
        parser.error('Supply the completed A40 TRAIN64 baseline')
    baseline = quick_check(source, args.baseline_panel, execution, 882072)
    output.mkdir()
    cache = output.parent / '.native_replay_cache'
    if cache.is_dir():
        (output / '.native_replay_cache').symlink_to(cache.resolve(), target_is_directory=True)
    status_file = output / 'status.json'

    def command(checkpoint, directory):
        result = train_command(registered, checkpoint, directory, True, stop_at=48, fresh=False,
                             prioritize_flight=True, start_lanes=32, guard_lanes=16, e1_v2=True,
                             labeled_start_checkpoint=args.history_checkpoint, labeled_snapshot_reserve=8)
        if args.directed_collection:
            result.extend(('--approach-directed-risk', '--wm-action-collection',
                           str(args.directed_collection.resolve(strict=True))))
            for lane in args.directed_validation_lane:
                result.extend(('--approach-directed-validation-lane', str(lane)))
        return result

    started = perf_counter()
    if args.prepared:
        prepared = args.prepared.resolve(strict=True)
    else:
        preparation = output / 'prepare'
        prepared = preparation / 'approach_prepared.pt'
        dump(status_file, dict(status='preparing', source=str(source)))
        reuse = ['--approach-reuse-collected', str(args.reuse_collected.resolve(strict=True))] if args.reuse_collected else []
        history = (['--approach-directed-guard-history', str(args.directed_guard_history.resolve(strict=True))]
                   if args.directed_guard_history else [])
        run([*command(source, preparation), '--approach-event-body-refresh', '--approach-prepare-only',
             *reuse, *history],
            output / 'prepare.log', 45)
    receipt = read(prepared.parent / 'approach_prepared.json')
    with prepared.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (receipt['checkpoint_sha256'] != digest or receipt['actor_sha256'] !=
            read(args.baseline_panel / 'metrics.json')['actor_state_sha256']):
        raise ValueError('Prepared checkpoint did not preserve the A40 actor identity')
    dump(status_file, dict(status='prepared', preparation=receipt, prepared=str(prepared)))
    if args.directed_collection:
        audit = ['scripts/audit_supermimic_directed_approach.py',
                 '--collection', str(args.directed_collection), '--checkpoint', str(prepared),
                 '--output', str(output / 'prepared_directed_response.json')]
        for lane in args.directed_validation_lane:
            audit.extend(('--validation-lane', str(lane)))
        run(audit, output / 'prepared_directed_response.log', 5)

    if args.prepare_only:
        return

    train = output / 'train'
    dump(status_file, dict(status='training', preparation=receipt, prepared=str(prepared)))
    run([*command(prepared, train), '--approach-speed-risk-history', str(args.risk_history.resolve(strict=True))],
        output / 'train.log', 45)
    boundary = read(train / 'evaluation_boundary.json')
    stage = boundary['approach_accepted']
    invocation = boundary['invocation']
    if (not 40 <= stage <= 48 or invocation['actor_updates'] != stage - 40
            or invocation['world_updates'] != 0 or boundary['native_ticks'] != receipt['native_ticks']):
        raise ValueError('Actor-only risk block exceeded its budget or changed its prepared WM')
    if stage == 40:
        dump(status_file, dict(status='no_accepted_proposals', preparation=receipt, baseline=baseline,
                               invocation=invocation, wall_seconds=perf_counter() - started))
        return
    checkpoint, invocation = verify_training_milestone(train, stage, prepared)
    panel = output / f'A{stage:03d}_quick64'
    dump(status_file, dict(status='native_quick_check', preparation=receipt, candidate=str(checkpoint), invocation=invocation))
    candidate = quick_check(checkpoint, panel, execution, 882072)
    paired = paired_quick_result(baseline, candidate, args.baseline_panel, panel)
    preferred = 'candidate' if not paired['regressed'] and candidate['success'] > baseline['success'] else 'a40'
    dump(status_file, dict(status='native_regression_review' if paired['regressed'] else 'quick_check_complete',
                           preparation=receipt, invocation=invocation, baseline=baseline, candidate=candidate,
                           preferred=preferred, paired=paired, wall_seconds=perf_counter() - started))


if __name__ == '__main__':
    main()
