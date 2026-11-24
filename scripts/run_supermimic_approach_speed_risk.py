"""Run one prepared 32+16 Approach speed-risk block, then freeze and check TRAIN64."""
import argparse
import hashlib
import json
from pathlib import Path
from subprocess import CalledProcessError
from time import perf_counter

from run_supermimic_approach_ball_only import registered_options, train_command, verify_training_milestone
from run_supermimic_approach_v2 import dump, paired_quick_result, quick_check, run


def read(path):
    return json.loads(path.read_text())


def recover_timed_out_update(train, preparation):
    receipt = read(train / 'latest_speed_risk_actor_update.json')
    stage = receipt['retained']
    checkpoint = train / f'approach_A{stage:03d}.pt'
    invocation = receipt['invocation']
    start = invocation['start']
    if (not 40 < stage <= 48 or checkpoint.name != receipt['checkpoint']
            or receipt['native_ticks'] != preparation['native_ticks']
            or start['actor_sha256'] != preparation['actor_sha256']
            or any(start[key] != preparation[key] for key in ('attempts', 'accepted'))
            or invocation['actor_updates'] != stage - 40
            or invocation['world_updates'] or invocation['critic_updates']
            or invocation['heads']['recovery']['accepted']):
        raise ValueError('Timeout recovery receipt does not match the prepared Approach block')
    with checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['checkpoint_sha256']:
            raise ValueError('Timeout recovery checkpoint checksum changed')
    return checkpoint, invocation


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
    stop_reason = None
    try:
        run([*command(prepared, train), '--approach-speed-risk-history', str(args.risk_history.resolve(strict=True))],
            output / 'train.log', 60)
    except CalledProcessError as error:
        if error.returncode not in (124, 137, -9):
            dump(status_file, dict(status='training_failed', returncode=error.returncode, preparation=receipt))
            raise
        stop_reason = 'outer_timeout' if error.returncode == 124 else 'forced_exit'
        if not (train / 'evaluation_boundary.json').is_file():
            if not (train / 'latest_speed_risk_actor_update.json').is_file():
                dump(status_file, dict(status=stop_reason, preparation=receipt, accepted_checkpoint=None))
                raise
    if (train / 'evaluation_boundary.json').is_file():
        boundary = read(train / 'evaluation_boundary.json')
        stage = boundary['approach_accepted']
        invocation = boundary['invocation']
        stop_reason = boundary.get('stop_reason', stop_reason)
        if (not 40 <= stage <= 48 or invocation['actor_updates'] != stage - 40
                or invocation['world_updates'] != 0 or boundary['native_ticks'] != receipt['native_ticks']):
            raise ValueError('Actor-only risk block exceeded its budget or changed its prepared WM')
        if stage == 40:
            dump(status_file, dict(status='no_accepted_proposals', preparation=receipt, baseline=baseline,
                                   invocation=invocation, stop_reason=stop_reason, wall_seconds=perf_counter() - started))
            return
        checkpoint, invocation = verify_training_milestone(train, stage, prepared)
    else:
        if stop_reason is None:
            raise FileNotFoundError('Approach block exited without an evaluation boundary')
        checkpoint, invocation = recover_timed_out_update(train, receipt)
        stage = 40 + invocation['actor_updates']
    panel = output / f'A{stage:03d}_quick64'
    dump(status_file, dict(status='native_quick_check', preparation=receipt, candidate=str(checkpoint),
                           invocation=invocation, stop_reason=stop_reason))
    candidate = quick_check(checkpoint, panel, execution, 882072)
    if read(panel / 'metrics.json')['actor_state_sha256'] != invocation['actor_end_sha256']:
        raise ValueError('Native quick check did not evaluate the retained Approach actor')
    paired = paired_quick_result(baseline, candidate, args.baseline_panel, panel)
    preferred = 'candidate' if not paired['regressed'] and candidate['success'] > baseline['success'] else 'a40'
    dump(status_file, dict(status='native_regression_review' if paired['regressed'] else 'quick_check_complete',
                           preparation=receipt, invocation=invocation, baseline=baseline, candidate=candidate,
                           preferred=preferred, paired=paired, stop_reason=stop_reason,
                           wall_seconds=perf_counter() - started))


if __name__ == '__main__':
    main()
