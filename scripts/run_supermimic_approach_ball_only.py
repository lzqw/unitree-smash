"""Launch an isolated E0 or E1 experiment stage on the registered G07 inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APPROACH_BUDGET = 256
ARCHIVE_INPUTS = (
    'archive', 'packed-data', 'task-checkpoint', 'safety-mapping', 'g07-execution-root',
    'ball-checkpoint', 'physical-checkpoint', 'action-ball-checkpoint', 'task-reference-bank',
    'wm-holdout-lanes', 'wm-action-collection', 'wm-forward-guard',
)


def registered_options(command, names):
    options = []
    for name in names:
        flag = '--' + name
        for i, token in enumerate(command):
            if token == flag:
                options.extend((flag, command[i + 1]))
    missing = [name for name in names if '--' + name not in options]
    if missing:
        raise ValueError(f'Missing verified registration options: {missing}')
    return options


def evaluate_command(source, output, motion, execution_root, *, seed, a0, independent_batch=None):
    command = ['scripts/evaluate_supermimic_v1.py', '--checkpoint', str(source),
               '--output', str(output), '--g07-execution-root', str(execution_root), '--seed', str(seed),
               '--approach-ball-only', '--physics-device', 'cuda:0', '--device', 'cuda:0',
               '--headless', '--kit_args=--renderer/multiGpu/enabled=false']
    if independent_batch is None:
        command.append('--research-fixed128')
    else:
        command.extend(('--independent-panel-batch', str(independent_batch)))
    if motion and a0:
        command.extend(('--early-arm-seconds', '.25'))
    return command


def train_command(registered, source, output, motion, *, stop_at, fresh, prioritize_flight=False,
                  training_seed=None, start_lanes=16, shuffle_unlabeled_starts=False):
    command = ['scripts/train_supermimic_physical_block.py', *registered_options(registered, ARCHIVE_INPUTS),
               '--output', str(output), '--resume-training', str(source),
               '--training-profile', 'research_scale', '--training-stage', 'research_contact',
               '--contact-bptt-mode', 'enabled', '--approach-objective', 'toi_racket_only',
               '--approach-ball-only', '--approach-update-budget', str(APPROACH_BUDGET), '--approach-proposal-budget', '512',
               '--head-block-size', '8', '--native-ticks-per-actor-update', '128',
               '--world-updates-per-actor-update', '16', '--approach-contact-updates', '32',
               '--wm-batch-size', '128', '--toi-effective-start-lanes', str(start_lanes), '--tracking-weight', '1',
               '--world-learning-rate', '1e-6', '--wm-separate-supervision',
               '--wm-source-quota', 'recent_online=24,older_online=8,initial_archive=32,action_identification=26,contact_event=38',
               '--command-delta-budget', '.02', '--target-delta-budget-rad', '.08',
               '--restart-uphill-approach-momentum', '--cpu-threads', '4', '--device', 'cuda:0',
               '--rendering_mode', 'performance', '--headless', '--kit_args=--renderer/multiGpu/enabled=false']
    if stop_at is not None and stop_at != APPROACH_BUDGET:
        command.extend(('--approach-stop-at', str(stop_at)))
    if training_seed is not None:
        command.extend(('--research-training-seed', str(training_seed)))
    if shuffle_unlabeled_starts:
        command.append('--shuffle-unlabeled-approach-starts')
    if fresh:
        command.append('--resume-reset-schedule')
    if motion:
        command.extend(('--early-arm-seconds', '.25'))
    if prioritize_flight:
        speed_target = 'relative-face' if motion else 'historical'
        command.extend(('--prioritize-approach-flight', '--approach-speed-target', speed_target))
    return command


def milestone_checkpoint(run, retained):
    return run / ('latest.pt' if retained == APPROACH_BUDGET else f'approach_A{retained:03d}.pt')


def verify_training_milestone(run, retained, source):
    receipt = run/('training_completed.json' if retained == APPROACH_BUDGET else 'evaluation_boundary.json')
    result = json.loads(receipt.read_text())
    invocation = result['invocation']
    start_argv = json.loads((run/'startup.json').read_text())['argv']
    if Path(start_argv[start_argv.index('--resume-training') + 1]).resolve() != source.resolve():
        raise ValueError('Milestone was produced from a different input checkpoint')
    with (run/'training_history.jsonl').open() as stream:
        proposals = [json.loads(line) for line in stream]
    accepted = [row for row in proposals if row['update_accepted']]
    if (result['approach_accepted'] != retained or result['recovery_accepted'] != 0
            or invocation['critic_updates'] != 0 or invocation['actor_updates'] != len(accepted)
            or invocation['actor_attempts'] != len(proposals) or not accepted
            or not all(row['frozen_base_recovery_unchanged'] and row['model_unchanged_during_actor'] for row in proposals)):
        raise ValueError('Milestone evidence does not match the Approach-only training contract')
    return milestone_checkpoint(run, retained).resolve(strict=True), invocation


def verify_frozen_milestone(evaluation, checkpoint, invocation):
    from compare_supermimic_approach_ball_only import load_panel

    metrics, cases, _, completion = load_panel(evaluation)
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    score = sum(row['ball_side_complete_except_recovery'] for row in cases.values())
    if (completion['episodes'] != 128 or not metrics['actor_parameters_unchanged']
            or any(metrics['optimizer_updates'].values()) or metrics['numerical_failures']
            or metrics['actor_state_sha256'] != invocation['actor_end_sha256']
            or metrics['evaluated_checkpoint_sha256'] != checkpoint_sha
            or metrics['ball_side_complete_except_recovery'] != score):
        raise ValueError('Milestone evaluation did not verify the saved policy on all 128 cases')
    return dict(checkpoint_sha256=checkpoint_sha, actor_sha256=metrics['actor_state_sha256'],
                cases=len(cases), valid_returns=score, evaluation_seconds=metrics['elapsed_s'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path, help='Common A0 policy or this arm’s prior checkpoint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('E0', 'E1'))
    parser.add_argument('--job', required=True, choices=('a0', 'train', 'evaluate', 'milestone'))
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--prioritize-approach-flight', action='store_true')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--toi-effective-start-lanes', type=int,
                        help='Target distinct episodes per Actor proposal (default: 16)')
    parser.add_argument('--shuffle-unlabeled-approach-starts', action='store_true')
    parser.add_argument('--training-seed', type=int, help='Learner seed for a fresh training branch or its continuation')
    parser.add_argument('--baseline', type=Path, help='Pair the milestone with a completed fixed128 A0 run')
    parser.add_argument('--fixed-output', type=Path, help='Directory for the milestone’s fixed128 evaluation')
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--independent-panel-batch', type=int, choices=range(4))
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    training = args.job in ('train', 'milestone')
    if not training and (args.fresh or args.stop_at is not None or args.prioritize_approach_flight
                         or args.training_seed is not None or args.toi_effective_start_lanes is not None
                         or args.shuffle_unlabeled_approach_starts):
        parser.error('Training options require a train or milestone job')
    if args.toi_effective_start_lanes is not None and not 1 <= args.toi_effective_start_lanes <= 128:
        parser.error('The distinct-episode target must be in [1,128]')
    if training and args.independent_panel_batch is not None:
        parser.error('Independent evaluation is a separate frozen job')
    if args.stop_at is not None and (args.stop_at not in range(8, APPROACH_BUDGET + 1, 8)):
        parser.error('The Approach milestone must be a block boundary through A256')
    if (args.baseline is not None or args.fixed_output is not None) and args.job != 'milestone':
        parser.error('Baseline and fixed-output options apply to milestone evaluation')
    registered = json.loads(args.registration.read_text())['command']
    source, output = args.source.resolve(strict=True), args.output.resolve()
    early = args.arm == 'E1'
    if training:
        command = train_command(registered, source, output, early, stop_at=args.stop_at, fresh=args.fresh,
                                prioritize_flight=args.prioritize_approach_flight, training_seed=args.training_seed,
                                start_lanes=args.toi_effective_start_lanes or 16,
                                shuffle_unlabeled_starts=args.shuffle_unlabeled_approach_starts)
    else:
        execution = registered_options(registered, ('g07-execution-root',))[1]
        command = evaluate_command(source, output, early, execution, seed=args.seed, a0=args.job == 'a0',
                                   independent_batch=args.independent_panel_batch)
    environment = dict(os.environ, PYTHONPATH='.:scripts', QUADSMASH_ZSTD_ARCHIVE='1')
    if args.job != 'milestone':
        if args.print_command:
            print(json.dumps(command, ensure_ascii=False))
        else:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=environment, check=True)
        return

    retained = args.stop_at if args.stop_at is not None else APPROACH_BUDGET
    frozen = (args.fixed_output.resolve() if args.fixed_output is not None
              else output.with_name(output.name + '_fixed128'))
    if frozen == output:
        parser.error('Training and frozen evaluation require distinct directories')
    execution = registered_options(registered, ('g07-execution-root',))[1]
    evaluation = evaluate_command(milestone_checkpoint(output, retained), frozen, early, execution,
                                  seed=args.seed, a0=False)
    if args.print_command:
        print(json.dumps(dict(train=command, evaluate=evaluation), ensure_ascii=False))
        return
    if not output.exists():
        subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=environment, check=True)
    checkpoint, invocation = verify_training_milestone(output, retained, source)
    if not frozen.exists():
        subprocess.run([sys.executable, '-u', *evaluation], cwd=ROOT, env=environment, check=True)
    evidence = verify_frozen_milestone(frozen, checkpoint, invocation)
    report = dict(arm=args.arm, approach_accepted=retained, training=str(output),
                  frozen_evaluation=str(frozen), checkpoint=str(checkpoint), **evidence)
    if args.baseline is not None:
        from compare_supermimic_approach_ball_only import compare
        report['paired_against_baseline'] = compare(args.baseline, frozen)
    temp = output/'milestone_receipt.json.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(output/'milestone_receipt.json')
    print(json.dumps({key: value for key, value in report.items() if key != 'paired_against_baseline'}), flush=True)


if __name__ == '__main__':
    main()
