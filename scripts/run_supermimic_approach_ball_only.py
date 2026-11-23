"""Launch an isolated E0 or E1 experiment stage on the registered G07 inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
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


def train_command(registered, source, output, motion, *, stop_at, fresh, prioritize_flight=False):
    command = ['scripts/train_supermimic_physical_block.py', *registered_options(registered, ARCHIVE_INPUTS),
               '--output', str(output), '--resume-training', str(source),
               '--training-profile', 'research_scale', '--training-stage', 'research_contact',
               '--contact-bptt-mode', 'enabled', '--approach-objective', 'toi_racket_only',
               '--approach-ball-only', '--approach-update-budget', '256', '--approach-proposal-budget', '512',
               '--head-block-size', '8', '--native-ticks-per-actor-update', '128',
               '--world-updates-per-actor-update', '16', '--approach-contact-updates', '32',
               '--wm-batch-size', '128', '--toi-effective-start-lanes', '16', '--tracking-weight', '1',
               '--world-learning-rate', '1e-6', '--wm-separate-supervision',
               '--wm-source-quota', 'recent_online=24,older_online=8,initial_archive=32,action_identification=26,contact_event=38',
               '--command-delta-budget', '.02', '--target-delta-budget-rad', '.08',
               '--restart-uphill-approach-momentum', '--cpu-threads', '4', '--device', 'cuda:0',
               '--rendering_mode', 'performance', '--headless', '--kit_args=--renderer/multiGpu/enabled=false']
    if stop_at is not None:
        command.extend(('--approach-stop-at', str(stop_at)))
    if fresh:
        command.append('--resume-reset-schedule')
    if motion:
        command.extend(('--early-arm-seconds', '.25'))
    if prioritize_flight:
        command.append('--prioritize-approach-flight')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path, help='Common A0 policy or this arm’s prior checkpoint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('E0', 'E1'))
    parser.add_argument('--job', required=True, choices=('a0', 'train', 'evaluate'))
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--prioritize-approach-flight', action='store_true')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--independent-panel-batch', type=int, choices=range(4))
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    if args.job != 'train' and (args.fresh or args.stop_at is not None or args.prioritize_approach_flight):
        parser.error('A training boundary and optimizer reset apply only to train')
    if args.job == 'train' and args.independent_panel_batch is not None:
        parser.error('Independent evaluation is a separate frozen job')
    registered = json.loads(args.registration.read_text())['command']
    source, output = args.source.resolve(strict=True), args.output.resolve()
    early = args.arm == 'E1'
    if args.job == 'train':
        command = train_command(registered, source, output, early, stop_at=args.stop_at, fresh=args.fresh,
                                prioritize_flight=args.prioritize_approach_flight)
    else:
        execution = registered_options(registered, ('g07-execution-root',))[1]
        command = evaluate_command(source, output, early, execution, seed=args.seed, a0=args.job == 'a0',
                                   independent_batch=args.independent_panel_batch)
    if args.print_command:
        print(json.dumps(command, ensure_ascii=False))
        return
    environment = dict(os.environ, PYTHONPATH='.:scripts', QUADSMASH_ZSTD_ARCHIVE='1')
    subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
