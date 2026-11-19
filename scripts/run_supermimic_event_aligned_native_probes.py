#!/usr/bin/env python3
"""Run every branch of declared event-aligned native probe specs.

The launcher performs no model fitting and never fabricates a native response.
Each spec is executed in a fresh Isaac process for the four branches
``nominal_a``, ``nominal_b``, ``minus`` and ``plus``.  A collection is marked
complete only when every child process exits successfully and writes an
immutable ``native_probe.pt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def kit_args_option(value: str) -> str:
    """Keep a Kit argument beginning with ``--`` as one argparse value."""
    if not value:
        raise ValueError('Kit arguments must not be empty')
    return f'--kit_args={value}'


def child_environment(existing: dict[str, str] | None = None) -> dict[str, str]:
    """Build a deterministic environment for direct Python child launchers."""
    env = dict(os.environ if existing is None else existing)
    project_path = str(ROOT)
    pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = project_path if not pythonpath else project_path + os.pathsep + pythonpath
    env['PYTHONUNBUFFERED'] = '1'
    return env


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _parse_source(args: argparse.Namespace) -> list[str]:
    if bool(args.checkpoint) == bool(args.init_policy_checkpoint):
        raise ValueError('Specify exactly one of --checkpoint and --init-policy-checkpoint')
    source = ['--checkpoint', str(args.checkpoint)] if args.checkpoint else [
        '--init-policy-checkpoint', str(args.init_policy_checkpoint),
        '--init-policy-recovery-gate', 'causal',
    ]
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--init-policy-checkpoint', type=Path)
    parser.add_argument('--spec-dir', type=Path, required=True)
    parser.add_argument('--g07-execution-root', type=Path, required=True,
                        help='Exact measured G07 execution root. No implicit worktree lookup.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed used by every Isaac child. Must match the packed archive seed.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional deterministic prefix of spec files for a smoke probe run')
    parser.add_argument('--kit-args', default='--renderer/multiGpu/enabled=false')
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        parser.error('Native event-aligned probes require a visible CUDA device')
    if not args.physics_device.startswith('cuda'):
        parser.error('Native event-aligned probes require a CUDA physics device')
    if not args.g07_execution_root.is_dir():
        parser.error(f'G07 execution root does not exist: {args.g07_execution_root}')
    if not args.spec_dir.is_dir():
        parser.error(f'Probe spec directory does not exist: {args.spec_dir}')
    if args.output.exists():
        parser.error(f'Refusing to reuse probe output: {args.output}')

    spec_paths = sorted(args.spec_dir.glob('spec_*.json'))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        spec_paths = spec_paths[:args.limit]
    if not spec_paths:
        parser.error('No spec_*.json files found')
    specs = []
    for path in spec_paths:
        spec = json.loads(path.read_text())
        validate_spec(spec)
        specs.append((path, spec))

    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    source_args = _parse_source(args)
    startup = dict(
        schema='event-aligned-native-probe-collection-v1',
        argv=sys.argv,
        cwd=os.getcwd(),
        executable=sys.executable,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        script_sha256=sha256(Path(__file__)),
        spec_dir=str(args.spec_dir.resolve()),
        spec_manifest_sha256=sha256(args.spec_dir / 'manifest.json') if (args.spec_dir / 'manifest.json').is_file() else None,
        specs=[dict(path=str(path.resolve()), sha256=sha256(path), spec=spec) for path, spec in specs],
        g07_execution_root=str(args.g07_execution_root.resolve()),
        physics_device=args.physics_device,
        seed=args.seed,
        cuda_device_count=torch.cuda.device_count(),
        started_at=time.time(),
    )
    (args.output / 'startup.json').write_text(json.dumps(startup, indent=2) + '\n')

    completed = []
    evaluate = ROOT / 'scripts' / 'evaluate_supermimic_v1.py'
    child_env = child_environment()
    for spec_path, spec in specs:
        spec_name = spec_path.stem
        for branch in BRANCHES:
            branch_output = args.output / spec_name / branch
            # The child owns creation of its unique output directory.  The
            # evaluator intentionally uses exist_ok=False so creating this
            # directory here would make every probe fail before Isaac starts.
            branch_output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(evaluate), *source_args,
                '--output', str(branch_output),
                '--g07-execution-root', str(args.g07_execution_root.resolve()),
                '--physics-device', args.physics_device,
                *(['--seed', str(args.seed)] if args.seed is not None else []),
                '--headless',
                '--native-action-probe-spec', str(spec_path.resolve()),
                '--native-action-probe-branch', branch,
                # AppLauncher receives a single value.  Using the equals
                # form is required when the Kit value itself starts with
                # ``--`` or argparse treats it as another option.
                kit_args_option(args.kit_args),
            ]
            # Keep the child output path absent until the evaluator creates it
            # with exist_ok=False.  Store the launcher log beside it.
            log_path = branch_output.parent / f'{branch}.launcher.log'
            started = time.time()
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=child_env)
            probe_path = branch_output / 'native_probe.pt'
            item = dict(spec=spec_name, branch=branch, command=command,
                        exit_code=result.returncode, elapsed_s=time.time() - started,
                        output=str(branch_output), probe_path=str(probe_path),
                        probe_sha256=sha256(probe_path) if probe_path.is_file() else None)
            completed.append(item)
            (args.output / 'status.json').write_text(json.dumps(dict(
                schema='event-aligned-native-probe-collection-v1', status='RUNNING',
                completed=completed), indent=2) + '\n')
            if result.returncode != 0 or not probe_path.is_file():
                (args.output / 'failure.json').write_text(json.dumps(item, indent=2) + '\n')
                return 1

    status = dict(schema='event-aligned-native-probe-collection-v1', status='COLLECTION_FINISHED',
                  actor_updates=0, world_updates=0, critic_updates=0,
                  completed=completed, finished_at=time.time())
    (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
