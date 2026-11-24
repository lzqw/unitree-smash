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
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES, probe_branches, validate_spec


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


def source_mismatch(expected, observed):
    if not isinstance(observed, dict):
        return 'Missing execution_code_sha256'
    changed = sorted(key for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key))
    return 'Different execution code: ' + ', '.join(changed) if changed else None


def captured_source_mismatch(path, expected):
    import torch
    archive = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return source_mismatch(expected, archive.get('execution_code_sha256'))


def completed_probe_hash(directory: Path, branch: str, spec: dict) -> str | None:
    receipt_path = directory / 'native_probe_manifest.json'
    probe = directory / 'native_probe.pt'
    shutdown = directory / 'shutdown_diagnostic.json'
    if (directory / 'failure.json').exists() or not all(
            path.is_file() for path in (receipt_path, probe, shutdown)):
        return None
    receipt = json.loads(receipt_path.read_text())
    diagnostic = json.loads(shutdown.read_text())
    if (receipt.get('branch') != branch or receipt.get('spec') != spec or
            not spec['tick'] + 1 <= diagnostic.get('control_ticks', 0) <= spec.get('terminal_tick_cap', spec['tick'] + 1)):
        return None
    digest = sha256(probe)
    return digest if digest == receipt.get('sha256') else None


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
    parser.add_argument('--resume', action='store_true',
                        help='Resume an existing collection and archive incomplete branch attempts')
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed used by every Isaac child. Must match the packed archive seed.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional deterministic prefix of spec files for a smoke probe run')
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent Isaac probe processes. Two is the recommended RTX 5080 setting.')
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
    if args.output.exists() and not args.resume:
        parser.error(f'Refusing to reuse probe output: {args.output}')

    spec_paths = sorted(args.spec_dir.glob('spec_*.json'))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        spec_paths = spec_paths[:args.limit]
    if not spec_paths:
        parser.error('No spec_*.json files found')
    if args.workers < 1:
        parser.error('--workers must be positive')
    specs = []
    for path in spec_paths:
        spec = json.loads(path.read_text())
        validate_spec(spec)
        specs.append((path, spec))

    args.output = args.output.resolve()
    if not args.output.exists():
        args.output.mkdir(parents=True)
    source_args = _parse_source(args)
    from quadsmash.rl.supermimic_evidence_binding_v1 import execution_code_hashes
    pinned_source = execution_code_hashes()
    startup = dict(execution_code_sha256=pinned_source,
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
        source_sha256=sha256(args.checkpoint or args.init_policy_checkpoint),
        cuda_device_count=torch.cuda.device_count(),
        started_at=time.time(),
    )
    startup_path = args.output / 'startup.json'
    if args.resume and startup_path.is_file():
        previous = json.loads(startup_path.read_text())
        if 'execution_code_sha256' in previous:
            mismatch = source_mismatch(pinned_source, previous['execution_code_sha256'])
            if mismatch:
                parser.error(mismatch)
        for field in ('spec_manifest_sha256', 'specs', 'seed', 'physics_device', 'g07_execution_root'):
            if previous.get(field) != startup[field]:
                parser.error(f'Resume collection differs in {field}')
        if 'source_sha256' in previous:
            if previous['source_sha256'] != startup['source_sha256']:
                parser.error('Resume checkpoint differs from recorded checkpoint')
        else:
            previous_argv = previous.get('argv', [])
            option, path = source_args[:2]
            if option not in previous_argv or previous_argv[previous_argv.index(option) + 1:][:1] != [path]:
                parser.error('Resume checkpoint path differs from legacy collection')
    else:
        startup_path.write_text(json.dumps(startup, indent=2) + '\n')

    completed = []
    evaluate = ROOT / 'scripts' / 'evaluate_supermimic_v1.py'
    child_env = child_environment()
    jobs = []
    for spec_path, spec in specs:
        spec_name = spec_path.stem
        for branch in probe_branches(spec):
            branch_output = args.output / spec_name / branch
            probe_path = branch_output / 'native_probe.pt'
            digest = completed_probe_hash(branch_output, branch, spec) if args.resume else None
            if digest is not None:
                mismatch = captured_source_mismatch(probe_path, pinned_source)
                if mismatch:
                    parser.error(f'{spec_name}/{branch}: {mismatch}')
                completed.append(dict(spec=spec_name, branch=branch, command=[], exit_code=0,
                                      elapsed_s=0.0, output=str(branch_output),
                                      probe_path=str(probe_path), probe_sha256=digest,
                                      resumed=True))
                continue
            if args.resume and branch_output.exists():
                suffix = 1
                archived = branch_output.with_name(f'{branch_output.name}.incomplete_{suffix}')
                while archived.exists():
                    suffix += 1
                    archived = branch_output.with_name(f'{branch_output.name}.incomplete_{suffix}')
                branch_output.rename(archived)
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
            jobs.append((spec_name, branch, branch_output, command))

    total_branches = len(jobs) + len(completed)

    spec_by_name = {path.stem: spec for path, spec in specs}

    def run_job(job):
        spec_name, branch, branch_output, command = job
        log_path = branch_output.parent / f'{branch}.launcher.log'
        started = time.time()
        mismatch = source_mismatch(pinned_source, execution_code_hashes())
        probe_path = branch_output / 'native_probe.pt'
        exit_code = None
        digest = None
        if not mismatch:
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=child_env)
            exit_code = result.returncode
            digest = completed_probe_hash(branch_output, branch, spec_by_name[spec_name])
            if digest:
                mismatch = captured_source_mismatch(probe_path, pinned_source)
        return dict(spec=spec_name, branch=branch, command=command, source_error=mismatch,
                    exit_code=exit_code, elapsed_s=time.time() - started,
                    output=str(branch_output), probe_path=str(probe_path), probe_sha256=digest)

    # Each Isaac child uses roughly 3 GB on the RTX 5080.  Keep the default
    # serial mode for compatibility, while allowing a bounded two-child run
    # to use the available GPU without competing with the learner process.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            completed.sort(key=lambda row: (row['spec'], (*BRANCHES, 'candidate').index(row['branch'])))
            (args.output / 'status.json').write_text(json.dumps(dict(
                schema='event-aligned-native-probe-collection-v1', status='RUNNING',
                workers=args.workers, total=total_branches, completed=completed), indent=2) + '\n')
            if item['source_error'] or item['exit_code'] != 0 or item['probe_sha256'] is None:
                (args.output / 'failure.json').write_text(json.dumps(item, indent=2) + '\n')
                for other in futures:
                    other.cancel()
                return 1

    status = dict(schema='event-aligned-native-probe-collection-v1', status='COLLECTION_FINISHED',
                  workers=args.workers, total=total_branches,
                  actor_updates=0, world_updates=0, critic_updates=0,
                  completed=completed, finished_at=time.time())
    (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
