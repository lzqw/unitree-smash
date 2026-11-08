"""Run one prepared diagnostic after a verified fifth-level deferred native job.

Waits for a verified existing deferred owner and successful native receipt. No
retry, promotion, overlapping GPU launch, or mutation of predecessor inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def identity(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent = read(args.after/'deferred_owner.json')
    pid = int(parent['pid'])
    arguments = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    arguments = [value.decode() for value in arguments if value]
    if ('scripts/run_after_deferred_native_v5.py' not in arguments or
            Path(arguments[arguments.index('--plan')+1]).resolve() != (args.after/'plan.json').resolve()):
        raise ValueError('Expected the exact live deferred owner of the predecessor')
    start_ticks = identity(pid)
    if start_ticks is None:
        raise ValueError('Predecessor owner disappeared')
    output = args.plan.parent
    if (output/'serial_owner.json').exists() or (output/'status.json').exists():
        raise ValueError('Candidate already queued or running')
    pins = {str(p.resolve()):sha(p) for p in [Path(__file__), args.plan, args.after/'plan.json', args.after/'deferred_owner.json', ROOT/'scripts/run_after_deferred_native_v5.py',
        ROOT/'scripts/run_serial_pure_control_current_v4.py']}
    if parent['source_sha256'].get(str((args.after/'plan.json').resolve())) != sha(args.after/'plan.json'):
        raise ValueError('Existing queued input changed')
    with (output/'deferred_owner.json').open('x') as file:
        json.dump(dict(pid=os.getpid(), predecessor_deferred_pid=pid,
            predecessor_deferred_start_ticks=start_ticks, source_sha256=pins,
            max_wait_seconds=48000, max_child_seconds=8400), file, indent=2)
    def status(state, **extra):
        path = output/'deferred_status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(),
            time_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n')
        temporary.replace(path)
    try:
        status('WAITING_QUEUED_NATIVE_COMPLETION', predecessor_deferred_pid=pid)
        deadline = time.monotonic()+48000.
        while True:
            native_file = args.after/'status.json'
            native = read(native_file) if native_file.exists() else {}
            if native.get('state') == 'COMPLETED_DIAGNOSTIC':
                break
            serial = read(args.after/'deferred_status.json')
            if native.get('state') == 'FAILED' or serial['state'] == 'FAILED':
                raise ValueError('Predecessor failed; no automatic retry or launch')
            if identity(pid) != start_ticks:
                raise ValueError('Predecessor deferred owner disappeared without completion')
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded queued wait expired; no candidate launched')
            time.sleep(5)
        prior = read(args.after/'plan.json')
        if read(Path(prior['output'])/'job.json')['status'] != 'SUCCEEDED':
            raise ValueError('No successful native completion receipt')
        for filename,digest in pins.items():
            if sha(filename) != digest:
                raise ValueError('Changed deferred input '+filename)
        status('DELEGATING_ONE_NATIVE_DIAGNOSTIC', preceding_policy_promoted=False)
        with (output/'deferred.log').open('x') as log:
            subprocess.run(['bash', 'scripts/python.sh', '-u',
                'scripts/run_serial_pure_control_current_v4.py', '--after', str(args.after.resolve()),
                '--plan', str(args.plan.resolve())], cwd=ROOT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=8400)
        status('COMPLETED_DEFERRED_NATIVE_DIAGNOSTIC', result=read(output/'status.json'))
    except BaseException as error:
        status('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
