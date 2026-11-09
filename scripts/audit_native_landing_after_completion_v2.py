"""Bounded read-only force/stability audit after an exact native runner.

Never launches physics, restarts jobs, or promotes a controller. Uses the
existing strict contact audit; partial results cannot become landing scores.
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
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    preparation = args.preparation.resolve()
    plan_path = preparation/'plan.json'
    plan = read(plan_path)
    initial = read(preparation/'status.json')
    pid = int(initial['pid'])
    started = identity(pid)
    if initial['state'] != 'COMPLETED_DIAGNOSTIC':
        cmd = [s.decode() for s in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if s]
        allowed = {'run_native_pure_control_long_v2.py', 'run_native_pure_control_long_v3.py'}
        if (started is None or not any(Path(s).name in allowed for s in cmd)
                or '--plan' not in cmd
                or Path(cmd[cmd.index('--plan')+1]).resolve() != plan_path):
            raise ValueError('Expected the exact native runner and plan')
    files = [Path(__file__), plan_path, preparation/'controller.json',
             ROOT/'scripts/audit_native_sustained_support_v1.py']
    pins = {str(f.resolve()): sha(f) for f in files}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'inputs.json').write_text(json.dumps(dict(
        source_sha256=pins, predecessor_pid=pid, predecessor_start_ticks=started,
        max_wait_seconds=3600, max_audit_seconds=600,
        physics_execution=False, training=False), indent=2)+'\n')

    def status(state, **extra):
        dest = args.output/'status.json'
        temporary = dest.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(),
            time_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n')
        temporary.replace(dest)

    try:
        status('WAITING_EXACT_NATIVE_COMPLETION', predecessor_pid=pid)
        deadline = time.monotonic()+3600
        while read(preparation/'status.json')['state'] != 'COMPLETED_DIAGNOSTIC':
            if (read(preparation/'status.json')['state'] == 'FAILED'
                    or identity(pid) != started or time.monotonic() > deadline):
                raise RuntimeError('No verified completion; no restart or score inferred')
            time.sleep(5)
        run = Path(plan['output'])
        if read(run/'job.json')['status'] != 'SUCCEEDED':
            raise ValueError('Successful full native receipt required')
        for path, digest in {**plan['source_sha256'], **pins}.items():
            if sha(path) != digest:
                raise ValueError('Changed declared input '+path)
        status('AUDITING_COMPLETE_CONTACT_AND_STABILITY')
        output = args.output/'support.json'
        command = ['bash', 'scripts/python.sh', 'scripts/audit_native_sustained_support_v1.py',
            '--run', str(run), '--audit', str(preparation/'native_audit.json'),
            '--library', read(preparation/'controller.json')['model_reference_library'],
            '--output', str(output)]
        with (args.output/'support.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ,
                CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        status('COMPLETED_NATIVE_LANDING_AUDIT', summary=read(output)['summary'],
            landing_segment_only=True, formal_acceptance=False, valid_expert_labels=False)
    except BaseException as error:
        status('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
