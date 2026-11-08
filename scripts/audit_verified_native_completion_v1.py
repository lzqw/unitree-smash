"""Read-only follow-up of one verified, bounded native control evaluation.

Waits for its existing runner, never restarts or promotes it. Final force and
stability audits require the completed native receipt, not partial progress.
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
    read = lambda p: json.loads(Path(p).read_text())
    plan = read(args.preparation/'plan.json')
    prior = read(args.preparation/'status.json')
    pid = int(prior['pid'])
    start = identity(pid)
    if prior['state'] != 'COMPLETED_DIAGNOSTIC':
        cmd = Path('/proc', str(pid), 'cmdline').read_bytes()
        if start is None or b'run_native_pure_control_long_v2.py' not in cmd:
            raise ValueError('Expected verified live native runner')
    args.output.mkdir(parents=True, exist_ok=False)
    def status(state, **extra):
        dest = args.output/'status.json'
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state, pid=os.getpid(),
            time_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n')
        tmp.replace(dest)
    try:
        status('WAITING_VERIFIED_NATIVE_COMPLETION', predecessor_pid=pid,
               predecessor_start_ticks=start)
        deadline = time.monotonic()+7200
        while True:
            state = read(args.preparation/'status.json')['state']
            if state == 'COMPLETED_DIAGNOSTIC':
                break
            if state == 'FAILED' or identity(pid) != start or time.monotonic() > deadline:
                raise RuntimeError('No valid native completion; no result inferred')
            time.sleep(5)
        run = Path(plan['output'])
        if read(run/'job.json')['status'] != 'SUCCEEDED':
            raise ValueError('Successful native receipt required')
        for path, digest in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('Changed declared experiment input '+path)
        cfg = read(args.preparation/'controller.json')
        status('AUDITING_COMPLETE_NATIVE_EVIDENCE')
        summaries = {}
        for name, script in [('support', 'audit_native_sustained_support_v1.py'),
                             ('reference', 'audit_native_reference_execution_v1.py')]:
            output = args.output/(name+'.json')
            command = ['bash', 'scripts/python.sh', 'scripts/'+script,
                '--run', str(run), '--audit', str(args.preparation/'native_audit.json'),
                '--library', cfg['model_reference_library'], '--output', str(output)]
            with (args.output/(name+'.log')).open('x') as log:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ,
                    CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            summaries[name] = read(output)['summary']
        status('COMPLETED_READ_ONLY_AUDIT', summaries=summaries,
               native_success_not_inferred_from_model=True, training=False)
    except BaseException as error:
        status('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
