"""Wait for one verified live native runner, then launch one guarded observer."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text()); output = a.plan.parent
    with (output / 'owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid()), f)
    def status(state, **kw):
        path = output / 'status.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state, pid=os.getpid(),
            time=datetime.now(timezone.utc).isoformat(), **kw), indent=2) + '\n'); temp.replace(path)
    def verify():
        for path, digest in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned source changed ' + path)
    try:
        verify()
        if identity(plan['parent_pid']) != plan['parent_start_ticks']:
            raise ValueError('Expected parent is not currently live')
        deadline = time.monotonic() + 2400
        status('WAITING_FOR_VERIFIED_LIVE_RUNNER', parent_pid=plan['parent_pid'])
        while identity(plan['parent_pid']) == plan['parent_start_ticks']:
            if time.monotonic() > deadline:
                raise TimeoutError('Wait deadline; parent remains untouched and is not restarted')
            time.sleep(5)
        if identity(plan['parent_pid']) is not None:
            raise ValueError('Parent PID was reused')
        parent = json.loads((Path(plan['parent_directory']) / 'status.json').read_text())
        if parent['state'] != 'COMPLETED_DIAGNOSTIC':
            raise ValueError('Prior diagnostic failed; preserve it and do not launch another GPU job')
        verify(); status('RUNNING_PASSIVE_CONTACT_OBSERVATION')
        with (output / 'observer.log').open('x') as f:
            subprocess.run(['python3', '-u', 'scripts/run_native_experimental_recovery_v1.py',
                '--plan', plan['observer_plan']], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=True)
        verify(); status('COMPLETED', valid_expert_labels=False)
    except BaseException as error:
        status('FAILED', error=repr(error)); raise


if __name__ == '__main__':
    main()
