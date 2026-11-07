"""Serialize an independent terminal-controller diagnostic after live GPU owners.

Waiting concerns resource ownership only, not promotion of the preceding
experiment. The next run retains its own unchanged strict audit.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent;began=time.monotonic()
    def write(name,value):
        path=out/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    with (out/'queue_owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest()),f)
    try:
        write('status.json',dict(state='WAITING_FOR_GPU_OWNERS_TO_EXIT',pid=os.getpid(),
            observed_at=datetime.now(timezone.utc).isoformat(),owners=plan['wait_processes']))
        while True:
            live=[]
            for owner in plan['wait_processes']:
                actual=identity(owner['pid'])
                if actual is not None and actual!=owner['start_ticks']:raise ValueError('PID reused while waiting')
                if actual is not None:live.append(owner['pid'])
            if not live:break
            if time.monotonic()-began>plan['wait_limit_s']:raise TimeoutError('Resource wait expired')
            time.sleep(5)
        write('resource_handoff.json',dict(previous_owners_exited=True,
            preceding_experiment_promoted=False,observed_at=datetime.now(timezone.utc).isoformat()))
        code=subprocess.call(['python3','-u','scripts/run_native_frictional_diagnostic_v1.py','--plan',str(a.plan)],
            cwd=Path(__file__).resolve().parents[1])
        write('queue_completion.json',dict(exit_code=code,automatic_promotion=False))
        raise SystemExit(code)
    except (ValueError,TimeoutError) as error:
        write('status.json',dict(state='FAILED_BEFORE_LAUNCH',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':main()
