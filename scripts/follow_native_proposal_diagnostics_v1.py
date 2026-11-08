"""Observe one existing native process, then run only read-only CPU diagnostics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'owner.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
    def status(state,**kw):
        path=out/'status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');tmp.replace(path)
    def check():
        for path,h in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Pinned diagnostic source changed '+path)
    try:
        check();deadline=time.monotonic()+plan['wait_limit_s'];parent=Path(plan['parent'])
        if identity(plan['parent_pid'])!=plan['parent_start_ticks']:raise ValueError('Expected live parent identity required')
        status('WAITING_FOR_EXISTING_NATIVE_RUN',parent_pid=plan['parent_pid'])
        while True:
            current=identity(plan['parent_pid'])
            if current is not None and current!=plan['parent_start_ticks']:raise ValueError('Parent PID reused')
            state=json.loads((parent/'status.json').read_text())
            if current is None:
                if state['state']!='COMPLETED_DIAGNOSTIC':raise ValueError('Native pipeline ended without completed diagnostic: '+state['state'])
                break
            if time.monotonic()>deadline:raise ValueError('Observation limit reached; native process was not stopped or restarted')
            time.sleep(5)
        for index,command in enumerate(plan['commands']):
            check();status('RUNNING_CPU_DIAGNOSTIC',index=index)
            with (out/f'command_{index}.log').open('x') as f:
                subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                    stdout=f,stderr=subprocess.STDOUT,check=True)
        check();status('COMPLETED_DIAGNOSTICS',valid_expert_labels=False)
    except BaseException as error:status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
