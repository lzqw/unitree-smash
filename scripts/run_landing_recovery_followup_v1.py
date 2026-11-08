"""One bounded serial pipeline: observed live run, CPU audits, native family trial."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'owner.json').open('x') as stream:json.dump(dict(pid=os.getpid()),stream)
    def status(state,**kw):
        path=out/'status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');tmp.replace(path)
    def check():
        for path,digest in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned source changed '+path)
    def run_cpu(command,name):
        check();status('RUNNING_CPU_AUDIT',name=name)
        with (out/name).open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,start_new_session=True,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
            try:
                code=child.wait(timeout=600)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if code:raise RuntimeError('CPU audit failed: '+name)
    try:
        check();parent=Path(plan['parent']);deadline=time.monotonic()+2400
        if identity(plan['parent_pid'])!=plan['parent_start_ticks']:raise ValueError('Live parent identity required at startup')
        status('WAITING_FOR_LIVE_NATIVE',parent_pid=plan['parent_pid'])
        while identity(plan['parent_pid'])==plan['parent_start_ticks']:
            if time.monotonic()>deadline:raise TimeoutError('Observation deadline; parent not restarted or killed')
            time.sleep(5)
        if identity(plan['parent_pid']) is not None:raise ValueError('Parent PID reused')
        if json.loads((parent/'status.json').read_text())['state']!='COMPLETED_DIAGNOSTIC':
            raise ValueError('Native observation run did not complete')
        for i,command in enumerate(plan['cpu_commands']):run_cpu(command,f'cpu_{i}.log')
        if not json.loads(Path(plan['family_check']).read_text())['passed']:
            raise ValueError('Recorded family interface check failed')
        check();status('RUNNING_NATIVE_FAMILY_TRIAL')
        # The established runner owns GPU0 exclusion, ledger, 2400s guard,
        # 102400-step cap and post-execution dispatch audit. No restart loop.
        with (out/'family.log').open('x') as stream:
            result=subprocess.run(['python3','-u','scripts/run_native_experimental_recovery_v1.py',
                '--plan',plan['family_plan']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        if result.returncode:raise RuntimeError('Family trial failed; inspect its preserved guard receipt')
        check();status('COMPLETED',valid_expert_labels=False)
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
