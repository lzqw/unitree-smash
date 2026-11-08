"""Run one new pinned pure-control job after a specific live predecessor.

No restart on an observation timeout. The existing GPU guard retains the
exclusive device lock, budget ledger and bounded execution for the new job.
"""
import argparse
from datetime import datetime,timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def identity(pid):
    try:
        return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text());out=a.plan.parent
    predecessor=read(a.after/'status.json');pid=int(predecessor['pid'])
    process=Path('/proc')/str(pid)
    if predecessor['state'] not in ['COMPLETED_DIAGNOSTIC','FAILED']:
        if not process.exists() or 'run_native_pure_control_long_v2.py' not in (process/'cmdline').read_bytes().replace(b'\0',b' ').decode():
            raise ValueError('Predecessor is not a verified live native runner')
    start_ticks=identity(pid)
    with (out/'serial_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),predecessor_pid=pid,predecessor_start_ticks=start_ticks,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest()),f)
    def status(state,**extra):
        path=out/'serial_status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n');tmp.replace(path)
    try:
        deadline=time.monotonic()+7200.;status('WAITING_VERIFIED_PREDECESSOR',predecessor_pid=pid)
        while True:
            state=read(a.after/'status.json')['state']
            if state=='COMPLETED_DIAGNOSTIC':break
            if state=='FAILED':raise ValueError('Predecessor failed; inspect before launching another native job')
            if identity(pid)!=start_ticks:raise ValueError('Predecessor disappeared or PID was reused without terminal receipt')
            if time.monotonic()>deadline:raise TimeoutError('Serial wait budget expired; no new job launched')
            time.sleep(5)
        prior_plan=read(a.after/'plan.json')
        if read(Path(prior_plan['output'])/'job.json')['status']!='SUCCEEDED':
            raise ValueError('Predecessor did not complete its bounded native job')
        status('STARTING_NEW_NATIVE_CONTROL',preceding_policy_promoted=False)
        with (out/'serial.log').open('x') as log:
            subprocess.run(['bash','scripts/python.sh','scripts/run_native_pure_control_long_v2.py','--plan',str(a.plan.resolve())],
                cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=6500)
        status('COMPLETED_NEW_NATIVE_DIAGNOSTIC',native_status=read(out/'status.json'))
    except BaseException as e:status('FAILED',error=repr(e));raise


if __name__=='__main__':main()
