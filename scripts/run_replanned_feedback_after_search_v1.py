"""Continue a verified live CPU search into bounded full-episode feedback checks."""
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
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['search','configuration','fixture-run','output','coordination']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.coordination.exists() or a.output.exists():raise ValueError('Fresh continuation required')
    state=read(a.search/'status.json');pid=state.get('pid');already=state['state']=='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH'
    if not already:
        command=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'search_actual_recovery_sequence_v1.py' not in command:raise ValueError('Wrong live predecessor')
    start_ticks=identity(pid) if not already else None
    command=['bash','scripts/python.sh','-u','scripts/audit_replanned_recovery_feedback_v1.py',
        '--configuration',str(a.configuration.resolve()),'--fixture-run',str(a.fixture_run.resolve()),
        '--search',str(a.search.resolve()),'--output',str(a.output.resolve())]
    files=[Path(__file__),ROOT/'scripts/audit_replanned_recovery_feedback_v1.py',a.configuration,a.search/'inputs.json']
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    a.coordination.mkdir(parents=True)
    (a.coordination/'inputs.json').write_text(json.dumps(dict(command=command,source_sha256=pins,
        search_pid=pid,search_start_ticks=start_ticks,max_wait_seconds=3000,max_child_seconds=2000),indent=2)+'\n')
    def status(name,**extra):
        f=a.coordination/'status.json';t=f.with_suffix('.tmp')
        t.write_text(json.dumps(dict(state=name,pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n');t.replace(f)
    try:
        deadline=time.monotonic()+3000.;status('WAITING_VERIFIED_SEQUENCE_SEARCH',search_pid=pid)
        while read(a.search/'status.json')['state']!='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
            if identity(pid)!=start_ticks:raise ValueError('Search ended without completion; do not restart')
            if time.monotonic()>deadline:raise TimeoutError('Search observation wait bound reached')
            time.sleep(5)
        for f,h in pins.items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed continuation input '+f)
        status('RUNNING_FULL_EPISODE_FEEDBACK_MODEL')
        with (a.coordination/'feedback.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2000)
        result=read(a.output/'result.json')
        status('COMPLETED_FEEDBACK_MODEL',result=result)
    except BaseException as e:
        status('FAILED',error=repr(e));raise


if __name__=='__main__':main()
