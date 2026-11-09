"""Bounded CPU feedback and end-state check after this exact margin search.

No restart, automatic native launch, policy promotion or expert labels.
"""
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
    for name in ['search','configuration','run','output','coordination']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or a.coordination.exists():raise ValueError('Fresh continuation outputs required')
    current=read(a.search/'status.json');pid=current.get('pid');started=None
    terminal='COMPLETED_FULL_RECOVERY_MARGIN_SEARCH'
    if current['state']!=terminal:
        cmd=[s.decode() for s in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0') if s]
        if ('scripts/search_full_recovery_margin_v1.py' not in cmd or
            Path(cmd[cmd.index('--output')+1]).resolve()!=a.search.resolve()):
            raise ValueError('Expected the exact live margin search')
        started=identity(pid)
        if started is None:raise ValueError('Predecessor identity missing')
    files=[Path(__file__),ROOT/'scripts/search_full_recovery_margin_v1.py',ROOT/'scripts/full_recovery_margin_rollout_v1.py',
        ROOT/'scripts/audit_full_margin_feedback_v1.py',ROOT/'scripts/audit_feedback_end_observation_v2.py',a.search/'inputs.json',a.configuration]
    pins={str(f.resolve()):sha(f) for f in files}
    a.coordination.mkdir(parents=True)
    def status(state,**extra):
        f=a.coordination/'status.json';temporary=f.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state,pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n');temporary.replace(f)
    (a.coordination/'inputs.json').write_text(json.dumps(dict(source_sha256=pins,search_pid=pid,search_start_ticks=started,
        max_wait_seconds=3300,max_feedback_seconds=2150,max_end_check_seconds=120,native_execution=False,training=False),indent=2)+'\n')
    try:
        status('WAITING_EXACT_MARGIN_SEARCH',search_pid=pid);deadline=time.monotonic()+3300
        while read(a.search/'status.json')['state']!=terminal:
            if identity(pid)!=started:raise ValueError('Search disappeared without completion; not restarting')
            if time.monotonic()>deadline:raise TimeoutError('Bounded wait expired; not restarting search')
            time.sleep(5)
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed continuation input '+f)
        result=read(a.search/'result.json')
        if not any(r['complete_stable_model'] and r['improved'] and r['best_minimum_nonfoot_m']>r['baseline_minimum_nonfoot_m']+1e-6 for r in result['results']):
            status('COMPLETED_NO_MARGIN_IMPROVEMENT',search_result=result);return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        command=['bash','scripts/python.sh','scripts/audit_full_margin_feedback_v1.py','--search',str(a.search.resolve()),
            '--configuration',str(a.configuration.resolve()),'--run',str(a.run.resolve()),'--output',str(a.output.resolve())]
        status('RUNNING_IMPROVED_PLAN_FEEDBACK')
        with (a.coordination/'feedback.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2150)
        end=a.coordination/'end_observation.json'
        command=['bash','scripts/python.sh','scripts/audit_feedback_end_observation_v2.py','--search',str(a.search.resolve()),
            '--feedback',str(a.output.resolve()),'--output',str(end.resolve())]
        with (a.coordination/'end.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        status('COMPLETED_MARGIN_FEEDBACK_AND_END_CHECK',result=read(a.output/'result.json'),
            end_observation=read(end)['results'],native_success=False,valid_expert_labels=False)
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
