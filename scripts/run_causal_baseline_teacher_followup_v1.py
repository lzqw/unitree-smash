#!/usr/bin/env python3
"""Run the missing original-student all128 teacher control after learning finishes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan)),f)
    def write(name,value):
        path=out/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    def status(state,**kw):write('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check(after=False):
        for name,digest in plan['source_sha256'].items():
            if after and name==plan['entry']['path']:digest=plan['entry']['after_sha256']
            if sha(ROOT/name)!=digest:raise ValueError('Pinned input changed: '+name)
    def run(command,log):
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (out/log).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        c=plan['command'];split=c.index('--');guard=c[:split];native=c[split+1:]
        if ('scripts/run_gpu_job.py' not in guard or int(guard[guard.index('--max-seconds')+1])>7200 or
                int(guard[guard.index('--env-steps-cap')+1])!=102400 or
                native[native.index('--device')+1]!='cuda:0' or
                '--landing-mpc-causal-posthit-only' not in native or '--learned-recovery-command-head' in native):
            raise ValueError('Require finite guarded original-student causal comparison on GPU0')
        check();status('WAITING_FOR_PAIRED_STUDENT_EVALUATION')
        deadline=time.monotonic()+plan['wait_limit_s'];parent=Path(plan['parent_worker'])
        while True:
            state=json.loads((parent/'status.json').read_text());live=identity(plan['parent_pid'])
            if state['pid']!=plan['parent_pid'] or (live is not None and live!=plan['parent_start_ticks']):
                raise ValueError('Parent process identity changed')
            if state['state']=='FAILED':raise ValueError('Learning pipeline failed; reconcile before next GPU run')
            if live is None:
                if state['state']!='COMPLETED':raise ValueError('Parent disappeared without completion')
                break
            if time.monotonic()>deadline:raise ValueError('Wait limit expired; no GPU run started')
            time.sleep(5)
        check()
        if not json.loads((parent/'command_audit.json').read_text())['passed']:
            raise ValueError('Parent evaluation audit failed')
        status('INSTALLING_REVIEWED_CAUSAL_BASELINE_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True)
        check(after=True)
        candidate=Path(plan['output'])
        if candidate.exists():raise ValueError('Preserve prior native evidence')
        status('RUNNING_ORIGINAL_STUDENT_WITH_ALL128_CAUSAL_TEACHER')
        run(c,'guard.log');check(after=True)
        status('AUDITING_CAUSAL_INTERVENTIONS_AND_OLD_SUCCESS_RETENTION')
        for i,command in enumerate(plan['post_commands']):run(command,f'audit{i}.log')
        metrics=json.loads((candidate/'metrics.json').read_text())
        audit=json.loads((out/'causal_command_audit.json').read_text())
        if metrics['numerical_failures'] or not audit['passed']:raise ValueError('Native validation failed')
        write('result.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],teacher_off_learning_result=False,
            automatic_promotion=False,formal_acceptance=False))
        status('COMPLETED',complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
