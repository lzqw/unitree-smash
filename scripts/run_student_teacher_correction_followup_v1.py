#!/usr/bin/env python3
"""One finite all128 assisted collection after the running controller trial.

The predecessor must exit with complete execution audits. No source changes
while it is live, no automatic model promotion, and no automatic positive
labelling or student fitting from unverified corrections.
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan)),f)
    def write(name,value):
        target=out/name;temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
    def status(state,**kw):write('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check_sources(after=False):
        for name,expected in plan['source_sha256'].items():
            if name==plan['entry']['path'] and after:expected=plan['entry']['after_sha256']
            if sha(ROOT/name)!=expected:raise ValueError('Pinned source changed: '+name)
    def run(command,log):
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (out/log).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        status('WAITING_FOR_FULL_COMMAND_TRIAL')
        deadline=time.monotonic()+plan['wait_limit_s']
        while True:
            state=json.loads((Path(plan['parent_worker'])/'status.json').read_text())
            identity=process_identity(plan['parent_pid'])
            if identity is not None and identity!=plan['parent_start_ticks']:raise ValueError('Parent PID reused')
            if state['state']=='FAILED':raise ValueError('Predecessor failed; reconcile before collection')
            if identity is None:
                if state['state']!='COMPLETED':raise ValueError('Predecessor missing without completed receipt')
                break
            if time.monotonic()>deadline:raise ValueError('Wait limit reached; no GPU job launched')
            time.sleep(5)
        previous=Path(plan['parent_output'])
        if json.loads((previous/'job.json').read_text())['status']!='SUCCEEDED' or not (previous/'completed.json').exists():
            raise ValueError('Predecessor native trial incomplete')
        audit=json.loads((Path(plan['parent_worker'])/'controller_command_audit.json').read_text())
        if not audit['passed'] or Path(audit['run']).resolve()!=previous.resolve():raise ValueError('Predecessor command audit failed')
        check_sources()
        status('INSTALLING_REVIEWED_COLLECTION_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True)
        check_sources(after=True)
        run(['bash','scripts/python.sh','scripts/check_student_teacher_native_entry_v1.py'],'entry_check.log')
        candidate=Path(plan['output'])
        if candidate.exists():raise ValueError('Preserve prior native evidence')
        status('RUNNING_ALL128_STUDENT_TEACHER',output=str(candidate))
        run(plan['command'],'guard.log');check_sources(after=True)
        status('AUDITING_ASSISTED_COLLECTION')
        for i,command in enumerate(plan['post_commands']):run(command,f'audit{i}.log')
        metrics=json.loads((candidate/'metrics.json').read_text())
        pair=json.loads((out/'paired_command_audit.json').read_text())
        runtime=json.loads((candidate/'landing_blas_runtime.json').read_text())
        if metrics['numerical_failures'] or not pair['passed']:raise ValueError('Native numerical or paired-data failure')
        if not metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']:raise ValueError('Wrong execution mode')
        if runtime['requested_threads']!=1 or any(r['after']!=1 for r in runtime['pools']):raise ValueError('BLAS setting differs')
        write('result.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],teacher_changed_ticks=pair['teacher_changed_ticks'],
            teacher_off_learning_result=False,automatic_promotion=False,positive_dataset_generated=False))
        status('COMPLETED',complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
