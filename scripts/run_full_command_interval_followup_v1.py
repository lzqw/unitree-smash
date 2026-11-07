#!/usr/bin/env python3
"""One guarded GPU0 trial after the existing queue and its audits finish.

No live source replacement, retries, automatic promotion, or task relaxation.
The original GPU guard retains the resource lock and cumulative accounting.
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


def process_start_ticks(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());output=args.plan.parent
    with (output/'queue_owner.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),plan_sha256=sha(args.plan)),stream)
    def save(name,value):
        path=output/name;temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    def status(state,**kw):
        save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def run(command,log):
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (output/log).open('x') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def check_sources(after=False):
        for name,expected in plan['required_source_sha256'].items():
            if name==plan['entry']['path'] and after:expected=plan['entry']['after_sha256']
            if sha(ROOT/name)!=expected:raise ValueError('Pinned source changed: '+name)
    try:
        status('WAITING_FOR_PARENT_AND_LEARNED_AUDIT')
        deadline=time.monotonic()+plan['parent_wait_limit_s']
        while True:
            parent=json.loads((Path(plan['parent_worker'])/'status.json').read_text())
            identity=process_start_ticks(plan['parent_pid'])
            if identity is not None and identity!=plan['parent_start_ticks']:
                raise ValueError('Predecessor PID reused; no execution')
            if parent['state']=='FAILED':raise ValueError('Parent failed; reconcile before further GPU execution')
            if identity is None:
                if parent['state']!='COMPLETED':raise ValueError('Parent missing without a terminal completion receipt')
                break
            if time.monotonic()>deadline:raise ValueError('Wait limit reached; no new GPU execution')
            time.sleep(5)
        for run_path in plan['predecessor_outputs']:
            path=Path(run_path);job=json.loads((path/'job.json').read_text())
            if job['status']!='SUCCEEDED' or not (path/'completed.json').exists():
                raise ValueError('Predecessor native run not completed')
        audit=json.loads((Path(plan['parent_worker'])/'learned_command_audit.json').read_text())
        if not audit.get('passed'):raise ValueError('Teacher-off action audit not passed')
        check_sources()
        status('INSTALLING_REVIEWED_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True)
        check_sources(after=True)
        run(['bash','scripts/python.sh','scripts/check_full_command_native_entry_v1.py'],'entry_check.log')
        candidate=Path(plan['candidate_output'])
        if candidate.exists():raise ValueError('Do not overwrite native evidence')
        status('RUNNING_FULL_COMMAND_NATIVE',output=str(candidate))
        run(plan['candidate_command'],'guard.log')
        check_sources(after=True)
        status('AUDITING_FULL_COMMAND_NATIVE')
        for index,command in enumerate(plan['post_commands']):run(command,f'audit{index}.log')
        metrics=json.loads((candidate/'metrics.json').read_text())
        control=metrics['landing_mpc']
        if not (control['full_command_damper'] and control['free_arm'] and control['interval_active_set']):
            raise ValueError('Declared full-command controller was not executed')
        if metrics['numerical_failures']:raise ValueError('Native numerical failure')
        runtime=json.loads((candidate/'landing_blas_runtime.json').read_text())
        if runtime['requested_threads']!=1 or any(x['after']!=1 for x in runtime['pools']):
            raise ValueError('Unexpected BLAS runtime')
        save('result.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],numerical_failures=metrics['numerical_failures'],
            automatic_promotion=False))
        status('COMPLETED',complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
