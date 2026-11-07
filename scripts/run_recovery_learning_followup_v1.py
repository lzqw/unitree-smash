#!/usr/bin/env python3
"""Finite serial queue: replicate the controller, then evaluate a learned head.

The entry patch is installed only after the unchanged controller run exits.
All GPU jobs still use the project guard, budgets and native episode receipts.
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());output=args.plan.parent
    if (output/'status.json').exists():raise ValueError('Do not launch the queue twice')
    def save(name,value):
        path=output/name;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    def status(state,**kw):
        save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check_sources():
        for name,expected in plan['required_source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Pinned source changed: '+name)
    def run(command,log):
        env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        with (output/log).open('x') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def paired(before,after,filename):
        run(['bash','scripts/python.sh','scripts/compare_frozen_complete5_v2.py',
            '--before',before,'--after',after,'--output',str(output/filename)],filename+'.log')
    try:
        status('WAITING_FOR_IMPLICIT_COMMAND_TRIAL')
        deadline=time.monotonic()+plan['parent_wait_limit_s']
        while True:
            parent=json.loads((Path(plan['parent_worker'])/'status.json').read_text())
            if parent['state']=='FAILED':raise ValueError('Parent evaluation failed; inspect before continuing')
            if parent['state']=='COMPLETED' and not Path('/proc',str(plan['parent_guard_pid'])).exists():break
            if time.monotonic()>deadline:raise ValueError('Parent wait deadline; no GPU job started')
            time.sleep(5)
        check_sources()
        if sha(ROOT/plan['entry']['path'])!=plan['entry']['before_sha256']:
            raise ValueError('Native entry changed')
        repeat=Path(plan['repeat_output'])
        if repeat.exists():raise ValueError('Repeat output already exists')
        status('RUNNING_CONTROLLER35_REPEAT',output=str(repeat))
        run(plan['repeat_command'],'repeat_guard.log')
        run(['bash','scripts/python.sh','scripts/audit_native_command_history_v1.py',
            '--run',str(repeat),'--arm-target-rate','10','--output',str(output/'repeat_command_audit.json')],
            'repeat_command_audit.log')
        paired(plan['controller35_output'],str(repeat),'repeat_paired.json')
        metrics=json.loads((repeat/'metrics.json').read_text())
        audit=json.loads((output/'repeat_command_audit.json').read_text())
        save('repeat_summary.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],numerical_failures=metrics['numerical_failures']))
        if metrics['numerical_failures'] or not audit['passed']:
            raise ValueError('Repeat numerical/command audit failed')
        check_sources()
        if sha(ROOT/plan['entry']['path'])!=plan['entry']['before_sha256']:
            raise ValueError('Entry changed during repeat')
        status('INSTALLING_LEARNED_HEAD_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True)
        if sha(ROOT/plan['entry']['path'])!=plan['entry']['after_sha256']:
            raise ValueError('Applied entry differs from reviewed source')
        compile((ROOT/plan['entry']['path']).read_text(),plan['entry']['path'],'exec')
        candidate=Path(plan['learned_output'])
        if candidate.exists():raise ValueError('Learned output already exists')
        status('RUNNING_LEARNED_HEAD_TEACHER_OFF',output=str(candidate))
        run(plan['learned_command'],'learned_guard.log')
        paired(plan['student27_output'],str(candidate),'learned_vs_student.json')
        paired(plan['controller35_output'],str(candidate),'learned_vs_teacher.json')
        metrics=json.loads((candidate/'metrics.json').read_text())
        if metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']:
            raise ValueError('Teacher-off trial unexpectedly enabled teacher')
        save('learned_summary.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],numerical_failures=metrics['numerical_failures'],
            learned_recovery_command=metrics['learned_recovery_command']))
        if metrics['numerical_failures']:raise ValueError('Learned trial numerical failure')
        status('COMPLETED',learned_complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
