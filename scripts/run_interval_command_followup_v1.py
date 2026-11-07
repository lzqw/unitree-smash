#!/usr/bin/env python3
"""Finite serial native trial after the controller-repeat/learning queue.

Do not mutate the live evaluator. Validate the teacher-off execution, install
the reviewed entry patch after the parent exits, and use the original GPU guard.
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
    if (output/'status.json').exists():raise ValueError('Do not launch this queue twice')
    def save(name,value):
        path=output/name;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    def status(state,**kw):
        save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def run(command,log):
        env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        with (output/log).open('x') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def check_sources():
        for name,expected in plan['required_source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Pinned source changed: '+name)
    try:
        status('WAITING_FOR_CONTROLLER_REPEAT_AND_LEARNED_HEAD')
        deadline=time.monotonic()+plan['parent_wait_limit_s']
        while True:
            parent=json.loads((Path(plan['parent_worker'])/'status.json').read_text())
            if parent['state']=='FAILED':raise ValueError('Parent failed; inspect before more GPU execution')
            if parent['state']=='COMPLETED' and not Path('/proc',str(plan['parent_pid'])).exists():break
            if time.monotonic()>deadline:raise ValueError('Parent deadline; no GPU job started')
            time.sleep(5)
        check_sources()
        if sha(ROOT/plan['entry']['path'])!=plan['entry']['before_sha256']:
            raise ValueError('Native entry changed')
        status('AUDITING_LEARNED_NATIVE_COMMANDS')
        run(['bash','scripts/python.sh','scripts/audit_native_learned_recovery_v1.py',
            '--run',plan['learned_output'],'--checkpoint',plan['learned_checkpoint'],
            '--output',str(output/'learned_command_audit.json')],'learned_command_audit.log')
        check_sources()
        status('INSTALLING_INTERVAL_COMMAND_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True)
        if sha(ROOT/plan['entry']['path'])!=plan['entry']['after_sha256']:
            raise ValueError('Entry differs from reviewed source')
        run(['bash','scripts/python.sh','scripts/check_interval_command_native_entry_v1.py'],'entry_check.log')
        candidate=Path(plan['candidate_output'])
        if candidate.exists():raise ValueError('Native output already exists')
        status('RUNNING_INTERVAL_COMMAND_NATIVE',output=str(candidate))
        run(plan['candidate_command'],'guard.log')
        run(['bash','scripts/python.sh','scripts/audit_native_command_history_v1.py',
            '--run',str(candidate),'--arm-target-rate','10',
            '--output',str(output/'native_command_audit.json')],'native_command_audit.log')
        for label,before in plan['comparisons'].items():
            run(['bash','scripts/python.sh','scripts/compare_frozen_complete5_v2.py',
                '--before',before,'--after',str(candidate),'--output',str(output/(label+'.json'))],label+'.log')
        run(['bash','scripts/python.sh','scripts/summarize_landing_controller_execution_v1.py',
            '--run',str(candidate),'--command-audit',str(output/'native_command_audit.json'),
            '--output',str(output/'execution_breakdown.json')],'execution_breakdown.log')
        metrics=json.loads((candidate/'metrics.json').read_text())
        save('result.json',dict(complete5=metrics['complete5'],air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'],numerical_failures=metrics['numerical_failures']))
        if metrics['numerical_failures']:raise ValueError('Native numerical failure')
        status('COMPLETED',complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
