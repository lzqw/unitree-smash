#!/usr/bin/env python3
"""Finite serial evaluations after an intentional runtime-only restart.

The prior interrupted run is retained. This worker neither edits live sources
nor retries jobs, changes physical parameters, or promotes a candidate.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());out=args.plan.parent
    with (out/'queue_owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),plan_sha256=sha(args.plan)),f)
    def save(name,value):
        path=out/name;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    def status(state,**kw):
        save('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check_sources():
        for name,expected in plan['required_source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Pinned input changed: '+name)
    def run(command,log):
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (out/log).open('x') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        status('CHECKING_TERMINAL_PREDECESSORS')
        for pid in plan['terminated_predecessor_pids']:
            if Path('/proc',str(pid)).exists():raise ValueError('Old project process remains live: '+str(pid))
        old=Path(plan['interrupted_run'])
        if (not (old/'interrupted.json').exists() or
                json.loads((old/'job.json').read_text())['status']!='FAILED'):
            raise ValueError('Require explicit preserved interruption and guard accounting')
        for job in plan['jobs']:
            check_sources()
            path=Path(job['output'])
            if path.exists():raise ValueError('Candidate output already exists: '+str(path))
            status('RUNNING_'+job['name'],output=str(path))
            run(job['command'],job['name']+'_guard.log')
            check_sources()
            runtime=json.loads((path/'landing_blas_runtime.json').read_text())
            if runtime['requested_threads']!=1 or any(x['after']!=1 for x in runtime['pools']):
                raise ValueError('Native BLAS pools did not retain requested thread count')
            status('AUDITING_'+job['name'])
            for index,command in enumerate(job['post_commands']):
                run(command,job['name']+f'_audit{index}.log')
            metrics=json.loads((path/'metrics.json').read_text())
            if metrics['numerical_failures']:raise ValueError('Native numerical failure')
            if job['name']=='LEARNED_EVENT_HEAD' and metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']:
                raise ValueError('Teacher unexpectedly enabled')
            save(job['name']+'_result.json',dict(complete5=metrics['complete5'],
                air=metrics['airborne_contacts'],recovery=metrics['conditional_recovery_count'],
                numerical_failures=metrics['numerical_failures'],automatic_promotion=False))
        status('COMPLETED')
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
