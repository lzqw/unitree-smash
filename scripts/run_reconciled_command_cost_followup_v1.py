#!/usr/bin/env python3
"""Finite matched all128 command-cost trials after explicit baseline reconciliation.

The old strict historical-prefix failure remains recorded; causal dispatch and
all ball-qualified prefixes must be independently verified before launching.
Runs weight0 and weight1 with identical constraints and causal eligibility.
No automatic policy promotion or relaxation of physical/task requirements.
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


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def validate_command(command,output,weight):
    split=command.index('--');guard,native=command[:split],command[split+1:]
    if ('scripts/run_gpu_job.py' not in guard or not 0<int(guard[guard.index('--max-seconds')+1])<=10800
            or int(guard[guard.index('--env-steps-cap')+1])!=102400
            or native[native.index('--device')+1]!='cuda:0'
            or native[native.index('--output')+1]!=str(output)
            or '--learned-recovery-command-head' in native
            or not all(x in native for x in ['--landing-mpc-causal-posthit-only',
                '--landing-mpc-minimum-command-cost','--landing-mpc-full-command-damper',
                '--landing-mpc-interval-active-set','--landing-mpc-free-arm'])
            or float(native[native.index('--landing-mpc-command-change-weight')+1])!=weight):
        raise ValueError('Explicit finite GPU0 matched causal command-cost trial required')
    start=native.index('--landing-mpc-cases')+1;cases=[]
    while start<len(native) and not native[start].startswith('--'):
        cases.append(int(native[start]));start+=1
    if cases!=list(range(128)):raise ValueError('All128 causal eligibility required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan)),f)
    def write(name,value):
        target=out/name;temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
    def status(state,**kw):write('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check(after=False):
        for name,digest in plan['source_sha256'].items():
            if after and name==plan['entry']['path']:digest=plan['entry']['after_sha256']
            if sha(ROOT/name)!=digest:raise ValueError('Pinned input changed: '+name)
    def run(command,log):
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        with (out/log).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        if [r['weight'] for r in plan['runs']]!=[0.,1.]:raise ValueError('Declared two-condition comparison required')
        for r in plan['runs']:validate_command(r['command'],r['output'],r['weight'])
        check()
        audit=json.loads(Path(plan['cpu_audit']).read_text())
        if (not audit['preserve_feasibility'] or audit['command_distance_increases'] or
                audit['summary']['minimum_command']['horizon_accepted']<audit['summary']['unchanged']['horizon_accepted']):
            raise ValueError('CPU candidate still destroys baseline horizon feasibility')
        status('VERIFYING_RECONCILED_NATIVE_BASELINE')
        receipt=json.loads(Path(plan['baseline_reconciliation']).read_text())
        if (receipt['schema']!='causal-dispatch-integrity-v2' or not receipt['passed'] or
                not receipt['global_prechange_bitwise_equal'] or
                not receipt['causal_gate_and_command_mapping_verified'] or
                not receipt['all_baseline_ball5_prefixes_equal'] or
                not receipt['strict_v1_prefix_failure_not_overridden']):
            raise ValueError('Insufficient causal baseline reconciliation')
        for path,digest in receipt['source_sha256'].items():
            if sha(path)!=digest:raise ValueError('Reconciled evidence changed: '+path)
        for pid in plan['finished_parent_pids']:
            if identity(pid) is not None:raise ValueError('Prior process still live')
        native_baseline=Path(receipt['run'])
        if json.loads((native_baseline/'job.json').read_text())['status']!='SUCCEEDED':
            raise ValueError('Baseline simulation not successfully completed')
        write('baseline_reconciliation_used.json',dict(receipt=plan['baseline_reconciliation'],
            receipt_sha256=sha(plan['baseline_reconciliation']),
            historical_prefix_divergent_cases=receipt['historical_prefix_divergent_cases'],
            strict_v1_failure_preserved=True,scope='Causal dispatch verified; no all-lane historical equivalence claim'))
        check()
        status('INSTALLING_REVIEWED_FEASIBLE_COMMAND_COST_ENTRY')
        subprocess.run(['git','apply','--check',plan['entry']['patch']],cwd=ROOT,check=True)
        subprocess.run(['git','apply',plan['entry']['patch']],cwd=ROOT,check=True);check(after=True)
        results=[]
        for index,r in enumerate(plan['runs']):
            target=Path(r['output'])
            if target.exists():raise ValueError('Preserve prior output')
            status('RUNNING_MATCHED_COMMAND_COST_TRIAL',index=index,weight=r['weight'],output=str(target))
            run(r['command'],f'run{index}_guard.log');check(after=True)
            status('AUDITING_MATCHED_COMMAND_COST_TRIAL',index=index,weight=r['weight'])
            for j,command in enumerate(r['post_commands']):run(command,f'run{index}_audit{j}.log')
            metrics=json.loads((target/'metrics.json').read_text());inputs=json.loads((target/'INPUTS.json').read_text())
            ca=json.loads(Path(r['causal_audit']).read_text())
            selection=json.loads(Path(r['selection_audit']).read_text())
            if (not ca['passed'] or not ca['all_baseline_ball5_prefixes_equal'] or not selection['passed'] or metrics['numerical_failures'] or
                    not inputs['landing_mpc_minimum_command_cost'] or
                    inputs['landing_mpc_command_change_weight']!=r['weight']):
                raise ValueError('Native candidate validation failed')
            result=dict(weight=r['weight'],complete5=metrics['complete5'],air=metrics['airborne_contacts'],
                recovery=metrics['conditional_recovery_count'],numerical_failures=metrics['numerical_failures'],
                output=str(target),automatic_promotion=False,formal_acceptance=False)
            results.append(result);write(f'run{index}_result.json',result)
        write('result.json',dict(runs=results,automatic_promotion=False,teacher_off_learning_result=False))
        status('COMPLETED',results=results)
    except BaseException as error:
        status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
