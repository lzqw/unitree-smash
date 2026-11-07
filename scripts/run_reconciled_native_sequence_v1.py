"""Run a reviewed independent sequence diagnostic after a reconciled timeout.

Does not declare the failed comparison successful or retry its incomplete run.
Original GPU admission, finite guard, input pins and dispatch audit all remain.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan)),f)
    def write(name,value):
        path=out/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    def status(value,**kwargs):write('status.json',dict(state=value,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kwargs))
    def check():
        for path,digest in plan['source_sha256'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Pinned input changed '+path)
    def run(command,name):
        with (out/name).open('x') as f:subprocess.run(command,cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        review=plan['timeout_reconciliation'];receipt=Path(review['job'])
        if sha(receipt)!=review['job_sha256']:raise ValueError('Changed timeout receipt')
        prior=json.loads(receipt.read_text())
        if prior['status']!='FAILED' or not prior['timed_out'] or prior['receipt_present']:
            raise ValueError('Expected terminal incomplete timeout')
        if any(Path('/proc',str(pid)).exists() for pid in review['exited_pids']):
            raise ValueError('Prior pipeline process remains live')
        for path,digest in review['terminal_status_sha256'].items():
            if sha(path)!=digest or json.loads(Path(path).read_text())['state']!='FAILED':
                raise ValueError('Unreconciled dependent queue')
        command=plan['command'];split=command.index('--');guard=command[:split];native=command[split+1:]
        if ('scripts/run_gpu_job.py' not in guard or int(guard[guard.index('--max-seconds')+1])>1800
                or int(guard[guard.index('--env-steps-cap')+1])!=102400
                or native[native.index('--device')+1]!='cuda:0'
                or '--recovery-sequence-bundle' not in native or '--landing-mpc-cases' in native
                or '--learned-recovery-command-head' in native):raise ValueError('Invalid isolated diagnostic')
        check()
        if Path(plan['output']).exists():raise ValueError('Preserve existing native evidence')
        status('RUNNING_REVIEWED_FULL_EPISODE_SEQUENCE')
        run(command,'guard.log');check()
        actual=json.loads((Path(plan['output'])/'sequence_entry_provenance.json').read_text())
        if actual['sha256']!=plan['entry_sha256']:raise ValueError('Different native entry executed')
        status('AUDITING_NATIVE_SEQUENCE')
        for i,c in enumerate(plan['post_commands']):run(c,f'audit{i}.log')
        audit=json.loads((out/'native_audit.json').read_text())
        if not audit['passed_dispatch_integrity']:raise ValueError('Dispatch audit failed')
        write('result.json',dict(complete5=audit['complete5'],selected_case_outcome=audit['selected_case_outcome'],
            last_observed_horizon_error=audit['last_observed_horizon_error'],
            native_execution=True,automatic_promotion=False,valid_expert_labels=False,teacher_off_learning_result=False))
        status('COMPLETED',complete5=audit['complete5'])
    except BaseException as error:status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
