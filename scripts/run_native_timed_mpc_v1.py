"""Run one bounded GPU0 timed MPC diagnostic with private CPU planning."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
    def status(state,**kw):
        path=out/'status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');tmp.replace(path)
    def check():
        for path,digest in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned input changed '+path)
    def run(command,name):
        with (out/name).open('x') as f:subprocess.run(command,cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        c=plan['command'];split=c.index('--');guard=c[:split];native=c[split+1:]
        if ('scripts/run_gpu_job.py' not in guard or int(guard[guard.index('--max-seconds')+1])>4200
                or int(guard[guard.index('--env-steps-cap')+1])!=102400
                or native[native.index('--device')+1]!='cuda:0' or plan['model_accepted']
                or '--landing-mpc-cases' in native or '--learned-recovery-command-head' in native):
            raise ValueError('Bounded explicit GPU0 diagnostic required')
        check()
        if Path(plan['output']).exists():raise ValueError('Preserve previous native evidence')
        status('CHECKING_NATIVE_MPC_ENTRY');run(plan['check_command'],'entry_check.log');check()
        status('RUNNING_NATIVE_TIMED_MPC');run(c,'guard.log');check()
        actual=json.loads((Path(plan['output'])/'sequence_entry_provenance.json').read_text())
        if actual['sha256']!=plan['entry_sha256']:raise ValueError('Different entry executed')
        status('AUDITING_NATIVE_PROPOSAL');run(plan['post_command'],'audit.log')
        audit=json.loads((out/'native_audit.json').read_text())
        if not audit['passed_dispatch_integrity']:raise ValueError('Actual dispatch/prefix check failed')
        status('COMPLETED_DIAGNOSTIC',complete5=audit['complete5'],
            strict_isolation_passed=audit['strict_untouched_isolation_passed'],valid_expert_labels=False)
    except BaseException as error:status('FAILED',error=repr(error));raise


if __name__=='__main__':main()
