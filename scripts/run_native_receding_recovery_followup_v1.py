"""Wait for the existing queue, then run one finite GPU0 measured-feedback diagnostic."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    with (out/'queue_owner.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan)),f)
    def write(name,value):
        path=out/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    def status(state,**kw):write('status.json',dict(state=state,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw))
    def check():
        for name,digest in plan['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Pinned sequence input changed: '+name)
    def run(command,log):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (out/log).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        command=plan['command'];separator=command.index('--');guard=command[:separator];native=command[separator+1:]
        if ('scripts/run_gpu_job.py' not in guard or int(guard[guard.index('--max-seconds')+1])>7200 or
                int(guard[guard.index('--env-steps-cap')+1])!=102400 or
                native[native.index('--device')+1]!='cuda:0' or
                '--recovery-sequence-bundle' not in native or '--landing-mpc-cases' in native or
                '--learned-recovery-command-head' in native):
            raise ValueError('Require isolated finite full128 GPU0 sequence diagnostic')
        check();status('WAITING_FOR_FULL_EPISODE_OPEN_LOOP_AUDIT')
        parent=Path(plan['parent_worker']);deadline=time.monotonic()+plan['wait_limit_s']
        while True:
            state=json.loads((parent/'status.json').read_text());live=identity(plan['parent_pid'])
            if state['pid']!=plan['parent_pid'] or (live is not None and live!=plan['parent_start_ticks']):
                raise ValueError('Parent worker identity changed')
            if state['state']=='FAILED':raise ValueError('Prior queue failed; reconcile before native sequence run')
            if live is None:
                if state['state']!='COMPLETED':raise ValueError('Prior queue disappeared without completion')
                break
            if time.monotonic()>deadline:raise ValueError('Wait expired; no GPU run started')
            time.sleep(5)
        check()
        if Path(plan['output']).exists():raise ValueError('Preserve previous native result')
        status('RUNNING_QUERY_MATCHED_NATIVE_FEEDBACK')
        run(command,'guard.log');check()
        actual=json.loads((Path(plan['output'])/'sequence_entry_provenance.json').read_text())
        if actual['sha256']!=plan['entry_sha256']:raise ValueError('Unexpected native entry executed')
        snapshot=json.loads((Path(plan['output'])/'execution_source_hashes.json').read_text())
        for name,digest in plan['source_sha256'].items():
            path=ROOT/name
            try:relative=str(path.relative_to(ROOT))
            except ValueError:continue
            if relative.startswith(('scripts/','quadsmash/','configs/')) and relative.endswith(('.py','.json')):
                if snapshot.get(relative)!=digest:raise ValueError('Actual source snapshot differs '+relative)
        status('AUDITING_FULL_EPISODE_AND_PREDICTION_ERRORS')
        for i,c in enumerate(plan['post_commands']):run(c,f'audit{i}.log')
        audit=json.loads((out/'native_audit.json').read_text())
        if not audit['passed_dispatch_integrity']:raise ValueError('Native dispatch comparison failed')
        write('result.json',dict(complete5=audit['complete5'],selected_case_outcome=audit['selected_case_outcome'],
            dispatch=audit['dispatch'],native_execution=True,
            sustained_mpc_feedback_implemented=True,
            automatic_promotion=False,valid_expert_labels=False,teacher_off_learning_result=False))
        status('COMPLETED',complete5=audit['complete5'])
    except BaseException as error:
        status('FAILED',error=repr(error));raise

if __name__=='__main__':main()
