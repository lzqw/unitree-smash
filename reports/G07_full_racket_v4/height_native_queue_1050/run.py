"""One conditional G07 follow-up, after verified exit of the existing GPU job."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
PRIOR=REPORTS/'train128_lateral_c12_1025'
OUTPUT=REPORTS/'cem33_height_mix50_train32_1050'
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
POLICY=REPORTS/'height_intercept_ready_1045/policy.pt'
OLD_POLICY=REPORTS/'cem32_lateral_feedback_1015/best_candidate.pt'
STATE=HERE/'state.json'
stopping=False
guard=None


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(state,**extra):
    data=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              prior_output=str(PRIOR),next_output=str(OUTPUT),formal_acceptance=False,**extra)
    temporary=STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(STATE)
    print(json.dumps({k:v for k,v in data.items() if k!='source_hashes'}),flush=True)


def live(pid,needle):
    p=Path(f'/proc/{pid}')
    if not p.exists(): return False
    try:
        state=(p/'stat').read_text().split(') ',1)[1].split()[0]
        if state in ('Z','X'): return False
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:
        return False
    if needle not in command:
        raise RuntimeError(f'PID{pid}identity changed; do not signal or reuse it')
    return True


def stop(*_):
    global stopping
    stopping=True
    if guard is not None and guard.poll() is None:
        guard.terminate()  # Project guard forwards this to its own child.


def main():
    global guard
    if STATE.exists(): raise RuntimeError('Existing queue state; inspect its process instead of duplicating')
    if OUTPUT.exists(): raise RuntimeError('Follow-up output must be fresh')
    files=sorted(ROOT.glob('quadsmash/**/*.py'))+[ROOT/'scripts/search_full_racket_ball_v4.py',
        ROOT/'scripts/run_gpu_job.py',ROOT/'configs/g07_full_racket_eval_v4_r3.json',POLICY]
    hashes={str(p):sha(p) for p in files}
    previous_tick=None
    while not stopping:
        running=[pid for pid in [1544539,1544553] if live(pid,PRIOR.name)]
        if not running: break
        progress_path=PRIOR/'progress.json'
        try: tick=json.loads(progress_path.read_text()).get('control_ticks')
        except (FileNotFoundError,json.JSONDecodeError): tick=None
        if tick!=previous_tick or not STATE.exists():
            write('WAITING_FOR_VERIFIED_PRIOR_EXIT',verified_live_pids=running,prior_tick=tick,source_hashes=hashes)
            previous_tick=tick
        time.sleep(10)
    if stopping:
        write('CANCELLED_BEFORE_LAUNCH');return
    # A disappeared handle alone never authorizes a restart. Require the
    # original guard's success, completion receipt and all128 distinct cases.
    job=json.loads((PRIOR/'job.json').read_text())
    if job['pid']!=1544539 or job['child_pid']!=1544553 or job['status']!='SUCCEEDED':
        write('SKIPPED_PRIOR_NOT_SUCCEEDED',prior_job=job);return
    completed=json.loads((PRIOR/'completed.json').read_text())
    metric=json.loads((PRIOR/'metrics.json').read_text())
    assert metric['episodes']==metric['distinct_cases']==128
    assert metric['distinct_TRAIN_indices']==list(range(128))
    assert metric['policy_sha256']==sha(OLD_POLICY)
    assert 0<completed['env_steps']<=102400
    if metric['numerical_failures']:
        write('SKIPPED_NUMERICAL_FAILURE',prior_metrics=metric);return
    thresholds=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())['thresholds']
    if all(metric[k]>=thresholds[k] for k in ['complete_success_rate','airborne_contact_rate','conditional_recovery_rate']):
        write('SKIPPED_PROMISING_TRAIN_REQUIRES_FROZEN_EVALUATION',prior_metrics=metric);return
    if any(sha(Path(p))!=expected for p,expected in hashes.items()):
        write('SKIPPED_SOURCE_CHANGED');return
    indices=list(range(0,128,4))
    command=[PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds','2400','--env-steps-cap','102400',
        '--output',str(OUTPUT),'--require-artifact','completed.json','--',PYTHON,
        'scripts/search_full_racket_ball_v4.py','--output',str(OUTPUT),'--warm-start',str(POLICY),
        '--candidates','4','--generations','1','--sigma-fraction','.04','--training-indices',
        *map(str,indices),'--continue-after-miss','--contact-curriculum','--retain-contact-recovery',
        '--target-landing-curriculum','--recovery-search-min','1.20','--search-arm-velocity',
        '--search-recovery-parameters','--search-contact-normal','--search-coordinates',
        '2','3','7','14','18','25','37','--push-search-range','.30','--headless']
    write('READY_TO_LAUNCH',prior_metrics=metric,command=command,source_hashes=hashes)
    with (HERE/'guard.stdout.log').open('w') as stream:
        guard=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write('RUNNING_FOLLOWUP',guard_pid=guard.pid,command=command,source_hashes=hashes)
        code=guard.wait()
    job_path=OUTPUT/'job.json'
    result=json.loads(job_path.read_text()) if job_path.is_file() else None
    okay=code==0 and result is not None and result['status']=='SUCCEEDED'
    write('SUCCEEDED' if okay else 'FOLLOWUP_FAILED',exit_code=code,job=result)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try: main()
    except BaseException as exc:
        write('QUEUE_ERROR',error=repr(exc));raise
