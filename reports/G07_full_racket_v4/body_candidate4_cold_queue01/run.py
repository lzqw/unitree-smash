"""Cold-check the measured high-contact alternative after the actual active job."""
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
PRIOR=REPORTS/'fixed_actor_body_cem_gen2_queue01'
VIDEO=ROOT.parent/'unitree_smash_video_framing/reports/G07_full_racket_v4/orthographic_fixed_actor_body_gen2_after_native_queue01'
OUTPUT=REPORTS/'fixed_actor_body_gen1_candidate4_fixed_train128_01'
POLICY=HERE/'candidate4_measured_public.pt'
PYTHON='/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN,STATE=HERE/'PLAN.json',HERE/'state.json'
spec=importlib.util.spec_from_file_location('actual_body_search_owner',PRIOR/'run.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
core=owner.core
stopping=False;child=None


def write(state,**extra):
    core.atomic(STATE,dict(state=state,pid=os.getpid(),observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def check(expected):
    for name,digest in expected.items():assert core.sha(ROOT/name)==digest,name


def main():
    global child
    assert not PLAN.exists() and not STATE.exists() and not OUTPUT.exists()
    old=core.read(PRIOR/'PLAN.json');expected=dict(old['source_hashes'])
    for p in [Path(__file__),POLICY,HERE/'ACTUAL_EXPORT_RECEIPT.json']:
        expected[str(p.relative_to(ROOT))]=core.sha(p)
    check(expected)
    command=[PYTHON,str(ROOT/'scripts/evaluate_full_racket_fixed_actor_body_train_v4.py'),
        '--output',str(OUTPUT),'--checkpoint',str(POLICY),'--noise','none','--seed','724201','--headless']
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        command=command,wait_actual_parent_pid=1794481,wait_for_video_native_input_admission=True,
        reason='Parallel equal-parameter references varied, while measured candidate4 had41air/29recovered. Cold-check its128 cases before discarding a potentially better contact/recovery teacher.',
        chosen_after_internal_search_results=True,case_selection='All original128, no omissions',
        max_seconds=2700,env_steps_cap=102400,automatic_retry=False,formal_cases_read=False,formal_acceptance=False))
    expected[str(PLAN.relative_to(ROOT))]=core.sha(PLAN)
    write('WAITING_FOR_ACTUAL_PRIOR_NATIVE_EXIT_AND_VIDEO_ADMISSION')
    while core.live(1794481,'fixed_actor_body_cem_gen2_queue01/run.py'):
        assert not stopping;time.sleep(5)
    assert core.read(PRIOR/'state.json')['state']=='COMPLETED_FIXED_ACTOR_BODY_SEARCH_AND_COLD128'
    while not (VIDEO/'ACTUAL_NATIVE_INPUT_CHECK.json').is_file():
        assert not stopping
        if core.read(VIDEO/'state.json')['state'].startswith('FAILED'):raise RuntimeError('Preserve actual video-admission failure')
        assert core.live(1795557,'orthographic_fixed_actor_body_gen2_after_native_queue01/run.py')
        time.sleep(5)
    check(expected)
    prior=core.verify_evaluation(owner.FROZEN,owner.TRAIN/'best_candidate.pt')
    if owner.previous.reference.ready(prior):
        write('SKIPPED_PRIOR_READY_FOR_FORMAL_REVIEW',prior_cold128=prior);return
    assert not any(x['status']=='RUNNING' for x in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    core.atomic(HERE/'ADMISSION_DECISION.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        prior_actual_parent_exited=True,video_admission_sha256=core.sha(VIDEO/'ACTUAL_NATIVE_INPUT_CHECK.json'),
        prior_cold128=prior,all_original128_cases=True,formal_acceptance=False))
    wrapped=[PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07','--budget-window',
        'g07-full-racket-unlimited-20261024','--max-seconds','2700','--env-steps-cap','102400',
        '--output',str(OUTPUT),'--require-artifact','completed.json','--',*command]
    with (HERE/'guard.log').open('x') as log:
        child=subprocess.Popen(wrapped,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('RUNNING_ALTERNATIVE_COLD128',guard_pid=child.pid)
        while child.poll() is None:
            assert not stopping;time.sleep(5)
    assert child.returncode==0 and not stopping,'Preserve actual failure; no automatic retry'
    result=core.verify_evaluation(OUTPUT,POLICY);check(expected)
    core.atomic(HERE/'COLD128_INDEPENDENT_RESULT.json',result)
    exported=subprocess.run([PYTHON,str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),
        '--run',str(OUTPUT),'--output-prefix',str(HERE/'COLD128_EPISODES')],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),capture_output=True,text=True)
    (HERE/'csv.log').write_text(exported.stdout+exported.stderr);assert exported.returncode==0
    write('COMPLETED_ALTERNATIVE_COLD128',summary=result,source_custody_released_on_actual_parent_exit=True)


def stop(*_):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)


if __name__=='__main__':
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:main()
        except BaseException:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM);child.wait(timeout=60)
            write('STOPPED' if stopping else 'FAILED',error=traceback.format_exc(),automatic_retry=False)
            raise
