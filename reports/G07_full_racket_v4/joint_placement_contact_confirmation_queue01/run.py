"""Confirm all joint TRAIN completions once the existing guard truly exits.

CPU-only evidence work; never starts/stops a GPU job or fabricates motion.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'
TRAIN=REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01'
MILESTONE=REPORTS/'joint_placement_first_training_complete_20261026'
PARENT=REPORTS/'joint_placement_resume32_queue01'
CORE=REPORTS/'residual_resume24_queue01/run.py'
MATCHER=REPORTS/'residual_iter96_delivery_20261025/contact_checks.py'
CONFIRMATION=MILESTONE/'FINAL_CONTACT_CONFIRMATION.json'
GUARD_PID,TRAINER_PID,PARENT_PID=1642834,1642846,1638238
PLAN,STATE=HERE/'PREDECLARED_PLAN.json',HERE/'state.json'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('original_shared_native_checks',CORE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
stopping=False


def write(state,**extra):
    data=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        existing_guard_pid=GUARD_PID,existing_trainer_pid=TRAINER_PID,
        confirmation=str(CONFIRMATION),GPU_jobs_started=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,data);print(json.dumps(data),flush=True)


def check_sources(expected):
    assert all(core.sha(ROOT/name)==value for name,value in expected.items()),'Pinned contact-confirmation source changed'


def prepare():
    assert not any(path.exists() for path in [PLAN,STATE,CONFIRMATION])
    assert core.live(GUARD_PID,TRAIN.name) and core.live(TRAINER_PID,TRAIN.name)
    assert core.live(PARENT_PID,'joint_placement_resume32_queue01/run.py')
    report=core.read(MILESTONE/'REPORT.json')
    assert report['all_closed_training_counts']['complete']==2
    expected=dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes'])
    for path in [Path(__file__).resolve(),CORE,MATCHER,MILESTONE/'REPORT.json',
            MILESTONE/'INITIAL_INDEPENDENT_CHECK.json',MILESTONE/'complete_episodes.json',
            MILESTONE/'all_closed_training_episodes.json.gz',MILESTONE/'weight_lineage.json',MILESTONE/'build_lineage.py']:
        expected[str(path.relative_to(ROOT))]=core.sha(path)
    check_sources(expected)
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_guard_pid=GUARD_PID,verified_live_trainer_pid=TRAINER_PID,
        verified_live_parent_pid=PARENT_PID,source_hashes=expected,
        action='Wait for actual original guard exit and SUCCEEDED8388608-step receipt. Recompute every final TRAIN outcome, verify the exact preserved15549-row prefix and both complete rows, then match every full episode to its actual first-string native contact and rigid point-velocity identity.',
        preserved_snapshot_update=35,already_independently_observed_complete_episodes=2,
        full_training_motion_recorded=False,renderer_uses_gpu=False,
        original_acceptance_unchanged=True,automatic_retries=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),source_files_verified=len(expected),GPU_jobs_started=False)),flush=True)


def main():
    assert PLAN.exists() and not STATE.exists() and not CONFIRMATION.exists()
    plan=core.read(PLAN);check_sources(plan['source_hashes'])
    write('WAITING_FOR_EXISTING_TRAINING_GUARD_EXIT')
    while not stopping and core.live(GUARD_PID,TRAIN.name):time.sleep(10)
    if stopping:raise InterruptedError('Contact observer cancelled; original native jobs untouched')
    job=core.verify_job(TRAIN,8388608)
    assert job['pid']==GUARD_PID and job['child_pid']==TRAINER_PID
    check_sources(plan['source_hashes'])
    write('INDEPENDENTLY_MATCHING_FINAL_NATIVE_CONTACTS_CPU')
    raw=(TRAIN/'episodes.json').read_bytes();rows=json.loads(raw)
    old_raw=gzip.decompress((MILESTONE/'all_closed_training_episodes.json.gz').read_bytes())
    old=json.loads(old_raw)
    assert hashlib.sha256((json.dumps(rows[:len(old)],indent=2)+'\n').encode()).hexdigest()==hashlib.sha256(old_raw).hexdigest()
    protocol=core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    spec=importlib.util.spec_from_file_location('unchanged_actual_contact_matcher',MATCHER)
    matcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(matcher)
    result=matcher.match_complete_contacts(rows,core.read(TRAIN/'contacts.json'),protocol,
        core.read(MILESTONE/'complete_episodes.json'))
    assert result['independently_recomputed_episode_counts']==core.read(TRAIN/'metrics.json')['counts']
    assert result['matched_complete_episodes']>=2
    result.update(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Final actual native changing-policy TRAIN records and matching contact states. Not frozen performance or a generated training video.',
        original_contacts_sha256=core.sha(TRAIN/'contacts.json'),original_episodes_sha256=hashlib.sha256(raw).hexdigest(),
        all15549_preserved_prefix_rows_exact=True,original_pending_report_flags_preserved_as_history=True,
        GPU_jobs_started=False,formal_cases_read=False,goal_complete=False)
    check_sources(plan['source_hashes']);core.atomic(CONFIRMATION,result)
    write('SUCCEEDED',matched_complete_episodes=result['matched_complete_episodes'],
        independent_training_counts=result['independently_recomputed_episode_counts'],
        confirmation_sha256=core.sha(CONFIRMATION),full_training_motion_recorded=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    with(HERE/'observer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            def stop(*_):
                global stopping
                stopping=True
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except InterruptedError as error:write('CANCELLED',error=str(error));sys.exit(2)
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_CONTACT_VALIDATION',error=repr(error));raise
