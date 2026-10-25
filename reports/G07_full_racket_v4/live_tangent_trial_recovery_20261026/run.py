"""Run the two unchanged, already registered probes after recovered joint40.

The historical observer's skipped state is retained. Proposed policies remain
unexecuted until their first real native jobs finish. Use the original GPU0
guard and cumulative ledger, with the original per-job caps and TRAIN128.
"""
import argparse
from datetime import datetime, timezone
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

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4';sys.path.insert(0,str(ROOT))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


recovery=module('joint40_assessment_recovery',REPORTS/'joint_placement_iter40_recovery_20261026/run.py')
probe=module('original_tangent_probe',REPORTS/'live_tangent_trial_20261025/queue.py')
core=probe.core;PLAN=HERE/'RECOVERY_PLAN.json';STATE=HERE/'state.json'
stopping=False;guard=None


def write(state,**extra):
    value=dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        historical_skipped_state_preserved=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    parent=core.read(recovery.STATE)
    assert parent['state']=='RUNNING_FROZEN40_ORIGINAL_TRAIN128'
    assert core.live(parent['queue_pid'],'joint_placement_iter40_recovery_20261026/run.py')
    historical=core.read(probe.STATE)
    assert historical['state']=='SKIPPED_PARENT_DID_NOT_SUCCEED'
    assert not core.live(historical['queue_pid'],'live_tangent_trial_20261025/queue.py')
    original=core.read(probe.PLAN)
    expected=dict(original['source_hashes']);expected.update(core.read(recovery.PLAN)['source_hashes'])
    for path in [probe.PLAN,recovery.PLAN,Path(__file__).resolve()]:
        expected[str(path.relative_to(ROOT))]=core.sha(path)
    probe.check_sources(expected)
    assert original['commands']==[probe.command(variant)for variant in original['preparation']['variants']]
    for variant in original['preparation']['variants']:
        assert not (probe.HERE/(variant['name']+'_native_train128')).exists()
        assert core.sha(ROOT/variant['proposal'])==variant['proposal_sha256']
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        verified_live_recovery_pid=parent['queue_pid'],original_trial_plan=str(probe.PLAN),
        original_commands_unchanged=True,original_proposals_unchanged=True,
        reason='The original observer skipped after a post-training verifier bug. Consume the separately verified recovered frozen40 result, retaining all original probe settings and cases.',
        decision_order=original['decision_order'],comparison_baseline=original['comparison_baseline'],
        preparation=original['preparation'],commands=original['commands'],
        native_v3_reward_candidate_not_integrated_or_executed=True,
        source_custody='Main sources stay unchanged through both real probe jobs and CPU video observers. Integrate the separate reward candidate only after every source owner exits.',
        physical_parameters_rules_cases_thresholds_changed=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),new_native_job_started=False)),flush=True)


def stop(*_):
    global stopping
    stopping=True
    if guard is not None and guard.poll() is None:guard.terminate()


def main():
    global guard
    assert PLAN.exists() and not STATE.exists()
    plan=core.read(PLAN);probe.check_sources(plan['source_hashes'])
    write('WAITING_FOR_RECOVERED_FROZEN40',verified_parent_pid=plan['verified_live_recovery_pid'])
    while not stopping and core.live(plan['verified_live_recovery_pid'],'joint_placement_iter40_recovery_20261026/run.py'):
        time.sleep(10)
    if stopping:raise InterruptedError('Trial observer cancelled; frozen evaluation untouched')
    parent=core.read(recovery.STATE)
    assert parent['state']=='SUCCEEDED',parent
    core.verify_job(probe.PARENT_TRAIN,8388608)
    summary=core.verify_evaluation(probe.PARENT_EVAL,probe.PARENT_TRAIN/'policy.pt')
    recovery.save_once(HERE/'PARENT_FIXED_TRAIN128_CHECK.json',summary)
    if probe.original_formal_ready(summary):
        write('SKIPPED_PARENT_READY_FOR_ORIGINAL_FORMAL_REVIEW',independent_fixed_TRAIN128=summary);return
    results=[]
    for variant,command in zip(plan['preparation']['variants'],plan['commands']):
        assert not stopping and command==probe.command(variant)
        probe.check_sources(plan['source_hashes'])
        output=probe.HERE/(variant['name']+'_native_train128');assert not output.exists()
        with (HERE/(variant['name']+'.guard.log')).open('x')as log:
            guard=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),start_new_session=True)
            write('RUNNING_NATIVE_PUBLIC_TANGENT_PROBE',variant=variant['name'],guard_pid=guard.pid,output=str(output))
            code=guard.wait()
        assert code==0 and not stopping,'Native guard failed/cancelled; retain artifacts, no automatic restart'
        probe.check_sources(plan['source_hashes'])
        receipt=probe.verify_probe(variant,output)
        recovery.save_once(HERE/(variant['name']+'_RESULT.json'),receipt);results.append(receipt)
        assert receipt['independent_fixed_TRAIN128']['numerical_failures']==0
    write('SUCCEEDED',results=results,
        scope='Both unchanged first native executions on original TRAIN128; no A/B score or full-goal completion.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    with (HERE/'observer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_RECOVERED_PROBE',error=repr(error));raise
