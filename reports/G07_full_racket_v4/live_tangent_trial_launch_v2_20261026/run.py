"""Run both original proposals after reviewing and fixing the import-path failure.

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


def trial_command(variant):
    result=probe.command(variant)
    original=str(probe.HERE/(variant['name']+'_native_train128'))
    new=str(probe.HERE/(variant['name']+'_native_train128_v2'))
    assert result.count(original)==2
    result=[new if value==original else value for value in result]
    source=str(probe.HERE/'native_probe.py');assert result.count(source)==1
    return [str(ROOT/'scripts/run_full_racket_public_tangent_probe_v2.py')if value==source else value for value in result]


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    parent=core.read(recovery.STATE);assert parent['state']=='SUCCEEDED'
    assert not core.live(parent['queue_pid'],'joint_placement_iter40_recovery_20261026/run.py')
    failed=REPORTS/'live_tangent_trial_20261025/gain035_native_train128'
    job=core.read(failed/'job.json');assert job['status']=='FAILED' and not job['receipt_present']
    assert not core.live(job['pid'],failed.name) and not core.live(job['child_pid'],failed.name)
    assert not (failed/'INPUTS.json').exists() and not (failed/'completed.json').exists()
    assert "ImportError: cannot import name 'Queue' from 'queue'" in (failed/'stdout.log').read_text()
    ledger=core.read(ROOT/'status/GPU_LEDGER.json');assert not any(row['status']=='RUNNING'for row in ledger['runs'])
    original=core.read(probe.PLAN);expected=dict(core.read(REPORTS/'live_tangent_trial_recovery_20261026/RECOVERY_PLAN.json')['source_hashes'])
    for path in [Path(__file__).resolve(),ROOT/'scripts/run_full_racket_public_tangent_probe_v2.py',
        HERE/'ENTRYPOINT_DIFF.txt',failed/'job.json',failed/'stdout.log']:
        expected[str(path.relative_to(ROOT))]=core.sha(path)
    probe.check_sources(expected)
    import_check=subprocess.run([probe.PYTHON,'scripts/run_full_racket_public_tangent_probe_v2.py','--check-imports-only'],
        cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),capture_output=True,text=True,check=True)
    checked=json.loads(import_check.stdout);assert not checked['simulator_started']
    core.atomic(HERE/'IMPORT_PATH_CHECK.json',checked)
    for variant in original['preparation']['variants']:
        assert not (probe.HERE/(variant['name']+'_native_train128_v2')).exists()
        assert core.sha(ROOT/variant['proposal'])==variant['proposal_sha256']
    summary=core.verify_evaluation(probe.PARENT_EVAL,probe.PARENT_TRAIN/'policy.pt')
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        reviewed_startup_failure=str(failed),failure_job=job,
        reason='The original report-local queue.py shadows the Python standard-library queue module when native_probe.py is the entrypoint. PyTorch import failed before main and before any native control step.',
        correction='Place an explicitly reviewed equivalent entrypoint in scripts, retain the original proposal source directory for immutable hashes, and additionally archive the actual new entrypoint. No proposal/action/geometry/scoring changes.',
        conservative_old_ledger_charge_preserved=True,actual_failed_attempt_control_steps=0,
        original_variants_and_case_selection_unchanged=True,preparation=original['preparation'],
        commands=[trial_command(variant)for variant in original['preparation']['variants']],
        recovered_frozen40_independent_summary=summary,new_outputs_preserve_first_failed_directory=True,
        import_path_CPU_check=checked,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),startup_failure_reviewed=True)),flush=True)


def stop(*_):
    global stopping
    stopping=True
    if guard is not None and guard.poll() is None:guard.terminate()


def main():
    global guard
    assert PLAN.exists() and not STATE.exists()
    plan=core.read(PLAN);probe.check_sources(plan['source_hashes'])
    parent=core.read(recovery.STATE)
    assert parent['state']=='SUCCEEDED',parent
    core.verify_job(probe.PARENT_TRAIN,8388608)
    summary=core.verify_evaluation(probe.PARENT_EVAL,probe.PARENT_TRAIN/'policy.pt')
    recovery.save_once(HERE/'PARENT_FIXED_TRAIN128_CHECK.json',summary)
    if probe.original_formal_ready(summary):
        write('SKIPPED_PARENT_READY_FOR_ORIGINAL_FORMAL_REVIEW',independent_fixed_TRAIN128=summary);return
    results=[]
    for variant,command in zip(plan['preparation']['variants'],plan['commands']):
        assert not stopping and command==trial_command(variant)
        probe.check_sources(plan['source_hashes'])
        output=probe.HERE/(variant['name']+'_native_train128_v2');assert not output.exists()
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
