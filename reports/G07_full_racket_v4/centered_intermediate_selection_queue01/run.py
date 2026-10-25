"""Compare intermediate frozen native actors after the current owner exits.

Original TRAIN128, original single shared GPU0 guard/ledger, no formal case
contents, no training restart and no source mutation while either owner lives.
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
PARENT = REPORTS/'centered_residual_resume32_queue01'
TRAIN = REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
FINAL = REPORTS/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
BASELINE_EVAL = REPORTS/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
BASELINE_POLICY = REPORTS/'residual_ppo_centered_base8_n2048_probe8_02/policy.pt'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN = HERE/'PREDECLARED_PLAN.json'
STATE = HERE/'state.json'
ITERATIONS = [16,24,32]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import export_diagnostic as export_module
spec = importlib.util.spec_from_file_location('preserved_centered_selection_checks',REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
child = None
stopping = False


def write(state,**extra):
    value = dict(state=state,queue_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra)
    core.atomic(STATE,value);print(json.dumps(value),flush=True)


def check_sources(expected):
    for path,digest in expected.items():assert core.sha(ROOT/path) == digest, 'Pinned source changed: '+path


def output(iteration):
    return REPORTS/f'residual_ppo_centered_base8_diag_iter{iteration}_fixed_train128_01'


def command(iteration):
    return [PYTHON,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07',
        '--budget-window','g07-full-racket-unlimited-20261024','--max-seconds','2700',
        '--env-steps-cap','102400','--output',str(output(iteration)),'--require-artifact','completed.json','--',
        PYTHON,str(ROOT/'scripts/evaluate_full_racket_train_v4.py'),'--output',str(output(iteration)),
        '--checkpoint',str(HERE/'exports'/f'iter{iteration:02d}'/'policy.pt'),'--noise','none','--seed','724201','--headless']


def ready(summary):
    return (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
            and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def rank(entry):
    summary = entry['summary']
    return (summary['complete'],summary['attack_outgoing'],summary['conditional_recovery_count'],
            summary['airborne_contacts'],-entry['iteration'])


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    assert not any(output(iteration).exists() for iteration in ITERATIONS)
    parent = core.read(PARENT/'state.json')
    assert core.live(parent['queue_pid'],'centered_residual_resume32_queue01/run.py')
    active = [row for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if row['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == TRAIN
    assert core.live(active[0]['pid'],TRAIN.name) and core.live(active[0]['child_pid'],TRAIN.name)
    expected = dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes']);check_sources(expected)
    # These are real saved public inputs and actual native weight files, not
    # a mock environment or a newly measured policy success.
    reference = export_module.reference_check()
    policy,export = export_module.export(16)
    preparation = dict(prepared_at=datetime.now(timezone.utc).isoformat(),actual_source8_reference=reference,
        actual_intermediate16_export=export,new_native_job_started=False,formal_acceptance=False,goal_complete=False)
    path = HERE/'CPU_PREPARATION_CHECK.json';assert not path.exists();core.atomic(path,preparation)
    for source in [Path(__file__).resolve(),HERE/'export_diagnostic.py',path,policy,policy.parent/'EXPORT_CHECK.json']:
        expected[str(source.relative_to(ROOT))] = core.sha(source)
    for source,digest in reference['sample_source_sha256'].items():expected[source] = digest
    check_sources(expected)
    core.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=expected,
        verified_parent_pid=parent['queue_pid'],verified_guard_pid=active[0]['pid'],verified_native_pid=active[0]['child_pid'],
        candidate_intermediate_iterations=ITERATIONS,evaluation_commands={str(i):command(i)for i in ITERATIONS},
        original_all128_TRAIN_cases=True,reset_noise='none',reset_noise_seed=724201,
        baseline_iterations=[8,40],rank_order=['complete','attack_outgoing','conditional_recovery_count','airborne_contacts','earlier_iteration_on_exact_tie'],
        original_readiness_gate=dict(complete_success_rate=.5,airborne_contact_rate=.7,conditional_recovery_rate=.85,numerical_failures=0),
        skip_extra_assessments_if_actual40_already_ready=True,
        rationale='Retain an actually measured intermediate actor if the final40 changes unfavorably; the earlier mutable-actor40 failed despite rare training successes. Intermediate choices registered before their frozen results, after observing the present rare TRAIN success spanning actor10..16.',
        launch_condition='Actual parent process exit, READY_FOR_VISUAL_REVIEW, successful32-update training receipt and complete original40 TRAIN128, selected trace audit passed, no active GPU ledger job, unchanged held sources. No timer-only restart.',
        intermediate_weights='Actual native model_i diagnostic actors and frozen public normalizers exported with the existing production helper. These are inference policies, never canonical restart checkpoints. Same fixed baseline, physical asset, original joint scales and rules.',
        sources_must_remain_unchanged_until_this_workflow_exits=True,
        resource_authorization='Only original main shared GPU0 guard/ledger and unlimited cumulative authorization. Three finite2700s/102400step evaluation caps; no cloud.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),pinned_sources=len(expected),intermediate16_actor_error=export['actor_export_max_error'],
        reference8_actor_error=reference['actual_source8_actor_export_max_error'],actual_public_samples=reference['actual_public_samples'],
        parent_pid=parent['queue_pid'],new_native_job_started=False)),flush=True)


def cpu_csv(run,iteration):
    global child
    prefix = HERE/f'iter{iteration}_episodes'
    with (HERE/f'iter{iteration}_csv.log').open('x') as stream:
        child = subprocess.Popen([PYTHON,str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),
            '--run',str(run),'--output-prefix',str(prefix)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        assert child.wait() == 0 and not stopping


def main():
    global child
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN);check_sources(plan['source_hashes'])
    assert plan['evaluation_commands'] == {str(i):command(i)for i in ITERATIONS}
    write('WAITING_FOR_ACTUAL_PARENT_EXIT',parent_pid=plan['verified_parent_pid'],plan_sha256=core.sha(PLAN),new_native_job_started=False)
    while core.live(plan['verified_parent_pid'],'centered_residual_resume32_queue01/run.py'):
        assert not stopping,'Observer cancelled'
        time.sleep(10)
    check_sources(plan['source_hashes'])
    parent = core.read(PARENT/'state.json')
    assert parent['state'] == 'READY_FOR_VISUAL_REVIEW', 'Parent requires review: '+parent['state']
    assert parent['selected_training_trace_audit_passed']
    assert not any(row['status'] == 'RUNNING' for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    for key in ['verified_guard_pid','verified_native_pid']:
        assert not core.live(plan[key],TRAIN.name)
    parent_spec = importlib.util.spec_from_file_location('actual_completed_centered40_owner',PARENT/'run.py')
    owner = importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(owner)
    training = owner.verify_training(core.verify_job(TRAIN,32*2048*128))
    final = core.verify_evaluation(FINAL,TRAIN/'policy.pt')
    baseline = core.verify_evaluation(BASELINE_EVAL,BASELINE_POLICY)
    entries = [dict(iteration=8,policy=str(BASELINE_POLICY),summary=baseline),
               dict(iteration=40,policy=str(TRAIN/'policy.pt'),summary=final)]
    core.atomic(HERE/'VERIFIED_PARENT_RESULT.json',dict(training=training,final40=final,
        actual_parent_exited=True,formal_acceptance=False,goal_complete=False))
    if ready(final):
        write('READY_FOR_ORIGINAL_FORMAL_REVIEW',selected=entries[1],extra_intermediate_evaluations_skipped=True)
        return
    for iteration in ITERATIONS:
        assert not stopping and not output(iteration).exists();check_sources(plan['source_hashes'])
        policy,export = export_module.export(iteration)
        assert export['actor_export_max_error'] <= 2e-5
        write('RUNNING_FIXED_INTERMEDIATE_TRAIN128',iteration=iteration,policy=str(policy),policy_sha256=core.sha(policy))
        with (HERE/f'iter{iteration}_guard.log').open('x') as stream:
            child = subprocess.Popen(command(iteration),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            code = child.wait()
        assert code == 0 and not stopping,'Guard failed; preserve the real attempt and do not auto-restart'
        check_sources(plan['source_hashes'])
        summary = core.verify_evaluation(output(iteration),policy)
        core.atomic(HERE/f'iter{iteration}_RESULT.json',dict(iteration=iteration,policy_sha256=core.sha(policy),
            independent_original_TRAIN128=summary,formal_acceptance=False,goal_complete=False))
        cpu_csv(output(iteration),iteration)
        entries.append(dict(iteration=iteration,policy=str(policy),summary=summary))
    selected = max(entries,key=rank)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),all_candidates=entries,selected=selected,
        selected_policy_sha256=core.sha(Path(selected['policy'])),rank_order=plan['rank_order'],
        next_action='READY_FOR_ORIGINAL_FORMAL_REVIEW' if ready(selected['summary']) else 'CONTINUE_METHOD_FROM_BEST_MEASURED_FROZEN',
        scope='Same original128 TRAIN development cases/noise/seed and unchanged original criteria. Five measured frozen actors, no blind-test or formal score.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    core.atomic(HERE/'RESULT.json',result);check_sources(plan['source_hashes'])
    write('COMPLETED_INTERMEDIATE_SELECTION',selected_iteration=selected['iteration'],selected_policy=selected['policy'],
        selected_summary=selected['summary'],next_action=result['next_action'])


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:child.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:prepare()
        else:
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc());write('FAILED_PARENT_OR_INTERMEDIATE_REVIEW',error=repr(error));raise
