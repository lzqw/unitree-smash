"""Run original A/B200 panels only after a qualifying actual iter96 TRAIN128.

No training, source edits, retries, cloud resources or goal-completion claims.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT = REPORTS / 'residual_resize2048_queue01'
PARENT_PID = 1576853
RESULTS = REPORTS / 'formal_ppo_iter96_20261025'
STATE = HERE / 'state.json'
PROTOCOL = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
module_spec = importlib.util.spec_from_file_location('iter96_native_checks', PARENT / 'run.py')
native = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(native)
core = native.core
POLICY = native.TRAIN / 'policy.pt'
LIMITS = dict(complete_success_rate=.5, airborne_contact_rate=.7,
              conditional_recovery_rate=.85, numerical_failures=0)
stopping = False
child = None


def decide(summary, protocol):
    if protocol['thresholds'] != LIMITS:
        raise ValueError('Original formal thresholds must remain unchanged')
    if summary['episodes'] != 128:
        raise ValueError('Require the complete fixed TRAIN128 assessment')
    numerical = summary['numerical_failures']
    if isinstance(numerical, bool) or not isinstance(numerical, int) or numerical < 0:
        raise ValueError('Require a nonnegative actual numerical failure count')
    if numerical:
        return 'STOP_FOR_NUMERICAL_FAILURE'
    for key in ['complete_success_rate', 'airborne_contact_rate', 'conditional_recovery_rate']:
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Require finite valid fixed-policy rates')
    return ('RUN_ORIGINAL_FORMAL_AB' if all(summary[k] >= LIMITS[k] for k in
        ['complete_success_rate', 'airborne_contact_rate', 'conditional_recovery_rate'])
        else 'SKIPPED_TRAIN_BELOW_ORIGINAL_THRESHOLDS')


def hashes():
    result = native.hashes()
    for path in [Path(__file__).resolve(), ROOT / 'scripts/lock_full_racket_policy_v4.py',
                 ROOT / 'scripts/evaluate_full_racket_v4.py', ROOT / 'scripts/aggregate_full_racket_eval_v4.py']:
        result[str(path)] = core.sha(path)
    return result


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        parent_pid=PARENT_PID, results=str(RESULTS), policy=str(POLICY), goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def cpu(command, log_name):
    global child
    if stopping:
        raise InterruptedError('Cancelled before an owned CPU step')
    with (HERE / log_name).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)
        code = child.wait(timeout=600)
    assert code == 0 and not stopping, 'CPU step failed; inspect ' + log_name


def formal_command(panel):
    return [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', '2700', '--env-steps-cap', '160000',
        '--output', str(RESULTS / panel), '--require-artifact', 'completed.json', '--',
        PYTHON, 'scripts/evaluate_full_racket_v4.py', '--output', str(RESULTS / panel),
        '--checkpoint', str(POLICY), '--policy-lock', str(RESULTS / 'lock/LOCK.json'),
        '--protocol', str(PROTOCOL), '--panel', panel, '--headless']


def verify_panel(panel):
    from quadsmash.evaluation_v4 import validate_frozen_inputs, summarize_panel, digest
    protocol, lock = validate_frozen_inputs(ROOT, PROTOCOL, RESULTS / 'lock/LOCK.json', POLICY)
    protocol['_root'] = str(ROOT)
    output = RESULTS / panel
    job = core.read(output / 'job.json')
    steps = core.read(output / 'completed.json')['env_steps']
    assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == steps
    assert isinstance(steps, int) and not isinstance(steps, bool) and 0 < steps <= 160000 and steps % 200 == 0
    assert not core.live(job['pid'], str(output)) and not core.live(job['child_pid'], str(output))
    assert not (output / 'error.txt').exists()
    summary = summarize_panel(core.read(output / 'episodes.json'), protocol, panel)
    recorded = core.read(output / 'metrics.json')
    assert all(recorded[key] == value for key, value in summary.items())
    assert recorded['policy_sha256'] == lock['policy_sha256'] == core.sha(POLICY)
    assert recorded['protocol_sha256'] == core.sha(PROTOCOL)
    assert recorded['policy_lock_sha256'] == digest(RESULTS / 'lock/LOCK.json')
    assert summary['numerical_failures'] == 0
    assert len(core.read(output / 'initial_states.json')) == 200
    return dict(summary, actual_control_env_steps=steps, charged_gpu_hours=job['charged_gpu_hours'])


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    global child
    assert not STATE.exists() and not RESULTS.exists()
    assert core.live(PARENT_PID, 'residual_resize2048_queue01/run.py')
    expected = hashes()
    protocol = core.read(PROTOCOL)
    assert protocol['thresholds'] == LIMITS
    core.atomic(HERE / 'PREDECLARED_PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=PARENT_PID, source_hashes=expected,
        planned_panels=['A', 'B'], cases_per_panel=200, aggregation='Worse panel; original thresholds',
        thresholds=LIMITS, maximum_panel_control_env_steps=160000, maximum_panel_seconds=2700,
        GPU='Existing allocated GPU0 only; no new resources',
        qualification='Wait for the actual existing native workflow to exit SUCCEEDED; independently verify canonical96/export and every fixed TRAIN128 metric. Run formal A/B only if all original rates are met with zero numerical failure.',
        early_failure_scope='A panel may legitimately finish all200 first episodes before800ticks. Preserve every case and verify its actual step count against the guard cap; never require an invented full160000 steps.',
        goal_completion='Policy threshold aggregation is separate from the full model/source/video/comparison deliverable audit; this queue cannot mark the thread goal complete.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    write('WAITING_FOR_NATIVE_WORKFLOW', source_hashes=expected, formal_cases_read=False,
        note='No formal lock or GPU job has started. Existing training and CPU videos remain owned by their original workflows.')
    while not stopping and core.live(PARENT_PID, 'residual_resize2048_queue01/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; original native workflow untouched')
    parent = core.read(PARENT / 'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_NOT_SUCCEEDED', parent_state=parent['state'], formal_cases_read=False)
        return
    assert hashes() == expected, 'Source changed; inspect before using another GPU job'
    core.verify_job(native.TRAIN, native.STEPS)
    training = native.verify_training()
    summary = core.verify_evaluation(native.EVALUATION, POLICY)
    decision = decide(summary, protocol)
    core.atomic(HERE / 'QUALIFICATION.json', dict(decision=decision,
        independently_verified_training=training, independently_recomputed_fixed_TRAIN128=summary,
        source_hashes=expected, formal_cases_read=False, formal_acceptance=False))
    if decision != 'RUN_ORIGINAL_FORMAL_AB':
        write(decision, independently_recomputed_fixed_TRAIN128=summary, formal_cases_read=False,
            note='Original formal thresholds were not qualified; review the actual fixed result and continue improving the strategy.')
        return
    RESULTS.mkdir()
    (RESULTS / 'ORCHESTRATOR_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    core.atomic(RESULTS / 'QUALIFICATION.json', core.read(HERE / 'QUALIFICATION.json'))
    write('LOCKING_ORIGINAL_FORMAL_INPUTS', source_hashes=expected, formal_cases_read=True)
    cpu([PYTHON, 'scripts/lock_full_racket_policy_v4.py', '--checkpoint', str(POLICY),
        '--protocol', str(PROTOCOL), '--output', str(RESULTS / 'lock')], 'lock.log')
    for panel in ['A', 'B']:
        assert not stopping and hashes() == expected
        assert not (RESULTS / panel).exists()
        command = formal_command(panel)
        with (HERE / (panel + '.guard.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write('RUNNING_FORMAL_' + panel, source_hashes=expected, guard_pid=child.pid,
                command=command, formal_cases_read=True)
            code = child.wait()
        assert code == 0 and not stopping, 'Owned formal guard stopped/failed; no automatic retry'
        result = verify_panel(panel)
        core.atomic(RESULTS / (panel + '_INDEPENDENT_RESULT.json'), result)
    assert hashes() == expected and not stopping
    cpu([PYTHON, 'scripts/aggregate_full_racket_eval_v4.py', '--panel-a', str(RESULTS / 'A'),
        '--panel-b', str(RESULTS / 'B'), '--checkpoint', str(POLICY),
        '--policy-lock', str(RESULTS / 'lock/LOCK.json'), '--protocol', str(PROTOCOL),
        '--output', str(RESULTS / 'AGGREGATE.json')], 'aggregate.log')
    aggregate = core.read(RESULTS / 'AGGREGATE.json')
    assert aggregate['sample_count'] == 400 and aggregate['numerical_failures'] == 0
    assert aggregate['policy_sha256'] == core.sha(POLICY)
    write('FORMAL_THRESHOLDS_PASSED_DELIVERABLES_PENDING' if aggregate['formal_acceptance']
        else 'FORMAL_THRESHOLDS_NOT_MET', formal_cases_read=True,
        formal_policy_thresholds_met=aggregate['formal_acceptance'], aggregate=aggregate,
        note='Original A/B policy thresholds are independently aggregated. The full thread goal and required deliverables still need a separate audit.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except InterruptedError as error:
        write('CANCELLED', error=str(error))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        write('FAILED_VALIDATION_OR_JOB', error=repr(error))
        raise
