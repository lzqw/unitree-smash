"""Wait for actual15/cold128 owner exit, then observe the original warmup replay on GPU0."""
from datetime import datetime, timezone
import fcntl
import hashlib
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
REPORTS = ROOT / 'reports/G07_full_racket_v4'
UPSTREAM = REPORTS / 'origin0_residual_after_gain_body_queue01'
ORIGINAL_TRAIN = REPORTS / 'origin0_residual_fresh_n128_updates15_01'
COLD = REPORTS / 'origin0_residual_fresh15_fixed_train128_01'
OUTPUT = REPORTS / 'recovery_sequence_replay7_n128_01'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT_PID = 1896271
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('existing_queue_checks', REPORTS / 'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
child = None
stopping = False


def write(state, **extra):
    core.atomic(HERE / 'state.json', dict(state=state, owner_pid=os.getpid(),
        observed_at=datetime.now(timezone.utc).isoformat(), formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, **extra))


def check(expected):
    assert all(core.sha(Path(path)) == digest for path, digest in expected.items())


def save(path, value):
    assert not path.exists()
    core.atomic(path, value)


def prepare():
    assert not any(p.exists() for p in [HERE / 'QUEUE_PLAN.json', HERE / 'state.json', OUTPUT])
    expected = dict(core.read(UPSTREAM / 'ADMISSION_DECISION.json')['source_sha256'])
    for name, digest in core.read(HERE / 'ENTRY_PLAN.json')['runtime_source_sha256'].items():
        assert core.sha(Path(name)) == digest
        expected[name] = digest
    proof = core.read(HERE / 'CPU_INTERFACE_CHECK.json')
    assert proof['environment_tensor_values_unchanged'] and proof['CPU_torch_rng_unchanged']
    assert not proof['native_execution']
    for path in [Path(__file__), HERE / 'ENTRY_PLAN.json', HERE / 'prepare_entry.py',
                 HERE / 'check_cpu_interface.py', UPSTREAM / 'run.py', UPSTREAM / 'PLAN.json',
                 UPSTREAM / 'ADMISSION_DECISION.json', ORIGINAL_TRAIN / 'fresh_initial_model_state.pt',
                 REPORTS / 'origin0_residual_warmup7_actor8_20261027/FIRST128_EPISODES.json']:
        expected[str(path)] = core.sha(path)
    check(expected)
    save(HERE / 'QUEUE_PLAN.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=expected, actual_parent_pid=PARENT_PID,
        required_terminal_state='COMPLETED_ORIGIN0_RESIDUAL15_AND_COLD128',
        hypothesis='Original air TRAIN69 has stable3.73s and no body force, but final landing qualification is false after a reported1ms flight; observe the actual metric sequence without changing criteria.',
        source_choice='Replay original first7 critic-only iterations, same initial source, batch and seed; compare every first128 raw row and initial model before drawing conclusions.',
        next_output=str(OUTPUT), env_steps_cap=114688, max_seconds=5400,
        skip_replay_if_actual_new_cold_source_meets_original_thresholds=True,
        automatic_retry=False, original_scoring_retained=True, formal_cases_read=False, formal_acceptance=False))
    print(json.dumps(dict(prepared=True, bound_inputs=len(expected), native_job_started=False)), flush=True)


def verify_upstream():
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json')
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    result = summarize_training(core.read(COLD / 'episodes.json'), cases, indices, protocol)
    assert result == core.read(UPSTREAM / 'COLD_TRAIN128_RESULT.json')
    assert result['numerical_failures'] == 0
    inputs = core.read(COLD / 'INPUTS.json')
    assert inputs['policy_sha256'] == core.sha(ORIGINAL_TRAIN / 'policy.pt')
    assert inputs['training_indices'] == list(range(128)) and not any(inputs['reset_noise_amplitudes'].values())
    for output in [ORIGINAL_TRAIN, COLD]:
        job = core.read(output / 'job.json')
        assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
        assert not core.live(job['pid'], output.name) and not core.live(job['child_pid'], output.name)
    return result


def verify_observation():
    import torch
    job, completion = core.read(OUTPUT / 'job.json'), core.read(OUTPUT / 'completed.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
    assert completion['env_steps'] == job['charged_env_steps'] == 114688
    assert not core.live(job['pid'], OUTPUT.name) and not core.live(job['child_pid'], OUTPUT.name)
    source = torch.load(ORIGINAL_TRAIN / 'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    replay = torch.load(OUTPUT / 'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    initial_equal = all(torch.equal(value, replay['model_state_dict'][name]) for name, value in source['model_state_dict'].items())
    first = {r['env_id']: r for r in core.read(OUTPUT / 'episodes.json') if r['episode_index'] == 0}
    assert set(first) == set(range(128))
    original = core.read(REPORTS / 'origin0_residual_warmup7_actor8_20261027/FIRST128_EPISODES.json')
    differences = [i for i in range(128) if json.dumps(first[i], sort_keys=True) != json.dumps(original[i], sort_keys=True)]
    observation = core.read(OUTPUT / 'recovery_sequence/OBSERVATION.json')
    assert observation['native_source_confirmed'] and not observation['scoring_changed']
    assert [r['env_id'] for r in observation['episodes']] == [69, 112]
    for row in observation['episodes']:
        assert row['closure']['actual_closed_episode_index'] == 0
        assert core.sha(OUTPUT / 'recovery_sequence' / row['data_file']) == row['data_sha256']
    save(HERE / 'ACTUAL_REPLAY_COMPARISON.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
        initial_model_tensors_identical=initial_equal, first128_all_raw_rows_identical=not differences,
        differing_TRAIN_indices=differences, actual_selected_metric_episodes=observation['episodes'],
        original_scoring_retained=True, causal_metric_failure_proven=False,
        scope='Actual replay provenance and original-row comparison only; event-sequence analysis remains separate.',
        formal_cases_read=False, formal_acceptance=False))
    return initial_equal, differences


def main():
    global child
    plan = core.read(HERE / 'QUEUE_PLAN.json')
    expected = dict(plan['source_sha256'])
    expected[str(HERE / 'QUEUE_PLAN.json')] = core.sha(HERE / 'QUEUE_PLAN.json')
    check(expected)
    write('WAITING_FOR_ACTUAL_RESIDUAL15_AND_COLD128_OWNER_EXIT', wait_parent_pid=PARENT_PID, native_job_started=False)
    while core.live(PARENT_PID, 'origin0_residual_after_gain_body_queue01/run.py'):
        assert not stopping
        time.sleep(5)
    state = core.read(UPSTREAM / 'state.json')['state']
    if state != plan['required_terminal_state']:
        write('UPSTREAM_NOT_COMPLETE', actual_terminal_state=state, native_job_started=False)
        return
    check(expected)
    summary = verify_upstream()
    if (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
            and summary['conditional_recovery_rate'] >= .85):
        write('SKIPPED_NEW_SOURCE_READY_FOR_FORMAL_REVIEW', native_job_started=False)
        return
    assert not OUTPUT.exists() and not any(r['status'] == 'RUNNING' for r in core.read(ROOT / 'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', '5400', '--env-steps-cap', '114688',
        '--output', str(OUTPUT), '--require-artifact', 'completed.json', '--', PYTHON,
        str(HERE / 'audit_training_entry.py'), '--output', str(OUTPUT), '--base-checkpoint',
        str(REPORTS / 'gain_body8_sequential_n128_01/best_candidate.pt'), '--body-source-evaluation',
        str(REPORTS / 'gain_body8_best_fixed_train128_01'), '--precision-receipt',
        str(REPORTS / 'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'), '--config',
        str(ROOT / 'configs/g07_full_racket_origin0_residual_leg15_v4.json'), '--num-envs', '128',
        '--updates', '7', '--seed', '751001', '--headless']
    save(HERE / 'ACTUAL_LAUNCH_COMMAND.json', dict(command=command, previous_owner_actually_exited=True,
        verified_completed_cold_source=summary, formal_acceptance=False))
    with (HERE / 'native.guard.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2'),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write('RUNNING_READ_ONLY_WARMUP7_SEQUENCE_REPLAY', guard_pid=child.pid, output=str(OUTPUT))
        while child.poll() is None:
            assert not stopping
            time.sleep(5)
    assert child.returncode == 0 and not stopping
    check(expected)
    initial_equal, differences = verify_observation()
    write('COMPLETED_READ_ONLY_SEQUENCE_REPLAY', initial_model_identical=initial_equal,
          differing_first128_indices=differences, requires_actual_sequence_analysis=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    else:
        with (HERE / 'observer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not (HERE / 'state.json').exists(), 'Preserve previous execution state'
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                main()
            except BaseException:
                if child is not None and child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    child.wait(timeout=60)
                write('STOPPED' if stopping else 'FAILED', error=traceback.format_exc(), automatic_retry=False)
                raise
