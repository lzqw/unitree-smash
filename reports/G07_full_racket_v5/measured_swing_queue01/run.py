"""Wait for the live recovery-v5 owner, then run measured swing search on GPU0."""
from datetime import datetime, timezone
import fcntl
import hashlib
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
WORK = ROOT.parent / 'unitree_smash_contact_recovery'
OUTPUTS = ROOT / 'reports/G07_full_racket_v5'
PRIOR = OUTPUTS / 'contact_recovery_queue02'
SOURCE = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
SOURCE_COLD = OUTPUTS / 'teacher_contact_recovery_cold128_02'
PRECISION = ROOT / 'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
PROPOSALS = OUTPUTS / 'origin0_measured_swing_velocity_20261027'
SEARCH = OUTPUTS / 'measured_swing9_n128_01'
COLD = OUTPUTS / 'measured_swing9_best_cold128_01'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
stopping = False
child = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def state(name, **extra):
    atomic(HERE / 'state.json', dict(state=name, owner_pid=os.getpid(),
        observed_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra))


def verify(expected):
    for path, digest in expected.items():
        assert sha(path) == digest, path


def prepare():
    assert not (HERE / 'PLAN.json').exists()
    expected = dict(read(PRIOR / 'PLAN.json')['source_sha256'])
    for path in [Path(__file__), HERE / 'evaluate_swing.py', HERE / 'check_cpu.py',
                 HERE / 'PRE_NATIVE_CPU_CHECK.json', PROPOSALS / 'RESULT.json',
                 PROPOSALS / 'swing_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json']:
        expected[str(path)] = sha(path)
    for name in ['INPUTS.json', 'metrics.json', 'episodes.json', 'job.json', 'completed.json',
                 'execution_source_hashes.json', 'origin_layout.json']:
        path = SOURCE_COLD / name
        expected[str(path)] = sha(path)
    verify(expected)
    assert read(HERE / 'PRE_NATIVE_CPU_CHECK.json')['all_nine_candidates_checked']
    parent = 1939834
    proc = Path('/proc') / str(parent)
    if proc.exists():
        assert str(PRIOR / 'run.py').encode() in (proc / 'cmdline').read_bytes()
        start_ticks = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    else:
        assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_REVISED_SOURCE_TRAIN15_AND_COLD128'
        start_ticks = None
    atomic(HERE / 'PLAN.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        parent_owner_pid=parent, parent_owner_start_ticks=start_ticks,
        parent_owner_program=str(PRIOR / 'run.py'), source_sha256=expected,
        execution_worktree=str(WORK), source_policy_sha256=sha(SOURCE),
        unchanged_original_physics_cases_target_and_thresholds=True,
        explicit_recovery_schema='contact-flight-recovery-v1',
        source_comparison='Original measured teacher, nine shared swing proposals, all original128 per candidate; no inherited outcomes.',
        rationale='Actual original-origin0 contacts have retreating racket-point velocity; change instantiated shared arm swing direction within original guards.',
        candidate_definitions=read(PROPOSALS / 'RESULT.json')['candidates'],
        prior_result_gate='Only after the prior owner and actual child exit successfully; skip extra search if prior frozen128 meets the original50/70/85 development gates.',
        search_selection='Original independent complete then full attack-cross-target chain then existing TRAIN partial score then conditional recovery then air; ties keep earlier candidate.',
        sequence=[dict(output=str(SEARCH), env_steps_cap=921600, max_seconds=21600),
                  dict(output=str(COLD), env_steps_cap=102400, max_seconds=5400)],
        automatic_retry=False, formal_cases_read=False, formal_acceptance=False))
    print(json.dumps(dict(prepared=True, bound_inputs=len(expected))))


def job(output, mode, source, cap, seconds, expected):
    global child
    assert not stopping and not output.exists()
    verify(expected)
    assert not any(r['status'] == 'RUNNING' for r in read(ROOT / 'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds), '--env-steps-cap', str(cap),
        '--output', str(output), '--require-artifact', 'completed.json', '--', PYTHON,
        str(HERE / 'evaluate_swing.py'), '--output', str(output), '--mode', mode,
        '--source', str(source), '--source-evaluation', str(SOURCE_COLD),
        '--precision-receipt', str(PRECISION), '--seed', '724201', '--headless']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
    environment.pop('OPENBLAS_NUM_THREADS', None)
    child = subprocess.Popen(command, cwd=ROOT, env=environment)
    state('RUNNING_MEASURED_SWING_' + mode.upper(), guard_pid=child.pid, output=str(output))
    result = child.wait()
    assert result == 0 and not stopping
    job_info = read(output / 'job.json')
    completed = read(output / 'completed.json')
    assert job_info['status'] == 'SUCCEEDED' and job_info['exit_code'] == 0 and job_info['receipt_present']
    assert 0 < completed['env_steps'] == job_info['charged_env_steps'] <= cap
    assert not Path('/proc', str(job_info['child_pid'])).exists()
    verify(expected)
    child = None


def main():
    plan = read(HERE / 'PLAN.json')
    expected = plan['source_sha256']
    verify(expected)
    parent = Path('/proc') / str(plan['parent_owner_pid'])
    state('WAITING_FOR_CONFIRMED_PRIOR_RECOVERY_OWNER', parent_owner_pid=plan['parent_owner_pid'])
    while parent.exists():
        assert not stopping
        assert plan['parent_owner_program'].encode() in (parent / 'cmdline').read_bytes()
        assert (parent / 'stat').read_text().rsplit(')', 1)[1].split()[19] == plan['parent_owner_start_ticks']
        time.sleep(15)
    assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_REVISED_SOURCE_TRAIN15_AND_COLD128'
    final = OUTPUTS / 'residual_contact_recovery15_cold128_02'
    info = read(final / 'job.json')
    assert info['status'] == 'SUCCEEDED' and info['receipt_present'] and info['exit_code'] == 0
    assert not Path('/proc', str(info['child_pid'])).exists()
    assert read(final / 'completed.json')['env_steps'] == info['charged_env_steps'] == 102400
    verify(expected)
    sys.path.insert(0, str(WORK))
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    protocol = read(WORK / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json')
    protocol['_root'] = str(WORK)
    cases, indices = training_cases(WORK, protocol)
    summary = summarize_training(read(final / 'episodes.json'), cases, indices, protocol)
    metrics = read(final / 'metrics.json')
    assert all(metrics[key] == value for key, value in summary.items())
    assert summary['numerical_failures'] == 0
    atomic(HERE / 'PRIOR_COMPLETED_RESULT.json', dict(observed_at=datetime.now(timezone.utc).isoformat(),
        independent_prior_summary=summary, job=info,
        source_sha256={str(final / name): sha(final / name) for name in ['episodes.json', 'metrics.json', 'completed.json', 'job.json']}))
    if all(summary[key] >= protocol['thresholds'][key] for key in
           ['complete_success_rate', 'airborne_contact_rate', 'conditional_recovery_rate']):
        state('SKIPPED_PRIOR_FROZEN_TRAIN_GATES_MET_FORMAL_STILL_REQUIRED', summary=summary)
        return
    job(SEARCH, 'search', SOURCE, 921600, 21600, expected)
    history = read(SEARCH / 'metrics.json')['history']
    assert len(history) == 9 and [x['candidate_id'] for x in history] == list(range(9))
    for entry in history:
        rows = read(SEARCH / ('episodes_' + str(entry['candidate_id']).zfill(3) + '.json'))
        from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
        actual = summarize_candidate(rows, protocol)
        assert all(entry['summary'][key] == value for key, value in actual.items())
        assert actual['episodes'] == 128 and actual['numerical_failures'] == 0
    best = max(history, key=lambda entry: tuple(entry['summary']['rank']))
    selected = SEARCH / ('candidate_' + str(best['candidate_id']).zfill(3) + '.pt')
    assert sha(selected) == sha(SEARCH / 'best_candidate.pt')
    atomic(HERE / 'ACTUAL_SELECTED_SEARCH_RESULT.json', dict(selected=best, source_policy_sha256=sha(SOURCE),
        selected_policy_sha256=sha(selected), native_all128_per_each9_verified=True,
        formal_acceptance=False))
    job(COLD, 'cold', SEARCH / 'best_candidate.pt', 102400, 5400, expected)
    cold_metrics = read(COLD / 'metrics.json')
    assert len(cold_metrics['history']) == 1
    summary = summarize_training(read(COLD / 'episodes.json'), cases, indices, protocol)
    assert all(cold_metrics['history'][0]['summary'][key] == summary[key] for key in
               ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures'])
    state('COMPLETED_MEASURED_SWING9_AND_INDEPENDENT_COLD128', final_summary=summary)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    else:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (HERE / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                main()
            except BaseException:
                atomic(HERE / 'ERROR.json', dict(error=traceback.format_exc(), automatic_retry=False))
                state('FAILED_PRESERVED_FOR_REVIEW')
                raise
