"""After recovery-prior9/cold128 exits, execute3 air-only countermotion proposals."""
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
WORK = ROOT.parent / 'unitree_smash_contact_recovery'
OUTPUTS = ROOT / 'reports/G07_full_racket_v5'
PRIOR = OUTPUTS / 'recovery_prior_queue01'
SOURCE = OUTPUTS / 'recovery_prior9_n128_01/best_candidate.pt'
SOURCE_COLD = OUTPUTS / 'recovery_prior9_best_cold128_01'
PROPOSALS = OUTPUTS / 'public_air_countermotion_20261027'
PRECISION = ROOT / 'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
SEARCH = OUTPUTS / 'air_countermotion3_n128_01'
COLD = OUTPUTS / 'air_countermotion3_best_cold128_01'
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
    for name, digest in expected.items():
        assert sha(name) == digest, name


def prepare():
    assert not (HERE / 'PLAN.json').exists()
    expected = dict(read(PRIOR / 'PLAN.json')['source_sha256'])
    expected.update(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['source_sha256'])
    expected.update(read(PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json')['source_sha256'])
    files = [Path(__file__), HERE / 'evaluate_air_countermotion.py', HERE / 'prepare_entry.py',
        HERE / 'ENTRY_GENERATION_CHECK.json', HERE / 'source_support.py', HERE / 'PRE_NATIVE_CPU_CHECK.json',
        HERE / 'check_preparation.py', PRIOR / 'PLAN.json', PROPOSALS / 'native_factory.py',
        PROPOSALS / 'countermotion.py', PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
        PROPOSALS / 'PROPOSAL.json', PROPOSALS / 'check_native_factory.py', PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json']
    for path in files:
        expected[str(path)] = sha(path)
    verify(expected)
    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 3
    assert read(HERE / 'PRE_NATIVE_CPU_CHECK.json')['native_entry_help_exit_code'] == 0
    parent = Path('/proc/1968858')
    assert parent.exists() and str(PRIOR / 'run.py').encode() in (parent / 'cmdline').read_bytes()
    start_ticks = (parent / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    proposal = read(PROPOSALS / 'PROPOSAL.json')
    assert [row['candidate_id'] for row in proposal['candidates']] == list(range(3))
    atomic(HERE / 'PLAN.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        parent_owner_pid=1968858, parent_owner_start_ticks=start_ticks, parent_owner_program=str(PRIOR / 'run.py'),
        source_sha256=expected, actual_source_pending_recovery9_and_independent_cold=True,
        prospective_source_policy=str(SOURCE), prospective_source_cold=str(SOURCE_COLD),
        source_gate='Prior owner and native children must exit successfully; all9 recovery-prior original128 trials and independent cold128 must be complete; independently recompute original criteria. Skip this proposal if source meets original50/70/85 TRAIN gates.',
        source_actor_and_normalizer_and_case_and_physics_preserved=True,
        algorithm='Public post-hit air-only bounded leg countermotion; validated known28-body model; original.15rad mixer and hardware limits, arms unchanged.',
        candidate_definitions=proposal['candidates'],
        no_training_or_formal_cases_removed=True,
        selection='Original independent complete then attack-cross-target chains then existing TRAIN partial score then recovery then air; tie keeps earlier candidate.',
        sequence=[dict(output=str(SEARCH), env_steps_cap=307200, max_seconds=9000),
                  dict(output=str(COLD), env_steps_cap=102400, max_seconds=5400)],
        automatic_retry=False, formal_cases_read=False, formal_acceptance=False))
    print(json.dumps(dict(prepared=True, bound_inputs=len(expected), parent_owner_pid=1968858)))


def factory():
    spec = importlib.util.spec_from_file_location('selected_early_lateral_factory', PROPOSALS / 'native_factory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(output, mode, source, cap, seconds, expected):
    global child
    assert not stopping and not output.exists()
    verify(expected)
    assert not any(row['status'] == 'RUNNING' for row in read(ROOT / 'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds), '--env-steps-cap', str(cap),
        '--output', str(output), '--require-artifact', 'completed.json', '--', PYTHON,
        str(HERE / 'evaluate_air_countermotion.py'), '--output', str(output), '--mode', mode,
        '--source', str(source), '--source-evaluation', str(SOURCE_COLD),
        '--precision-receipt', str(PRECISION), '--seed', '724201', '--headless']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
    environment.pop('OPENBLAS_NUM_THREADS', None)
    child = subprocess.Popen(command, cwd=ROOT, env=environment)
    state('RUNNING_AIR_COUNTERMOTION_' + mode.upper(), guard_pid=child.pid, output=str(output))
    code = child.wait()
    assert code == 0 and not stopping
    record = read(output / 'job.json')
    assert record['status'] == 'SUCCEEDED' and record['exit_code'] == 0 and record['receipt_present']
    assert 0 < read(output / 'completed.json')['env_steps'] == record['charged_env_steps'] <= cap
    assert not Path('/proc', str(record['child_pid'])).exists()
    verify(expected)
    child = None


def main():
    plan = read(HERE / 'PLAN.json')
    expected = dict(plan['source_sha256'])
    expected[str(HERE / 'PLAN.json')] = sha(HERE / 'PLAN.json')
    verify(expected)
    parent = Path('/proc') / str(plan['parent_owner_pid'])
    state('WAITING_FOR_CONFIRMED_RECOVERY9_OWNER_AND_COLD128', parent_owner_pid=1968858,
          prospective_source=str(SOURCE), source_selected=False)
    while parent.exists():
        assert not stopping
        assert plan['parent_owner_program'].encode() in (parent / 'cmdline').read_bytes()
        assert (parent / 'stat').read_text().rsplit(')', 1)[1].split()[19] == plan['parent_owner_start_ticks']
        time.sleep(15)
    assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128'
    source_job = read(SOURCE_COLD / 'job.json')
    assert source_job['status'] == 'SUCCEEDED' and source_job['exit_code'] == 0 and source_job['receipt_present']
    assert not Path('/proc', str(source_job['child_pid'])).exists()
    verify(expected)
    sys.path.insert(0, str(WORK))
    import torch
    from source_support import verify_prior_source, replay_selected_public
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    protocol_path = WORK / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    check = verify_prior_source(SOURCE, source, SOURCE_COLD, WORK, sha(protocol_path))
    summary = check['source_cold_summary']
    protocol = read(protocol_path)
    atomic(HERE / 'ACTUAL_COMPLETED_SOURCE_DECISION.json', dict(source_check=check,
        source_selected=True, selected_at=datetime.now(timezone.utc).isoformat(), formal_acceptance=False))
    if all(summary[key] >= protocol['thresholds'][key] for key in
           ['complete_success_rate', 'airborne_contact_rate', 'conditional_recovery_rate']):
        state('SKIPPED_SOURCE_TRAIN_GATES_MET_FORMAL_STILL_REQUIRED', summary=summary)
        return
    state('CHECKING_SELECTED_SOURCE_CPU_BEFORE_AIR_COUNTERMOTION_NATIVE', source_policy_sha256=sha(SOURCE))
    replay = replay_selected_public(SOURCE, source, SOURCE_COLD, WORK, factory())
    atomic(HERE / 'SELECTED_SOURCE_CPU_CHECK.json', replay)
    expected.update(check['source_sha256'])
    expected[str(HERE / 'SELECTED_SOURCE_CPU_CHECK.json')] = sha(HERE / 'SELECTED_SOURCE_CPU_CHECK.json')
    atomic(HERE / 'BOUND_SELECTED_SOURCE_INPUTS.json', expected)
    job(SEARCH, 'search', SOURCE, 307200, 9000, expected)
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    history = read(SEARCH / 'metrics.json')['history']
    assert [row['candidate_id'] for row in history] == list(range(3))
    protocol['_root'] = str(WORK)
    for entry in history:
        rows = read(SEARCH / f"episodes_{entry['candidate_id']:03d}.json")
        actual = summarize_candidate(rows, protocol)
        assert all(actual[key] == entry['summary'][key] for key in actual)
        assert actual['episodes'] == 128 and actual['numerical_failures'] == 0
    selected = max(history, key=lambda entry: tuple(entry['summary']['rank']))
    assert sha(SEARCH / f"candidate_{selected['candidate_id']:03d}.pt") == sha(SEARCH / 'best_candidate.pt')
    atomic(HERE / 'ACTUAL_SELECTED_SEARCH_RESULT.json', dict(selected=selected,
        selected_policy_sha256=sha(SEARCH / 'best_candidate.pt'), formal_acceptance=False))
    job(COLD, 'cold', SEARCH / 'best_candidate.pt', 102400, 5400, expected)
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    cases, indices = training_cases(WORK, protocol)
    summary = summarize_training(read(COLD / 'episodes.json'), cases, indices, protocol)
    actual = read(COLD / 'metrics.json')['history'][0]['summary']
    assert all(actual[key] == summary[key] for key in
               ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures'])
    state('COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128', final_summary=summary)


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
