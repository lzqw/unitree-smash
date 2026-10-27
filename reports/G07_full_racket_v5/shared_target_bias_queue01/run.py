"""After the existing chain, conditionally run5 bounded shared-target trials."""
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
BASE = HERE.parent
WORK = ROOT.parent/'unitree_smash_contact_recovery'
PRIOR = BASE/'stance_recovery_queue01'
PROPOSALS = BASE/'current_intercept_diagnosis_20261027'
SEARCH = BASE/'shared_target_bias5_n128_01'
COLD = BASE/'shared_target_bias5_best_cold128_01'
PRECISION = ROOT/'reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT_PID = 1990613
PARENT_TICKS = '71376602'
stopping = False
child = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, record):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def state(name, **extra):
    atomic(HERE/'state.json', dict(state=name, owner_pid=os.getpid(),
        observed_at=datetime.now(timezone.utc).isoformat(), formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, **extra))


def verify(expected):
    for path, digest in expected.items():
        assert sha(path) == digest, path


def parent_identity():
    path = Path('/proc', str(PARENT_PID))
    if not path.exists():
        return False
    parts = (path/'stat').read_text().rsplit(')', 1)[1].split()
    assert parts[19] == PARENT_TICKS
    if parts[0] == 'Z':
        return True  # Wait for reaping; a transient zombie is not a new launch opportunity.
    assert str(PRIOR/'run.py').encode() in (path/'cmdline').read_bytes()
    return True


def factory():
    path = PROPOSALS/'native_factory.py'
    spec = importlib.util.spec_from_file_location('admitted_shared_target_factory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    assert not (HERE/'PLAN.json').exists() and parent_identity()
    expected = {}
    for path in [PRIOR/'PLAN.json', BASE/'early_lateral_queue01/BOUND_SELECTED_SOURCE_INPUTS.json',
                 BASE/'windowed_swing_queue01/BOUND_SELECTED_SOURCE_INPUTS.json']:
        record = read(path)
        expected.update(record.get('source_sha256', record))
        expected[str(path)] = sha(path)
    for name in ['CPU_SHARED_TARGET_CHECK.json', 'NATIVE_FACTORY_CPU_CHECK.json']:
        record = read(PROPOSALS/name)
        expected.update(record['source_sha256'])
    files = [Path(__file__), HERE/'source_support.py', HERE/'stance_source_admission.py',
        HERE/'evaluate_shared_target.py', HERE/'prepare_entry.py', HERE/'ENTRY_GENERATION_CHECK.json',
        HERE/'check_preparation.py', HERE/'PRE_NATIVE_CPU_CHECK.json',
        PROPOSALS/'native_factory.py', PROPOSALS/'shared_target_controller.py',
        PROPOSALS/'CPU_SHARED_TARGET_CHECK.json', PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json',
        PROPOSALS/'NATIVE_PROPOSAL.json']
    expected.update({str(p): sha(p) for p in files})
    verify(expected)
    check = read(HERE/'PRE_NATIVE_CPU_CHECK.json')
    assert check['native_entry_help_exit_code'] == 0 and check['native_loop_and_scoring_AST_unchanged']
    assert check['actual_early_independent_source_admitted'] and check['skip_and_fail_closed_gates_checked']
    proposal = read(PROPOSALS/'NATIVE_PROPOSAL.json')
    assert [c['candidate_id'] for c in proposal['candidates']] == list(range(5))
    atomic(HERE/'PLAN.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        parent_owner_pid=PARENT_PID, parent_owner_start_ticks=PARENT_TICKS,
        parent_owner_program=str(PRIOR/'run.py'), source_sha256=expected,
        source_selection='After the existing parent exits successfully or legitimately skips, choose the newest actually completed search with actual independent128; admit its full raw source chain. Never fabricate skipped-stage checkpoints.',
        source_gate='Original TRAIN airborne_contact_rate must remain below original0.70. Numerical failures prohibited; source actors/physics/original128 retained.',
        candidate_definitions=proposal['candidates'],
        actual_future_source_and_own_input_CPU_check_pending=True,
        selection='Original complete/attack-cross-target/TRAIN-partial/recovery/air rank; earlier candidate wins a tie.',
        sequence=[dict(output=str(SEARCH), env_steps_cap=512000, max_seconds=16200),
                  dict(output=str(COLD), env_steps_cap=102400, max_seconds=5400)],
        automatic_retry=False, no_cases_or_formal_gates_removed=True,
        formal_cases_read=False, formal_acceptance=False))
    print(json.dumps(dict(prepared=True, bound_inputs=len(expected), parent_owner_pid=PARENT_PID)))


def run_job(output, mode, source, source_cold, cap, seconds, expected):
    global child
    assert not stopping and not output.exists()
    verify(expected)
    assert not any(x['status'] == 'RUNNING' for x in read(ROOT/'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT/'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds), '--env-steps-cap', str(cap),
        '--output', str(output), '--require-artifact', 'completed.json', '--', PYTHON,
        str(HERE/'evaluate_shared_target.py'), '--output', str(output), '--mode', mode,
        '--source', str(source), '--source-evaluation', str(source_cold),
        '--precision-receipt', str(PRECISION), '--seed', '724201', '--headless']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
    env.pop('OPENBLAS_NUM_THREADS', None)
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    state('RUNNING_SHARED_TARGET_'+mode.upper(), guard_pid=child.pid, output=str(output))
    code = child.wait()
    assert code == 0 and not stopping
    record = read(output/'job.json')
    assert record['status'] == 'SUCCEEDED' and record['exit_code'] == 0 and record['receipt_present']
    assert 0 < record['charged_env_steps'] == read(output/'completed.json')['env_steps'] <= cap
    assert not Path('/proc', str(record['child_pid'])).exists()
    verify(expected)
    child = None


def main():
    plan = read(HERE/'PLAN.json')
    expected = dict(plan['source_sha256'])
    expected[str(HERE/'PLAN.json')] = sha(HERE/'PLAN.json')
    verify(expected)
    state('WAITING_FOR_CONFIRMED_EXISTING_STANCE_CHAIN_AND_COLD128', parent_owner_pid=PARENT_PID,
          actual_source_pending=True, new_GPU_child=False)
    while parent_identity():
        assert not stopping
        time.sleep(15)
    assert not stopping
    sys.path.insert(0, str(WORK))
    import torch
    from source_support import STAGES, select_completed_stage, stage_paths, verify_selected_source, replay_selected_public
    states = {s[0]: read(BASE/s[0]/'state.json') for s in STAGES}
    selected_stage = select_completed_stage(states)
    source_path, source_cold = stage_paths(selected_stage)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    protocol_path = WORK/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    check = verify_selected_source(source_path, source, source_cold, WORK, sha(protocol_path))
    protocol = read(protocol_path)
    summary = check['source_cold_summary']
    atomic(HERE/'ACTUAL_COMPLETED_SOURCE_DECISION.json', dict(source_stage=selected_stage[0],
        source_check=check, source_selected=True, selected_at=datetime.now(timezone.utc).isoformat(),
        formal_acceptance=False))
    if summary['airborne_contact_rate'] >= protocol['thresholds']['airborne_contact_rate']:
        state('SKIPPED_ORIGINAL_AIR_GATE_MET_FOCUS_ATTACK_OR_RECOVERY', summary=summary,
              source_policy=str(source_path), source_cold=str(source_cold), formal_still_required=True)
        return
    verify(expected)
    state('CHECKING_ACTUAL_FINAL_SOURCE_CPU_BEFORE_SHARED_TARGET_NATIVE', source_policy_sha256=sha(source_path))
    replay = replay_selected_public(source_path, source, source_cold, WORK, factory())
    atomic(HERE/'SELECTED_SOURCE_CPU_CHECK.json', replay)
    if not any(x['changed_command_rows'] for x in replay['actual_selected_source_all5_public_checks'][1:]):
        state('SKIPPED_NO_CAUSAL_COMMAND_CHANGE_PLANNER_OR_TRACKING_REVIEW_REQUIRED',
              source_policy=str(source_path), source_cold=str(source_cold))
        return
    expected.update(check['source_sha256'])
    expected[str(HERE/'SELECTED_SOURCE_CPU_CHECK.json')] = sha(HERE/'SELECTED_SOURCE_CPU_CHECK.json')
    atomic(HERE/'BOUND_SELECTED_SOURCE_INPUTS.json', expected)
    run_job(SEARCH, 'search', source_path, source_cold, 512000, 16200, expected)
    from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
    history = read(SEARCH/'metrics.json')['history']
    assert [x['candidate_id'] for x in history] == list(range(5))
    protocol['_root'] = str(WORK)
    for entry in history:
        rows = read(SEARCH/f"episodes_{entry['candidate_id']:03d}.json")
        actual = summarize_candidate(rows, protocol)
        assert actual['episodes'] == 128 and actual['numerical_failures'] == 0
        assert all(actual[k] == entry['summary'][k] for k in actual)
    best = max(history, key=lambda x: tuple(x['summary']['rank']))
    assert sha(SEARCH/f"candidate_{best['candidate_id']:03d}.pt") == sha(SEARCH/'best_candidate.pt')
    atomic(HERE/'ACTUAL_SELECTED_SEARCH_RESULT.json', dict(selected=best,
        selected_policy_sha256=sha(SEARCH/'best_candidate.pt'), formal_acceptance=False))
    run_job(COLD, 'cold', SEARCH/'best_candidate.pt', source_cold, 102400, 5400, expected)
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    cases, indices = training_cases(WORK, protocol)
    summary = summarize_training(read(COLD/'episodes.json'), cases, indices, protocol)
    recorded = read(COLD/'metrics.json')['history'][0]['summary']
    assert all(recorded[k] == summary[k] for k in ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures'])
    state('COMPLETED_SHARED_TARGET5_AND_INDEPENDENT_COLD128', final_summary=summary)


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
        with (HERE/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                main()
            except BaseException:
                atomic(HERE/'ERROR.json', dict(error=traceback.format_exc(), automatic_retry=False))
                state('FAILED_PRESERVED_FOR_REVIEW')
                raise
