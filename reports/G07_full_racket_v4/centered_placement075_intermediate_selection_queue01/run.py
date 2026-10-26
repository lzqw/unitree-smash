"""Assess075 intermediates only after the current train/eval/video owner exits.

Uses the original main shared GPU0 guard. No formal cases, training restarts,
concurrent GPU jobs, or runtime/configuration/asset edits.
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
PARENT = REPORTS/'centered_placement075_resume32_queue02'
TRAIN = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
FINAL = REPORTS/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02'
BASELINE_EVAL = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
BASELINE_POLICY = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN, STATE = HERE/'PREDECLARED_PLAN.json', HERE/'state.json'
ITERATIONS = [16, 24, 32]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import export_diagnostic as export_module
spec = importlib.util.spec_from_file_location('preserved075_intermediate_checks', REPORTS/'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
child = None
stopping = False


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps(value), flush=True)


def check_sources(expected):
    for path, digest in expected.items():
        assert core.sha(ROOT/path) == digest, 'Pinned source changed: '+path


def output(iteration):
    return REPORTS/f'residual_ppo_centered_placement075_diag_iter{iteration}_fixed_train128_01'


def command(iteration):
    return [PYTHON, str(ROOT/'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024', '--max-seconds', '2700',
        '--env-steps-cap', '102400', '--output', str(output(iteration)), '--require-artifact', 'completed.json', '--',
        PYTHON, str(ROOT/'scripts/evaluate_full_racket_train_v4.py'), '--output', str(output(iteration)),
        '--checkpoint', str(HERE/'exports'/f'iter{iteration:02d}'/'policy.pt'),
        '--noise', 'none', '--seed', '724201', '--headless']


def ready(summary):
    return (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
        and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def rank(entry):
    summary = entry['summary']
    return (summary['complete'], summary['conditional_recovery_count'], summary['airborne_contacts'],
        summary['attack_outgoing'], -entry['iteration'])


def prepare():
    assert ROOT == Path('/home/autodrive/LZQW/unitree_smash'), 'Use the original shared main guard/ledger only'
    assert not PLAN.exists() and not STATE.exists()
    assert not any(output(i).exists() for i in ITERATIONS)
    parent = core.read(PARENT/'state.json')
    assert core.live(parent['queue_pid'], 'centered_placement075_resume32_queue02/run.py')
    active = [r for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if r['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == TRAIN
    assert core.live(active[0]['pid'], TRAIN.name) and core.live(active[0]['child_pid'], TRAIN.name)
    expected = dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes'])
    expected[str((PARENT/'PREDECLARED_PLAN.json').relative_to(ROOT))] = core.sha(PARENT/'PREDECLARED_PLAN.json')
    check_sources(expected)
    reference = export_module.reference_check()
    policy, exported = export_module.export(16)
    path = HERE/'CPU_PREPARATION_CHECK.json'
    assert not path.exists()
    core.atomic(path, dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        actual_source8_reference=reference, actual_intermediate16_export=exported,
        new_native_job_started=False, formal_acceptance=False, goal_complete=False))
    for source in [Path(__file__).resolve(), HERE/'export_diagnostic.py', path, policy, policy.parent/'EXPORT_CHECK.json']:
        expected[str(source.relative_to(ROOT))] = core.sha(source)
    expected.update(reference['sample_source_sha256'])
    expected.update(exported['source_sha256'])
    check_sources(expected)
    progress = core.read(TRAIN/'progress.json')
    core.atomic(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_hashes=expected,
        verified_parent_pid=parent['queue_pid'], verified_guard_pid=active[0]['pid'], verified_native_pid=active[0]['child_pid'],
        candidate_intermediate_iterations=ITERATIONS, evaluation_commands={str(i): command(i) for i in ITERATIONS},
        original_all128_TRAIN_cases=True, reset_noise='none', reset_noise_seed=724201,
        baseline_iterations=[8, 40], rank_order=['complete', 'conditional_recovery_count', 'airborne_contacts',
            'attack_outgoing', 'earlier_iteration_on_exact_tie'],
        rank_is_development_selection_only=True,
        prior025_selection_preserved_with_its_original_attack_priority=True,
        original_readiness_gate=dict(complete_success_rate=.5, airborne_contact_rate=.7,
            conditional_recovery_rate=.85, numerical_failures=0),
        skip_remaining_extra_assessments_if_any_actual_candidate_ready=True,
        observed_current_TRAIN_update=progress['update'], observed_current_TRAIN_complete=progress['complete'],
        decision_timing='After actual placement075 frozen8 and three exploratory complete TRAIN episodes; before frozen075 intermediate16/24/32 or final40 results. This is development selection, not blind evaluation.',
        rationale='Retain a measured intermediate if final40 degrades. Prior025 final40 failed to reproduce rare TRAIN successes. Rank full task first, then recovered contacts and air opportunity before attack-only counts.',
        launch_condition='Actual1734658 parent exit after its entire train/fixed40/video pipeline, READY_FOR_VISUAL_REVIEW, completed native32 receipt and original40 TRAIN128, selected trace audit passed with original SHA checks, no live old guard/native and empty active shared GPU ledger.',
        intermediate_weights='Actual native diagnostic actors for inference only; preserve prior actor RMS10485760 offset and critic/stage counters separately. One unchanged fixed base846, same .03legs/.15arms, full-racket asset, rules and acceptance.',
        sources_must_remain_unchanged_until_this_workflow_exits=True,
        optional_leg06_seven_source_plan_not_launched_and_requires_later_reconciliation=True,
        resource_authorization='Original shared main GPU0 guard/ledger only; unlimited cumulative authorization, each assessment capped2700s/102400steps, at most three sequential assessments. No cloud.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), pinned_sources=len(expected),
        reference8_error=reference['actual_source8_actor_export_max_error'], intermediate16_error=exported['actor_export_max_error'],
        actual_public_samples=reference['actual_public_samples'], new_native_job_started=False)), flush=True)


def verify_fixed(run, policy):
    summary = core.verify_evaluation(run, policy)
    inputs = core.read(run/'INPUTS.json')
    assert inputs['reset_noise_seed'] == 724201
    return summary


def cpu_csv(run, iteration):
    global child
    with (HERE/f'iter{iteration}_csv.log').open('x') as stream:
        child = subprocess.Popen([PYTHON, str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),
            '--run', str(run), '--output-prefix', str(HERE/f'iter{iteration}_episodes')], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), stdout=stream, stderr=subprocess.STDOUT)
        assert child.wait() == 0 and not stopping


def finish(entries, plan, unassessed):
    selected = max((e for e in entries if ready(e['summary'])), key=rank) if any(ready(e['summary']) for e in entries) else max(entries, key=rank)
    next_action = 'READY_FOR_ORIGINAL_FORMAL_REVIEW' if ready(selected['summary']) else 'CONTINUE_METHOD_FROM_BEST_MEASURED_FROZEN'
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), measured_candidates=entries,
        selected=selected, selected_policy_sha256=core.sha(Path(selected['policy'])), rank_order=plan['rank_order'],
        skipped_intermediate_iterations=unassessed, next_action=next_action,
        scope='Original128 TRAIN development cases, same noise/seed/criteria. Ready here means eligible for original A/B review only; no formal panel has been read or passed.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    core.atomic(HERE/'RESULT.json', result)
    write('COMPLETED_INTERMEDIATE_SELECTION', selected_iteration=selected['iteration'],
        selected_policy=selected['policy'], selected_summary=selected['summary'],
        skipped_intermediate_iterations=unassessed, next_action=next_action)


def main():
    global child
    assert ROOT == Path('/home/autodrive/LZQW/unitree_smash')
    assert PLAN.exists() and not STATE.exists()
    plan = core.read(PLAN)
    expected = dict(plan['source_hashes'])
    expected[str(PLAN.relative_to(ROOT))] = core.sha(PLAN)
    check_sources(expected)
    assert plan['evaluation_commands'] == {str(i): command(i) for i in ITERATIONS}
    write('WAITING_FOR_ACTUAL_PARENT_EXIT', parent_pid=plan['verified_parent_pid'],
        plan_sha256=core.sha(PLAN), new_native_job_started=False)
    while core.live(plan['verified_parent_pid'], 'centered_placement075_resume32_queue02/run.py'):
        assert not stopping, 'Observer cancelled'
        time.sleep(10)
    check_sources(expected)
    parent = core.read(PARENT/'state.json')
    assert parent['state'] == 'READY_FOR_VISUAL_REVIEW', 'Actual parent requires review: '+parent['state']
    assert parent['selected_training_trace_audit_passed']
    attempt, audit = core.read(PARENT/'TRACE_AUDIT_ATTEMPT.json'), core.read(PARENT/'TRAINING_TRACE_AUDIT.json')
    assert attempt['passed'] and attempt['exit_code'] == 0
    assert audit['run'] == str(TRAIN) and audit['all_selected_trajectories_checked'] and audit['selected_episodes_checked'] > 0
    assert all(core.sha(Path(name)) == digest for name, digest in audit['source_sha256'].items())
    assert not any(r['status'] == 'RUNNING' for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
    for key in ['verified_guard_pid', 'verified_native_pid']:
        assert not core.live(plan[key], TRAIN.name)
    parent_spec = importlib.util.spec_from_file_location('actual_completed075_owner', PARENT/'run.py')
    owner = importlib.util.module_from_spec(parent_spec)
    parent_spec.loader.exec_module(owner)
    training = owner.verify_training(core.verify_job(TRAIN, 32*2048*128))
    final = verify_fixed(FINAL, TRAIN/'policy.pt')
    baseline = verify_fixed(BASELINE_EVAL, BASELINE_POLICY)
    entries = [dict(iteration=8, policy=str(BASELINE_POLICY), summary=baseline),
        dict(iteration=40, policy=str(TRAIN/'policy.pt'), summary=final)]
    core.atomic(HERE/'VERIFIED_PARENT_RESULT.json', dict(training=training, final40=final,
        actual_parent_exited=True, selected_trace_audit_passed=True, formal_acceptance=False, goal_complete=False))
    if any(ready(e['summary']) for e in entries):
        finish(entries, plan, list(ITERATIONS))
        return
    for index, iteration in enumerate(ITERATIONS):
        assert not stopping and not output(iteration).exists()
        check_sources(expected)
        assert not any(r['status'] == 'RUNNING' for r in core.read(ROOT/'status/GPU_LEDGER.json')['runs'])
        policy, exported = export_module.export(iteration)
        assert exported['actor_export_max_error'] <= 2e-5
        for source in [policy, policy.parent/'EXPORT_CHECK.json', TRAIN/'ppo'/f'model_{iteration-1}.pt']:
            expected[str(source.relative_to(ROOT))] = core.sha(source)
        write('RUNNING_FIXED_INTERMEDIATE_TRAIN128', iteration=iteration,
            policy=str(policy), policy_sha256=core.sha(policy))
        with (HERE/f'iter{iteration}_guard.log').open('x') as stream:
            child = subprocess.Popen(command(iteration), cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2'),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        assert code == 0 and not stopping, 'Preserve failed attempt; never restart on observation timeout'
        check_sources(expected)
        summary = verify_fixed(output(iteration), policy)
        core.atomic(HERE/f'iter{iteration}_RESULT.json', dict(iteration=iteration,
            policy_sha256=core.sha(policy), independent_original_TRAIN128=summary,
            formal_acceptance=False, goal_complete=False))
        cpu_csv(output(iteration), iteration)
        entries.append(dict(iteration=iteration, policy=str(policy), summary=summary))
        if ready(summary):
            check_sources(expected)
            finish(entries, plan, ITERATIONS[index+1:])
            return
    check_sources(expected)
    finish(entries, plan, [])


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
        else:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc())
                write('FAILED_PARENT_OR_INTERMEDIATE_REVIEW', error=repr(error))
                raise
