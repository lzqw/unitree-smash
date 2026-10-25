"""Run two predeclared causal feedback probes after the live joint40 workflow.

This observer never cancels/restarts its parent and never writes old sources,
old scores, the shared ledger or old runtime state. Each native probe uses the
original GPU0 guard; its own complete failures and all public streams remain.
"""
import argparse
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
REPORTS = ROOT/'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT = REPORTS/'joint_placement_resume32_queue01'
PARENT_TRAIN = REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01'
PARENT_EVAL = REPORTS/'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01'
CORE = REPORTS/'residual_resume24_queue01/run.py'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('preserved_original_train_checks', CORE)
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
PLAN, STATE = HERE/'PREDECLARED_PLAN.json', HERE/'state.json'
stopping, guard = False, None


def save_once(path, value):
    assert not path.exists(), 'Preserve evidence: '+str(path)
    core.atomic(path, value)


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps(value), flush=True)


def check_sources(expected):
    for name, expected_sha in expected.items():
        assert core.sha(ROOT/name) == expected_sha, 'Pinned source changed: '+name


def command(variant):
    output = HERE/(variant['name']+'_native_train128')
    return [PYTHON, str(ROOT/'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024', '--max-seconds', '2700',
        '--env-steps-cap', '102400', '--output', str(output), '--require-artifact', 'completed.json', '--',
        PYTHON, str(HERE/'native_probe.py'), '--output', str(output),
        '--checkpoint', str(ROOT/variant['proposal']), '--noise', 'none', '--seed', '724201', '--headless']


def prepare():
    assert not PLAN.exists() and not STATE.exists()
    preparation = core.read(HERE/'PREPARATION.json')
    assert not preparation['native_execution'] and len(preparation['variants']) == 2
    parent = core.read(PARENT/'state.json')
    assert parent['state'] == 'RUNNING_SAME_REWARD_RESUME32'
    assert core.live(parent['queue_pid'], str(PARENT.relative_to(ROOT)/'run.py'))
    assert core.live(parent['guard_pid'], PARENT_TRAIN.name)
    ledger = core.read(ROOT/'status/GPU_LEDGER.json')
    active = [r for r in ledger['runs'] if r['status'] == 'RUNNING']
    assert len(active) == 1 and active[0]['pid'] == parent['guard_pid']
    assert core.live(active[0]['child_pid'], PARENT_TRAIN.name)
    expected = dict(core.read(PARENT/'PREDECLARED_PLAN.json')['source_hashes'])
    extras = [CORE, HERE/'reactive.py', HERE/'prepare.py', HERE/'native_probe.py', Path(__file__).resolve(),
        HERE/'PREPARATION.json', PARENT/'PREDECLARED_PLAN.json',
        ROOT/preparation['variants'][0]['proposal'], ROOT/preparation['variants'][1]['proposal'],
        ROOT/'reports/G07_full_racket_v4/joint8_contact_centering_20261025/REPORT.json',
        ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',
        ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01/metrics.json',
        ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01/episodes.json']
    for path in extras:
        expected[str(path.relative_to(ROOT))] = core.sha(path)
    check_sources(expected)
    for variant in preparation['variants']:
        assert not (HERE/(variant['name']+'_native_train128')).exists()
        assert core.sha(ROOT/variant['proposal']) == variant['proposal_sha256']
    save_once(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=parent['queue_pid'], verified_live_guard_pid=parent['guard_pid'],
        verified_live_native_child_pid=active[0]['child_pid'],
        parent_training_progress={k:v for k,v in core.read(PARENT_TRAIN/'progress.json').items() if k!='losses'},
        parent_frozen_evaluation_not_yet_started=not PARENT_EVAL.exists(),
        hypotheses='Live ball/tool relative-state correction may move upper-frame contacts onto existing strings. Static inward retarget already failed historically; the current teacher already uses fixed-plan FK/DLS/60ms lead.',
        preparation=preparation, source_hashes=expected,
        commands=[command(v) for v in preparation['variants']],
        decision_order='Wait for the actual live parent workflow to exit successfully; independently verify all fixed joint40 TRAIN128 outcomes. If original formal readiness is reached, skip these diagnostic probes. Otherwise execute both prepared fixed proposals sequentially on all128 original TRAIN cases.',
        comparison_baseline='Completed joint8 fixed TRAIN128, same original physical asset/protocol, seed724201 and no reset noise; no inherited performance.',
        selection='Retain both complete result sets. Report full/air/qualified recovery and each failure; lexicographic full, air, recovery is a TRAIN diagnostic ranking only. No statistical superiority or formal pass is inferred.',
        original_formal_thresholds=dict(complete_success_rate=.5, airborne_contact_rate=.7,
            conditional_recovery_rate=.85, numerical_failures=0),
        each_native_job_control_steps_cap=102400, each_native_job_seconds_cap=2700,
        native_proposal_must_remain_false_until_completed_execution=True,
        resource_authorization='Existing GPU0 only, original cumulative GPU ledger/guard and unlimited authorization; no extra cloud or concurrent native training.',
        physical_parameters_rules_cases_thresholds_changed=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    save_once(HERE/'QUEUE_PREPARATION_CHECKS.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=parent['queue_pid'], verified_live_native_child_pid=active[0]['child_pid'],
        source_files_verified=len(expected), plan_sha256=core.sha(PLAN),
        no_new_native_job_started=True, both_proposals_CPU_replayed=True, formal_acceptance=False))
    print(json.dumps(dict(prepared=str(PLAN), source_files_verified=len(expected), no_native_job_started=True)), flush=True)


def original_formal_ready(summary):
    return (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
        and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def verify_probe(variant, output):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    job = core.read(output/'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0
    assert not core.live(job['pid'], output.name) and not core.live(job['child_pid'], output.name)
    assert not (output/'error.txt').exists()
    steps = core.read(output/'completed.json')['env_steps']
    assert steps == job['charged_env_steps'] and 0 < steps <= 102400
    inputs = core.read(output/'INPUTS.json')
    assert inputs['unexecuted_proposal'] and inputs['policy_sha256'] == variant['proposal_sha256']
    assert inputs['training_indices'] == list(range(128)) and inputs['reset_noise_seed'] == 724201
    assert all(v == 0. for v in inputs['reset_noise_amplitudes'].values())
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = core.read(protocol_path); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    rows = core.read(output/'episodes.json')
    assert len(rows) == 128 and {r['source_training_index'] for r in rows} == set(range(128))
    summary = summarize_training(rows, cases, indices, protocol)
    metrics = core.read(output/'metrics.json')
    assert all(metrics[k] == v for k,v in summary.items())
    assert metrics['actual_control_env_steps'] == steps
    assert metrics['protocol_sha256'] == inputs['protocol_sha256'] == core.sha(protocol_path)
    assert metrics['policy_sha256'] == variant['proposal_sha256']
    # Every proposed action and observation has its actual native counterpart.
    trace = np.load(output/'public_and_actions_all.npz')
    observations, actions = trace['observations'], trace['actions']
    assert observations.shape[1:] == (128, 794) and actions.shape == (len(observations), 128, 18)
    assert len(observations)*128 == steps
    assert np.isfinite(observations).all() and np.isfinite(actions).all() and np.max(abs(actions)) <= 1.
    proposed = torch.load(ROOT/variant['proposal'], map_location='cpu', weights_only=False)
    executed = torch.load(output/'executed_policy.pt', map_location='cpu', weights_only=False)
    assert not proposed['native_execution'] and executed['native_execution']
    assert proposed['residual_actor_jit'] == executed['residual_actor_jit']
    assert executed['native_trial_execution']['proposal_sha256'] == variant['proposal_sha256']
    with (output/'csv_export.log').open('x') as log:
        subprocess.run([PYTHON, str(ROOT/'scripts/export_full_racket_train_csv_v4.py'), '--run', str(output),
            '--output-prefix', str(output/'episodes_export')], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    csv = core.read(output/'episodes_export.json')
    assert csv['original_runtime_metrics_match'] and not csv['formal_acceptance']
    return dict(name=variant['name'], gain=variant['gain'], independent_fixed_TRAIN128=summary,
        native_job_gpu_hours=job['charged_gpu_hours'], actual_control_env_steps=steps,
        proposal_sha256=variant['proposal_sha256'], executed_policy_sha256=core.sha(output/'executed_policy.pt'),
        episodes_sha256=core.sha(output/'episodes.json'), public_trace_sha256=core.sha(output/'public_and_actions_all.npz'),
        all128_public_and_action_streams_preserved=True, original_proposal_not_overwritten=True,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)


def stop(*_):
    global stopping
    stopping = True
    if guard is not None and guard.poll() is None:
        guard.terminate()


def main():
    global guard
    assert PLAN.exists() and not STATE.exists(), 'Never restart an existing observer/output'
    plan = core.read(PLAN); check_sources(plan['source_hashes'])
    write('WAITING_FOR_LIVE_JOINT40_WORKFLOW', parent_pid=plan['verified_live_parent_pid'])
    while not stopping and core.live(plan['verified_live_parent_pid'], str(PARENT.relative_to(ROOT)/'run.py')):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Observer cancelled; parent training was not touched')
    parent = core.read(PARENT/'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_DID_NOT_SUCCEED', parent_state=parent['state']); return
    core.verify_job(PARENT_TRAIN, 8388608)
    summary = core.verify_evaluation(PARENT_EVAL, PARENT_TRAIN/'policy.pt')
    save_once(HERE/'PARENT_FIXED_TRAIN128_CHECK.json', summary)
    if original_formal_ready(summary):
        write('SKIPPED_PARENT_READY_FOR_ORIGINAL_FORMAL_REVIEW', independent_fixed_TRAIN128=summary); return
    results = []
    for variant, cmd in zip(plan['preparation']['variants'], plan['commands']):
        assert not stopping and cmd == command(variant)
        check_sources(plan['source_hashes'])
        output = HERE/(variant['name']+'_native_train128')
        assert not output.exists()
        with (HERE/(variant['name']+'.guard.log')).open('x') as log:
            guard = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2'), start_new_session=True)
            write('RUNNING_NATIVE_PUBLIC_TANGENT_PROBE', variant=variant['name'], guard_pid=guard.pid, output=str(output))
            while not stopping and guard.poll() is None:
                time.sleep(10)
            code = guard.wait()
        if stopping:
            raise InterruptedError('Current proposal guard terminated at observer cancellation')
        assert code == 0, 'Native proposal guard failed; preserve output for review'
        check_sources(plan['source_hashes'])
        receipt = verify_probe(variant, output)
        save_once(HERE/(variant['name']+'_RESULT.json'), receipt)
        results.append(receipt)
        assert receipt['independent_fixed_TRAIN128']['numerical_failures'] == 0
    write('SUCCEEDED', results=results,
        scope='Two fixed original TRAIN128 first-execution probes; no formal A/B cases read and original full goal still unproven.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
        else:
            signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
            try:
                main()
            except InterruptedError as error:
                write('CANCELLED', error=str(error)); sys.exit(2)
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc())
                write('FAILED_VALIDATION_OR_JOB', error=repr(error)); raise
