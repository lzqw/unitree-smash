"""Verify the fresh joint-reward learner, then assess its frozen actor on TRAIN128.

This observer owns only the later assessment. It never stops or restarts the
already running training job, changes its sources, or opens formal case files.
"""
import argparse
from collections import Counter
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
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
CORE = REPORTS / 'residual_resume24_queue01/run.py'
spec = importlib.util.spec_from_file_location('original_residual_evidence_checks', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
TRAIN = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_probe8_01'
EVALUATION = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
PLAN = HERE / 'POST_TRAINING_PLAN_V2.json'
STATE = HERE / 'state_v2.json'
STEPS = 2048 * 128 * 8
stopping = False
guard = None


def save_once(path, value):
    if path.exists():
        raise FileExistsError('Preserve existing evidence: ' + str(path))
    core.atomic(path, value)


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN), evaluation_output=str(EVALUATION),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def check_sources(expected):
    for name, digest in expected.items():
        assert core.sha(ROOT / name) == digest, 'Source changed: ' + name


def evaluation_command():
    return [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024',
        '--max-seconds', '2700', '--env-steps-cap', '102400',
        '--output', str(EVALUATION), '--require-artifact', 'completed.json', '--',
        PYTHON, str(ROOT / 'scripts/evaluate_full_racket_train_v4.py'),
        '--output', str(EVALUATION), '--checkpoint', str(TRAIN / 'policy.pt'),
        '--noise', 'none', '--seed', '724201', '--headless']


def prepare():
    assert not PLAN.exists() and not EVALUATION.exists() and not STATE.exists()
    launch = core.read(HERE / 'LAUNCH_RECEIPT.json')
    expected = core.read(TRAIN / 'source_hashes.json')
    for path in [CORE, Path(__file__).resolve(), HERE / 'PREDECLARED_PLAN.json',
                 HERE / 'LAUNCH_RECEIPT.json', HERE / 'ROOT_NATIVE_STARTUP_CHECK.json']:
        expected[str(path.relative_to(ROOT))] = core.sha(path)
    check_sources(expected)
    inputs = core.read(TRAIN / 'INPUTS.json')
    startup = core.read(HERE / 'ROOT_NATIVE_STARTUP_CHECK.json')
    assert startup['native_ground_initialization_verified']
    assert inputs['num_envs'] == 2048 and inputs['requested_updates'] == 8
    assert inputs['initialization']['mode'] == 'fresh_learner_with_transferred_public_actor'
    assert inputs['initialization']['initial_iteration'] == 0
    assert inputs['contract']['reward_version'] == 'native-full-chain-residual-v2-joint-placement'
    assert inputs['contract']['joint_placement_reward'] == {'weight': 20., 'distance_scale_m': .25}
    ledger = core.read(ROOT / 'status/GPU_LEDGER.json')
    records = [r for r in ledger['runs'] if r.get('output') == str(TRAIN)]
    assert len(records) == 1
    record = records[0]
    assert record['pid'] == launch['guard_pid']
    live_guard = core.live(record['pid'], str(TRAIN))
    live_child = core.live(record['child_pid'], str(TRAIN))
    assert (record['status'] == 'RUNNING' and live_guard and live_child) or (
        record['status'] == 'SUCCEEDED' and not live_guard and not live_child)
    progress = core.read(TRAIN / 'progress.json') if (TRAIN / 'progress.json').exists() else None
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        state_at_creation='PREDECLARED_FIXED_TRAIN128_NOT_STARTED',
        verified_guard_pid=record['pid'], verified_native_child_pid=record['child_pid'],
        training_guard_live_at_registration=live_guard,
        training_child_live_at_registration=live_child,
        training_progress_at_registration=progress, source_hashes=expected,
        command=evaluation_command(), evaluation_output_absent_at_registration=True,
        distinct_TRAIN_cases=128, first_episode_per_case=True, actual_ground_reset=True,
        reset_noise='none', seed=724201, control_env_steps=102400,
        required_training_updates=8, required_training_steps=STEPS,
        native_job_fault_timeout_s=2700,
        training_success_condition='Existing guard and child must actually exit successfully. Check canonical fresh8-update learner, finite model/Adam, export consistency, zero ground boundary clocks, all2048 first TRAIN episodes, original rules, and all source hashes before assessment.',
        original_formal_thresholds=dict(complete_success_rate=.5,
            airborne_contact_rate=.7, conditional_recovery_rate=.85, numerical_failures=0),
        next_action='Independently recompute all128 fixed outcomes and export separate TRAIN-changing-policy and fixed-policy CSVs. Record readiness for further training or original formal review; this observer does not execute A/B or claim acceptance.',
        future_video_preselection=core.read(HERE / 'PREDECLARED_PLAN.json')['future_video_preselection'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    save_once(PLAN, plan)
    save_once(HERE / 'POST_TRAINING_PREPARATION_CHECKS_V2.json', dict(
        checked_at=datetime.now(timezone.utc).isoformat(), command_prepared_not_executed=True,
        original_source_hashes_verified=len(expected), native_ground_inputs_checked=True,
        actual_training_guard_live=live_guard, actual_training_child_live=live_child,
        fixed_assessment_not_started=True, no_gpu_job_started_by_this_check=True,
        plan_sha256=core.sha(PLAN), formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), source_hashes=len(expected),
        actual_training_guard_live=live_guard, actual_training_child_live=live_child)), flush=True)


def verify_training():
    import numpy as np
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    job = core.verify_job(TRAIN, STEPS)
    inputs = core.read(TRAIN / 'INPUTS.json')
    metrics = core.read(TRAIN / 'metrics.json')
    assert metrics['completed_updates'] == len(metrics['history']) == 8
    assert [r['update'] for r in metrics['history']] == list(range(1, 9))
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS
    assert core.finite(metrics['history']) and metrics['counts']['numerical_failures'] == 0
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 8 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == STEPS
    assert native['native']['environment']['num_envs'] == 2048
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in native['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    assert native['infos']['residual_contract'] == inputs['contract']
    assert native['infos']['learner_initialization'] == metrics['initialization'] == inputs['initialization']
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert core.sha(TRAIN / 'policy.pt') == metrics['policy_sha256']
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == 8 and meta['actual_control_env_steps'] == STEPS
    assert meta['num_envs'] == 2048 and meta['base_policy_sha256'] == core.sha(core.BASE)
    assert core.finite(meta['export_actor_max_error']) and meta['export_actor_max_error'] <= 2e-5
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    initialization = meta['initialization']
    assert initialization == inputs['initialization']
    assert initialization['initial_iteration'] == 0 and initialization['source_iteration'] == 32
    assert initialization['source_checkpoint_sha256'] == '2060f15dbe0ebabbb4b4657a6906fbf40ca661564025e8e06888f55b2fe1a041'
    for name in ['critic_transferred', 'optimizer_transferred', 'source_rng_restored',
                 'old_physical_state_restored', 'previous_learning_counters_restored', 'performance_inherited']:
        assert initialization[name] is False, name
    ground = core.read(TRAIN / 'initial_ground_states.json')
    assert ground['all_episode_clocks_zero']
    assert ground['source_training_indices'] == list(range(128)) * 16
    roots = np.asarray(ground['root_state_local'])
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    expected_root = np.asarray(protocol['initialization']['root_position_m'])
    assert roots.shape == (2048, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - expected_root)) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    rows = core.read(TRAIN / 'episodes.json')
    assert core.sha(TRAIN / 'episodes.json') == core.sha(TRAIN / 'episodes.partial.json')
    assert {row['env_id'] for row in rows if row['episode_index'] == 0} == set(range(2048))
    assert all(row['ground_init'] and row['source_training_index'] == row['env_id'] % 128 for row in rows)
    assert Counter(row['env_id'] for row in rows if row['episode_index'] == 0) == Counter(range(2048))
    outcomes = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(v['airborne_contact'] for v in outcomes),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in outcomes),
        complete=sum(v['complete'] for v in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == metrics['counts'], 'Original independent TRAIN scoring disagrees'
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=core.sha(TRAIN / 'checkpoint.pt'), policy_sha256=core.sha(TRAIN / 'policy.pt'),
        actual_control_env_steps=STEPS, cumulative_new_learner_steps=STEPS,
        actual_gpu_hours=job['charged_gpu_hours'], iteration=8, num_envs=2048,
        original_TRAIN_cases=128, first_ground_episodes_verified=2048,
        initialization=initialization, independent_changing_policy_counts=counts,
        frozen_policy_performance=False, formal_acceptance=False, goal_complete=False)


def run_assessment(plan):
    global guard
    assert not stopping and not EVALUATION.exists()
    check_sources(plan['source_hashes'])
    assert plan['command'] == evaluation_command()
    with (HERE / 'fixed_TRAIN128.guard.log').open('x') as log:
        guard = subprocess.Popen(plan['command'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write('RUNNING_FIXED_TRAIN128_JOINT_REWARD_ITER8', guard_pid=guard.pid)
        code = guard.wait()
    assert code == 0 and not stopping, 'Fixed assessment failed/cancelled; inspect real receipts, no automatic retry'
    core.verify_job(EVALUATION, 102400)


def export_csvs(expected):
    exports = []
    for source, stem in [(TRAIN, 'joint_iter8_training_episodes'), (EVALUATION, 'joint_iter8_fixed_episodes')]:
        check_sources(expected)
        assert not stopping
        with (HERE / (stem + '.log')).open('x') as log:
            subprocess.run([PYTHON, 'scripts/export_full_racket_train_csv_v4.py', '--run', str(source),
                '--output-prefix', str(HERE / stem)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        manifest = core.read(HERE / (stem + '.json'))
        assert manifest['original_runtime_metrics_match'] and not manifest['formal_acceptance']
        exports.append(manifest)
    return exports


def stop(*_):
    global stopping
    stopping = True
    if guard is not None and guard.poll() is None:
        guard.terminate()


def main():
    assert PLAN.exists() and not STATE.exists() and not EVALUATION.exists()
    plan = core.read(PLAN)
    check_sources(plan['source_hashes'])
    write('WAITING_FOR_EXISTING_NATIVE_JOINT_REWARD_TRAINING',
        guard_pid=plan['verified_guard_pid'], child_pid=plan['verified_native_child_pid'],
        source_hashes=plan['source_hashes'])
    while not stopping:
        alive = [core.live(pid, str(TRAIN)) for pid in [plan['verified_guard_pid'], plan['verified_native_child_pid']]]
        if not any(alive):
            break
        time.sleep(10)
    if stopping:
        raise InterruptedError('Observer cancelled while waiting; existing training left untouched')
    check_sources(plan['source_hashes'])
    trained = verify_training()
    save_once(HERE / 'TRAINING_RECEIPT.json', trained)
    run_assessment(plan)
    check_sources(plan['source_hashes'])
    summary = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    inputs = core.read(EVALUATION / 'INPUTS.json')
    assert inputs['reset_noise_seed'] == 724201
    save_once(HERE / 'FIXED_TRAIN128_RESULT.json', summary)
    ready = (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
        and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)
    exports = export_csvs(plan['source_hashes'])
    write('SUCCEEDED', training=trained, independent_fixed_TRAIN128=summary,
        original_formal_gate_reached=ready, csv_exports=exports,
        next_action='Review original formal A/B readiness' if ready else 'Review failures and continue learning; original formal gate not reached')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    arguments = parser.parse_args()
    with (HERE / 'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if arguments.prepare_only:
            prepare()
        else:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                main()
            except InterruptedError as error:
                write('CANCELLED', error=str(error))
                sys.exit(2)
            except BaseException as error:
                (HERE / 'observer_error.txt').write_text(traceback.format_exc())
                write('FAILED_VALIDATION_OR_JOB', error=repr(error))
                raise
