"""Continue the new reward learner only after its complete frozen TRAIN128 check.

Preserve the fresh learner's critic/Adam/normalizers/RNG and counters under the
same reward contract. Use canonical ground restarts, never old physical motion.
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
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT = REPORTS / 'joint_placement_probe8_queue01'
PROBE_SOURCE = PARENT / 'after_training.py'
spec = importlib.util.spec_from_file_location('fresh_joint_reward_probe_checks', PROBE_SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
core = probe.core
TRAIN = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_resume32_01'
EVALUATION = REPORTS / 'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01'
CONFIG = ROOT / 'configs/g07_full_racket_residual_ppo_joint_placement_v4.json'
PLAN = HERE / 'PREDECLARED_PLAN.json'
STATE = HERE / 'state.json'
UPDATES, INITIAL_ITERATION, NUM_ENVS = 32, 8, 2048
STEPS = UPDATES * NUM_ENVS * 128
CUMULATIVE_STEPS = (UPDATES + INITIAL_ITERATION) * NUM_ENVS * 128
stopping = False
guard = None


def check_sources(expected):
    for name, digest in expected.items():
        assert core.sha(ROOT / name) == digest, 'Source changed: ' + name


def save_once(path, value):
    assert not path.exists(), 'Preserve existing evidence: ' + str(path)
    core.atomic(path, value)


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN), evaluation_output=str(EVALUATION),
        formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def decision(summary):
    if summary['numerical_failures'] != 0:
        return 'STOP_FOR_NUMERICAL_REVIEW'
    if (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
            and summary['conditional_recovery_rate'] >= .85):
        return 'READY_FOR_ORIGINAL_FORMAL_REVIEW'
    if summary['airborne_contacts'] < 10 or summary['conditional_recovery_count'] < 3:
        return 'STOP_FOR_FIXED_TRAIN_PERFORMANCE_REVIEW'
    return 'CONTINUE_SAME_REWARD_LEARNING_32_UPDATES'


def guarded_command(output, seconds, steps, script, arguments):
    return [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds),
        '--env-steps-cap', str(steps), '--output', str(output), '--require-artifact', 'completed.json',
        '--', PYTHON, str(ROOT / script), '--output', str(output), *arguments, '--headless']


def commands():
    train = guarded_command(TRAIN, 14400, STEPS, 'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint', str(core.BASE), '--config', str(CONFIG),
         '--resume', str(probe.TRAIN / 'checkpoint.pt'), '--num-envs', '2048',
         '--updates', '32', '--seed', '748501'])
    evaluate = guarded_command(EVALUATION, 2700, 102400, 'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint', str(TRAIN / 'policy.pt'), '--noise', 'none', '--seed', '724201'])
    return train, evaluate


def prepare():
    assert not any(p.exists() for p in [PLAN, STATE, TRAIN, EVALUATION])
    parent = core.read(PARENT / 'state_v2.json')
    assert parent['state'] == 'WAITING_FOR_EXISTING_NATIVE_JOINT_REWARD_TRAINING'
    assert core.live(parent['queue_pid'], 'joint_placement_probe8_queue01/after_training.py')
    expected = dict(core.read(PARENT / 'POST_TRAINING_PLAN_V2.json')['source_hashes'])
    for path in [Path(__file__).resolve(), PARENT / 'POST_TRAINING_PLAN_V2.json']:
        expected[str(path.relative_to(ROOT))] = core.sha(path)
    check_sources(expected)
    configuration = core.read(CONFIG)
    assert configuration['joint_placement_reward'] == {'weight': 20., 'distance_scale_m': .25}
    assert configuration['runner']['num_steps_per_env'] == 128
    train_command, evaluation_command = commands()
    progress = core.read(probe.TRAIN / 'progress.json')
    save_once(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_live_parent_pid=parent['queue_pid'], source_hashes=expected,
        parent_progress_at_registration={k: v for k, v in progress.items() if k != 'losses'},
        parent_fixed_assessment_not_started=not probe.EVALUATION.exists(),
        source_checkpoint=str(probe.TRAIN / 'checkpoint.pt'),
        source_checkpoint_must_be_validated_after_actual_completion=True,
        train_command=train_command, evaluation_command=evaluation_command,
        initial_iteration=8, additional_updates=32, target_iteration=40,
        num_envs=2048, distinct_original_TRAIN_cases=128, copies_per_case=16,
        additional_control_env_steps=STEPS, expected_cumulative_new_learner_steps=CUMULATIVE_STEPS,
        maximum_training_job_seconds=14400, maximum_assessment_job_seconds=2700,
        internal_continuation_check=dict(minimum_airborne_contacts=10, minimum_conditional_recoveries=3,
            numerical_failures=0, scope='Internal collapse review only; never formal acceptance.'),
        original_formal_gate=dict(complete_success_rate=.5, airborne_contact_rate=.7,
            conditional_recovery_rate=.85, numerical_failures=0),
        decision_order='Verify actual probe completion and every fixed TRAIN128 episode. If original formal gate is reached, do not extend learning; leave frozen actor for formal review. Otherwise continue only if the declared internal collapse check passes.',
        rationale='The initial8-update run includes2 critic-warmup updates and6 actor updates. If the new learner has not collapsed, allow additional learning under its same measured joint-placement reward, then assess the frozen actor again. No improvement is presumed.',
        restart_semantics='Strict canonical same-contract resume of model, critic, Adam, normalizers, RNG and counters; actual ground initialization, no claim of continuous old PhysX motion. Same2048 environments, no resizing or new actor transfer.',
        resource_authorization='Existing GPU0 only; original shared guard/ledger and unlimited cumulative authorization. Finite per-job fault/time/step caps remain. No new cloud.',
        physical_parameters_rules_cases_thresholds_changed=False,
        renderer_queue_can_continue_on_CPU_with_unchanged_sources=True,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    save_once(HERE / 'PREPARATION_CHECKS.json', dict(
        checked_at=datetime.now(timezone.utc).isoformat(), verified_live_parent_pid=parent['queue_pid'],
        source_hashes_verified=len(expected), same_reward_configuration_verified=True,
        future_outputs_absent=True, no_native_job_launched_by_preparation=True,
        plan_sha256=core.sha(PLAN), formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), verified_live_parent_pid=parent['queue_pid'],
        source_hashes_verified=len(expected), additional_steps=STEPS)), flush=True)


def run_job(command, output, steps, label, expected):
    global guard
    assert not stopping and not output.exists()
    check_sources(expected)
    with (HERE / (label + '.guard.log')).open('x') as log:
        guard = subprocess.Popen(command, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(label, guard_pid=guard.pid)
        code = guard.wait()
    assert code == 0 and not stopping, 'Guard failed/cancelled; inspect real artifacts, no automatic retry'
    return core.verify_job(output, steps)


def verify_resumed_training():
    import numpy as np
    import torch
    from quadsmash.evaluation_v4 import independent_episode
    job = core.verify_job(TRAIN, STEPS)
    inputs, metrics = core.read(TRAIN / 'INPUTS.json'), core.read(TRAIN / 'metrics.json')
    assert inputs['resume_checkpoint_sha256'] == core.sha(probe.TRAIN / 'checkpoint.pt')
    assert inputs['initial_actor_checkpoint_sha256'] is None and not inputs['resize_ground_requested']
    assert inputs['contract'] == core.read(probe.TRAIN / 'INPUTS.json')['contract']
    assert metrics['completed_updates'] == len(metrics['history']) == UPDATES
    assert [r['update'] for r in metrics['history']] == list(range(9, 41))
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS
    assert core.finite(metrics['history']) and metrics['counts']['numerical_failures'] == 0
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 40 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == CUMULATIVE_STEPS
    assert native['native']['environment']['num_envs'] == NUM_ENVS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for state in native['optimizer_state_dict']['state'].values()
               for v in state.values() if torch.is_tensor(v))
    initialization = inputs['initialization']
    assert initialization == metrics['initialization'] == native['infos']['learner_initialization']
    assert initialization['mode'] == 'strict_canonical_ground_resume'
    assert initialization['initial_iteration'] == 8 and initialization['num_envs'] == NUM_ENVS
    assert native['infos']['residual_contract'] == inputs['contract']
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert core.sha(TRAIN / 'policy.pt') == metrics['policy_sha256']
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == 40 and meta['actual_control_env_steps'] == STEPS
    assert meta['num_envs'] == NUM_ENVS and meta['base_policy_sha256'] == core.sha(core.BASE)
    assert meta['initialization'] == initialization
    assert core.finite(meta['export_actor_max_error']) and meta['export_actor_max_error'] <= 2e-5
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    ground = core.read(TRAIN / 'initial_ground_states.json')
    protocol = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    roots = np.asarray(ground['root_state_local'])
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128)) * 16
    assert roots.shape == (NUM_ENVS, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(protocol['initialization']['root_position_m']))) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    rows = core.read(TRAIN / 'episodes.json')
    assert core.sha(TRAIN / 'episodes.json') == core.sha(TRAIN / 'episodes.partial.json')
    assert {r['env_id'] for r in rows if r['episode_index'] == 0} == set(range(NUM_ENVS))
    assert sum(r['episode_index'] == 0 for r in rows) == NUM_ENVS
    assert all(r['ground_init'] and r['source_training_index'] == r['env_id'] % 128 for r in rows)
    outcomes = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(r['airborne_contact'] for r in outcomes),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in outcomes),
        complete=sum(r['complete'] for r in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == metrics['counts']
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), iteration=40,
        source_checkpoint_sha256=inputs['resume_checkpoint_sha256'],
        checkpoint_sha256=core.sha(TRAIN / 'checkpoint.pt'), policy_sha256=core.sha(TRAIN / 'policy.pt'),
        actual_control_env_steps=STEPS, cumulative_new_learner_control_env_steps=CUMULATIVE_STEPS,
        actual_gpu_hours=job['charged_gpu_hours'], num_envs=NUM_ENVS,
        initialization=initialization, independent_changing_policy_counts=counts,
        frozen_policy_performance=False, formal_acceptance=False, goal_complete=False)


def export_csvs(expected):
    exports = []
    for source, stem in [(TRAIN, 'joint_iter40_training_episodes'), (EVALUATION, 'joint_iter40_fixed_episodes')]:
        assert not stopping
        check_sources(expected)
        with (HERE / (stem + '.log')).open('x') as log:
            subprocess.run([PYTHON, 'scripts/export_full_racket_train_csv_v4.py', '--run', str(source),
                '--output-prefix', str(HERE / stem)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result = core.read(HERE / (stem + '.json'))
        assert result['original_runtime_metrics_match'] and not result['formal_acceptance']
        exports.append(result)
    return exports


def stop(*_):
    global stopping
    stopping = True
    if guard is not None and guard.poll() is None:
        guard.terminate()


def main():
    assert PLAN.exists() and not any(p.exists() for p in [STATE, TRAIN, EVALUATION])
    plan = core.read(PLAN)
    check_sources(plan['source_hashes'])
    write('WAITING_FOR_FROZEN_JOINT_REWARD_PROBE8', parent_pid=plan['verified_live_parent_pid'])
    while not stopping and core.live(plan['verified_live_parent_pid'], 'joint_placement_probe8_queue01/after_training.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Continuation observer cancelled; original probe jobs untouched')
    parent = core.read(PARENT / 'state_v2.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PROBE_NOT_SUCCEEDED', parent_state=parent['state'])
        return
    check_sources(plan['source_hashes'])
    trained = probe.verify_training()
    summary = core.verify_evaluation(probe.EVALUATION, probe.TRAIN / 'policy.pt')
    next_action = decision(summary)
    save_once(HERE / 'DECISION.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        verified_probe_training=trained, independent_fixed_TRAIN128=summary,
        next_action=next_action, formal_thresholds_unchanged=True,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    if next_action != 'CONTINUE_SAME_REWARD_LEARNING_32_UPDATES':
        write(next_action, independent_fixed_TRAIN128=summary)
        return
    assert (plan['train_command'], plan['evaluation_command']) == commands()
    run_job(plan['train_command'], TRAIN, STEPS, 'RUNNING_SAME_REWARD_RESUME32', plan['source_hashes'])
    result = verify_resumed_training()
    save_once(HERE / 'TRAINING_RECEIPT.json', result)
    run_job(plan['evaluation_command'], EVALUATION, 102400, 'RUNNING_FIXED_TRAIN128_JOINT_ITER40', plan['source_hashes'])
    check_sources(plan['source_hashes'])
    summary = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    assert core.read(EVALUATION / 'INPUTS.json')['reset_noise_seed'] == 724201
    save_once(HERE / 'FIXED_TRAIN128_RESULT.json', summary)
    exports = export_csvs(plan['source_hashes'])
    write('SUCCEEDED', training=result, independent_fixed_TRAIN128=summary,
        csv_exports=exports, next_action=decision(summary),
        scope='Native learning and fixed TRAIN diagnostic only; original400 formal episodes and full goal remain unproven.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with (HERE / 'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
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
                (HERE / 'error.txt').write_text(traceback.format_exc())
                write('FAILED_VALIDATION_OR_JOB', error=repr(error))
                raise
