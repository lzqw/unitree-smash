"""Continue the learner after verified iter32 assessment, then reassess iter96.

This queue only launches after the existing native owner exits. Batch growth
uses the original explicit learner fork on fresh ground, without transferring
old physical buffers or changing any physical, rule, reward or action contract.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT_PID = 1566225
PARENT = REPORTS / 'residual_resize512_queue01'
module_spec = importlib.util.spec_from_file_location('previous_native_workflow', PARENT / 'run.py')
previous = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(previous)
core = previous.core
PRIOR_TRAIN = previous.TRAIN
PRIOR_EVAL = previous.EVALUATION
TRAIN = REPORTS / 'residual_ppo_n2048_resume64_01'
EVALUATION = REPORTS / 'residual_ppo_fixed_train128_iter96_n2048_01'
STATE = HERE / 'state.json'
NUM_ENVS, UPDATES, INITIAL_ITERATION = 2048, 64, 32
STEPS = NUM_ENVS * 128 * UPDATES
PREVIOUS_CUMULATIVE_STEPS = 131072 + previous.STEPS
stopping = False
guard = None


def hashes():
    result = previous.hashes()
    for path in [Path(__file__).resolve(), PARENT / 'run.py',
                 ROOT / 'scripts/export_full_racket_train_csv_v4.py']:
        result[str(path)] = core.sha(path)
    return result


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN), evaluation_output=str(EVALUATION),
        formal_cases_read=False, formal_acceptance=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def decide(summary):
    """Predeclared internal TRAIN continuation checks; never formal acceptance."""
    if summary['numerical_failures'] != 0:
        return 'STOP_FOR_NUMERICAL_FAILURE'
    if (summary['complete_success_rate'] >= .5 and summary['airborne_contact_rate'] >= .7
            and summary['conditional_recovery_rate'] >= .85):
        return 'READY_FOR_FORMAL_REVIEW'
    if summary['airborne_contacts'] < 10 or summary['conditional_recovery_count'] < 3:
        return 'STOPPED_FOR_TRAIN_PERFORMANCE_REVIEW'
    return 'CONTINUE_NATIVE_LEARNING'


def confirm_previous_contact():
    import numpy as np
    milestone = REPORTS / 'residual_n512_first_complete_20261025/complete_episode.json'
    preserved = core.read(milestone)
    actual = [r for r in core.read(PRIOR_TRAIN / 'episodes.json')
              if r['env_id'] == 275 and r['episode_index'] == 1]
    assert len(actual) == 1 and actual[0] == preserved
    contacts_path = PRIOR_TRAIN / 'contacts.json'
    contacts = [c for c in core.read(contacts_path) if c['env_id'] == 275 and c['episode_index'] == 1]
    assert len(contacts) == preserved['contact_count'] == 1
    event = contacts[0]
    assert event['string_contact'] and not event['frame_contact'] and not event['fixture_contact']
    assert event['first_racket_contact'] and event['legal_initial_contact']
    assert not any(event['fault_flags'].values())
    assert abs(event['time_s'] - preserved['contact_time_s']) <= 1e-7
    assert np.max(abs(np.asarray(event['point_m']) - preserved['contact_point_m'])) <= 1e-7
    poses = np.asarray(event['body_link_poses_local'])
    assert poses.shape == (28, 7) and np.isfinite(poses).all()
    core.atomic(HERE / 'PREVIOUS_TRAINING_COMPLETE_CONTACT_CONFIRMATION.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), native_case_id=preserved['case_id'],
        preserved_episode_sha256=core.sha(milestone), original_contacts_sha256=core.sha(contacts_path),
        exact_contact_record_matches=True, actual_native_contact=event,
        point_velocity_scope='Post-PhysX contact integration state, not independently measured incident impact velocity.',
        full_motion_trajectory_available=False, frozen_policy_performance=False, formal_acceptance=False))


def run(output, seconds, steps, script, arguments, label, expected):
    global guard
    if stopping:
        raise InterruptedError('Cancelled before native launch')
    assert hashes() == expected, 'Source changed; review before launch'
    assert not output.exists(), 'Never reuse a native evidence directory'
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds),
        '--env-steps-cap', str(steps), '--output', str(output), '--require-artifact', 'completed.json',
        '--', PYTHON, script, '--output', str(output), *arguments, '--headless']
    with (HERE / (label + '.guard.log')).open('w') as log:
        guard = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(label, guard_pid=guard.pid, command=command, source_hashes=expected)
        code = guard.wait()
    if stopping:
        raise InterruptedError('Cancelled during native job; inspect its actual checkpoint/receipt')
    assert code == 0, f'Guard exit{code}; no automatic retry'
    core.verify_job(output, steps)


def verify_training():
    import numpy as np
    import torch
    metrics = core.read(TRAIN / 'metrics.json')
    assert metrics['completed_updates'] == len(metrics['history']) == UPDATES
    final_iteration = INITIAL_ITERATION + UPDATES
    assert [r['update'] for r in metrics['history']] == list(range(INITIAL_ITERATION + 1, final_iteration + 1))
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS
    assert core.finite(metrics['history']) and metrics['counts']['numerical_failures'] == 0
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == final_iteration and native['native']['canonical_restart']
    assert native['native']['environment']['num_envs'] == NUM_ENVS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert native['native']['total_timesteps'] == PREVIOUS_CUMULATIVE_STEPS + STEPS
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in native['optimizer_state_dict']['state'].values()
               for v in s.values() if torch.is_tensor(v))
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert core.sha(TRAIN / 'policy.pt') == metrics['policy_sha256']
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == final_iteration and meta['actual_control_env_steps'] == STEPS
    assert meta['num_envs'] == NUM_ENVS and meta['base_policy_sha256'] == core.sha(core.BASE)
    assert meta['export_actor_max_error'] <= 2e-5 and core.finite(meta['export_actor_max_error'])
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    initial = meta['initialization']
    assert initial == metrics['initialization'] and initial['mode'] == 'canonical_learner_fork_on_fresh_ground'
    assert initial['initial_iteration'] == INITIAL_ITERATION
    assert initial['previous_num_envs'] == 512 and initial['num_envs'] == NUM_ENVS
    assert initial['policy_optimizer_normalizers_rng_preserved'] and not initial['old_physical_state_restored']
    assert initial['previous_learner_control_env_steps'] == PREVIOUS_CUMULATIVE_STEPS
    inputs = core.read(TRAIN / 'INPUTS.json')
    assert inputs['resume_checkpoint_sha256'] == core.sha(PRIOR_TRAIN / 'checkpoint.pt')
    ground = core.read(TRAIN / 'initial_ground_states.json')
    assert ground['all_episode_clocks_zero']
    assert ground['source_training_indices'] == list(range(128)) * (NUM_ENVS // 128)
    roots = np.asarray(ground['root_state_local'])
    expected_root = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')['initialization']['root_position_m']
    assert roots.shape == (NUM_ENVS, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(expected_root))) <= 1e-5
    assert np.max(abs(roots[:, 3:7] - np.array([1., 0., 0., 0.]))) <= 1e-5
    assert np.max(abs(roots[:, 7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    return dict(policy_sha256=core.sha(TRAIN / 'policy.pt'), checkpoint_sha256=core.sha(TRAIN / 'checkpoint.pt'),
        iteration=final_iteration, num_envs=NUM_ENVS, distinct_TRAIN_cases=128,
        repeated_lanes_per_case=NUM_ENVS // 128, actual_control_env_steps=STEPS,
        cumulative_learner_control_env_steps=PREVIOUS_CUMULATIVE_STEPS + STEPS,
        initialization=initial, changing_policy_counts=metrics['counts'], frozen_performance=False)


def export_csvs(expected):
    exports = []
    for source, stem in [(TRAIN, 'iter96_training_episodes'), (EVALUATION, 'iter96_fixed_episodes')]:
        assert not stopping and hashes() == expected
        command = [PYTHON, 'scripts/export_full_racket_train_csv_v4.py', '--run', str(source),
            '--output-prefix', str(HERE / stem)]
        with (HERE / (stem + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
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
    assert not any(p.exists() for p in [STATE, TRAIN, EVALUATION])
    assert core.live(PARENT_PID, 'residual_resize512_queue01/run.py')
    expected = hashes()
    write('WAITING_FOR_FIXED_TRAIN128_ITER32', verified_parent_pid=PARENT_PID, source_hashes=expected,
        planned_num_envs=NUM_ENVS, distinct_TRAIN_cases=128, repeated_lanes_per_case=16,
        additional_updates=UPDATES, target_iteration=96, planned_control_env_steps=STEPS,
        maximum_job_seconds=28800,
        continuation_protection=dict(minimum_air_contacts=10, minimum_conditional_recoveries=3, numerical_failures=0,
            purpose='Internal collapse check, not an acceptance threshold'),
        rationale='512-env rollouts spend about211 seconds collecting versus0.09 seconds learning, with about11GiB GPU memory free. Test larger batching without claiming throughput or performance improvement before measurement.')
    while not stopping and core.live(PARENT_PID, 'residual_resize512_queue01/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; predecessor left untouched')
    parent = core.read(PARENT / 'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_NOT_SUCCEEDED', parent_state=parent['state'])
        return
    assert hashes() == expected
    trained = previous.verify_training()
    summary = core.verify_evaluation(PRIOR_EVAL, PRIOR_TRAIN / 'policy.pt')
    confirm_previous_contact()
    decision = decide(summary)
    core.atomic(HERE / 'DECISION.json', dict(prior_training=trained, independent_fixed_TRAIN128=summary,
        decision=decision, continuation_num_envs=NUM_ENVS, additional_updates=UPDATES,
        formal_thresholds_unchanged=True, formal_cases_read=False, formal_acceptance=False))
    if decision != 'CONTINUE_NATIVE_LEARNING':
        write(decision, independent_fixed_TRAIN128=summary, note='Review this result and take the next appropriate action; G07 is not complete.')
        return
    run(TRAIN, 28800, STEPS, 'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint', str(core.BASE), '--resume', str(PRIOR_TRAIN / 'checkpoint.pt'), '--resize-ground',
         '--num-envs', str(NUM_ENVS), '--updates', str(UPDATES), '--seed', '748501'],
        'RUNNING_NATIVE_N2048_LEARNER_FORK', expected)
    trained = verify_training()
    core.atomic(HERE / 'TRAINING_RECEIPT.json', trained)
    run(EVALUATION, 2700, 102400, 'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint', str(TRAIN / 'policy.pt'), '--noise', 'none'], 'RUNNING_FIXED_TRAIN128_ITER96', expected)
    summary = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    core.atomic(HERE / 'FIXED_TRAIN128_RESULT.json', summary)
    exports = export_csvs(expected)
    write('SUCCEEDED', training=trained, independent_fixed_TRAIN128=summary, csv_exports=exports,
        task_status='Learning and diagnostic assessment completed; original G07 formal evaluation and deliverables remain unproven.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except InterruptedError as error:
        write('CANCELLED', error=str(error))
    except BaseException as error:
        write('FAILED_VALIDATION_OR_JOB', error=repr(error))
        raise
