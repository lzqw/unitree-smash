"""Continue the checked pilot on512 fresh ground environments, then freeze and assess."""
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
module_spec = importlib.util.spec_from_file_location('residual_checks', REPORTS / 'residual_resume24_queue01/run.py')
core = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(core)
STATE = HERE / 'state.json'
TRAIN = REPORTS / 'residual_ppo_n512_resume24_01'
EVALUATION = REPORTS / 'residual_ppo_fixed_train128_iter32_n512_01'
STEPS = 512 * 128 * 24
stopping = False
guard = None


def hashes():
    result = core.source_hashes()
    for p in [Path(__file__).resolve(), ROOT / 'scripts/residual_ground_resize_v4.py']:
        result[str(p)] = core.sha(p)
    return result


def write(state, **extra):
    data = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        training_output=str(TRAIN), evaluation_output=str(EVALUATION), formal_acceptance=False, **extra)
    core.atomic(STATE, data)
    print(json.dumps({k: v for k, v in data.items() if k != 'source_hashes'}), flush=True)


def run(output, seconds, steps, script, arguments, label, expected):
    global guard
    if stopping:
        raise InterruptedError('Cancelled before launch')
    assert hashes() == expected, 'Source changed; review before launching'
    assert not output.exists(), 'Do not restart into existing evidence'
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
        raise InterruptedError('Cancelled during job; inspect its checkpoint and receipt')
    assert code == 0, f'Guard exit{code}; no automatic restart'
    core.verify_job(output, steps)


def verify_training():
    import torch
    import numpy as np
    metrics = core.read(TRAIN / 'metrics.json')
    assert metrics['completed_updates'] == len(metrics['history']) == 24
    assert [r['update'] for r in metrics['history']] == list(range(9, 33))
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS
    assert core.finite(metrics['history']) and metrics['counts']['numerical_failures'] == 0
    native = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    policy = torch.load(TRAIN / 'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 32 and native['native']['canonical_restart']
    assert native['native']['environment']['num_envs'] == 512
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert native['native']['total_timesteps'] == 131072 + STEPS
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for s in native['optimizer_state_dict']['state'].values()
               for v in s.values() if torch.is_tensor(v))
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert core.sha(TRAIN / 'policy.pt') == metrics['policy_sha256']
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == 32 and meta['actual_control_env_steps'] == STEPS
    assert meta['num_envs'] == 512 and meta['base_policy_sha256'] == core.sha(core.BASE)
    assert meta['export_actor_max_error'] <= 2e-5 and core.finite(meta['export_actor_max_error'])
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    initialization = meta['initialization']
    assert initialization == metrics['initialization']
    assert initialization['mode'] == 'canonical_learner_fork_on_fresh_ground'
    assert initialization['previous_num_envs'] == 128 and initialization['num_envs'] == 512
    assert initialization['policy_optimizer_normalizers_rng_preserved']
    assert not initialization['old_physical_state_restored']
    ground = core.read(TRAIN / 'initial_ground_states.json')
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128)) * 4
    roots = np.asarray(ground['root_state_local'])
    expected_root = core.read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')['initialization']['root_position_m']
    assert roots.shape == (512, 13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:, :3] - np.asarray(expected_root))) <= 1e-5
    return dict(policy_sha256=core.sha(TRAIN / 'policy.pt'), checkpoint_sha256=core.sha(TRAIN / 'checkpoint.pt'),
        iteration=32, num_envs=512, distinct_TRAIN_cases=128, repeated_lanes_per_case=4,
        actual_control_env_steps=STEPS, initialization=initialization,
        changing_policy_counts=metrics['counts'], frozen_performance=False)


def stop(*_):
    global stopping
    stopping = True
    if guard is not None and guard.poll() is None:
        guard.terminate()


def main():
    assert not STATE.exists() and not TRAIN.exists() and not EVALUATION.exists()
    assert core.read(REPORTS / 'residual_resume24_queue01/state.json')['state'] == 'CANCELLED'
    assert not core.live(1563040, 'residual_resume24_queue01/run.py')
    assert core.live(core.PRIOR_PID, 'residual_eval_queue_1250/run.py')
    expected = hashes()
    write('WAITING_FOR_FIXED_TRAIN128_ITER8', verified_previous_queue_pid=core.PRIOR_PID,
        source_hashes=expected, num_envs=512, additional_updates=24, planned_control_env_steps=STEPS,
        continuation_protection=dict(minimum_air_contacts=10, minimum_conditional_recoveries=3,
            numerical_failures=0, scope='Internal TRAIN continuation decision; original G07 gates unchanged'))
    while not stopping and core.live(core.PRIOR_PID, 'residual_eval_queue_1250/run.py'):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; predecessor untouched')
    assert hashes() == expected
    assert core.read(core.PRIOR_QUEUE / 'state.json')['state'] == 'SUCCEEDED'
    pilot = core.verify_training(core.PILOT, 8, 0)
    initial_summary = core.verify_evaluation(core.PRIOR_EVAL, core.PILOT / 'policy.pt')
    proceed = initial_summary['airborne_contacts'] >= 10 and initial_summary['conditional_recovery_count'] >= 3
    core.atomic(HERE / 'DECISION.json', dict(pilot=pilot, independently_recomputed_fixed_TRAIN128=initial_summary,
        continue_training=proceed, num_envs=512, distinct_TRAIN_cases=128, original_formal_thresholds_unchanged=True,
        formal_cases_read=False, formal_acceptance=False))
    if not proceed:
        write('STOPPED_FOR_TRAIN_PERFORMANCE_REVIEW', independent_fixed_TRAIN128=initial_summary)
        return
    run(TRAIN, 10800, STEPS, 'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint', str(core.BASE), '--resume', str(core.PILOT / 'checkpoint.pt'), '--resize-ground',
         '--num-envs', '512', '--updates', '24', '--seed', '748501'], 'RUNNING_NATIVE_N512_LEARNER_FORK', expected)
    trained = verify_training()
    core.atomic(HERE / 'TRAINING_RECEIPT.json', trained)
    run(EVALUATION, 2700, 102400, 'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint', str(TRAIN / 'policy.pt'), '--noise', 'none'], 'RUNNING_FIXED_TRAIN128_ITER32', expected)
    final_summary = core.verify_evaluation(EVALUATION, TRAIN / 'policy.pt')
    core.atomic(HERE / 'FIXED_TRAIN128_RESULT.json', final_summary)
    write('SUCCEEDED', training=trained, independent_fixed_TRAIN128=final_summary,
        task_status='Workflow completed; original G07 and full deliverables require separate proof')


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
