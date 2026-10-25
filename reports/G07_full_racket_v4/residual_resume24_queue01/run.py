"""Conditionally resume the native pilot after its independently checked TRAIN evaluation."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PILOT = REPORTS / 'residual_ppo_pilot_train128_1230'
PRIOR_QUEUE = REPORTS / 'residual_eval_queue_1250'
PRIOR_EVAL = REPORTS / 'residual_ppo_fixed_train128_1250'
TRAIN = REPORTS / 'residual_ppo_resume24_train128_01'
FINAL_EVAL = REPORTS / 'residual_ppo_fixed_train128_iter32_01'
BASE = REPORTS / 'cem32_lateral_feedback_1015/best_candidate.pt'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
STATE = HERE / 'state.json'
PRIOR_PID = 1560842
MIN_AIR = 10
MIN_RECOVERY = 3
stopping = False
guard = None
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        previous_queue=str(PRIOR_QUEUE), pilot=str(PILOT),
        train_output=str(TRAIN), evaluation_output=str(FINAL_EVAL),
        formal_acceptance=False, **extra)
    atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def live(pid, needle):
    proc = Path(f'/proc/{pid}')
    try:
        state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    if state in ('Z', 'X'):
        return False
    if needle not in command:
        raise RuntimeError(f'PID {pid} identity changed; refusing PID reuse')
    return True


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def source_hashes():
    paths = sorted(ROOT.glob('quadsmash/**/*.py')) + sorted(ROOT.glob('configs/*.json')) + sorted(ROOT.glob('assets/*.usd')) + [
        ROOT / 'scripts/train_full_racket_residual_ppo_v4.py',
        ROOT / 'scripts/evaluate_full_racket_train_v4.py',
        ROOT / 'scripts/run_gpu_job.py',
        ROOT / 'configs/g07_full_racket_eval_v4_r3.json',
        ROOT / 'configs/g07_full_racket_residual_ppo_v4.json',
        Path(__file__).resolve()]
    return {str(p): sha(p) for p in paths}


def verify_job(output, steps):
    job = read(output / 'job.json')
    assert job['status'] == 'SUCCEEDED', job['status']
    assert job['charged_env_steps'] == read(output / 'completed.json')['env_steps'] == steps
    assert not live(job['pid'], output.name)
    assert not live(job['child_pid'], output.name)
    assert not (output / 'error.txt').exists()
    return job


def verify_evaluation(output, policy):
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    inputs = read(output / 'INPUTS.json')
    assert inputs['training_indices'] == indices == list(range(128))
    assert inputs['policy_sha256'] == sha(policy)
    assert inputs['protocol_sha256'] == sha(protocol_path)
    assert not inputs.get('formal_cases_read', False)
    assert all(v == 0 for v in inputs['reset_noise_amplitudes'].values())
    summary = summarize_training(read(output / 'episodes.json'), cases, indices, protocol)
    metrics = read(output / 'metrics.json')
    for key, value in summary.items():
        assert metrics[key] == value, f'Independent TRAIN recomputation disagrees: {key}'
    assert metrics['noise_level'] == 'none'
    assert metrics['policy_sha256'] == sha(policy)
    assert metrics['protocol_sha256'] == sha(protocol_path)
    assert metrics['actual_control_env_steps'] == 102400
    assert summary['numerical_failures'] == 0
    verify_job(output, 102400)
    return summary


def verify_training(output, additional_updates, initial_update):
    import torch
    steps = 128 * 128 * additional_updates
    verify_job(output, steps)
    metrics = read(output / 'metrics.json')
    assert metrics['completed_updates'] == additional_updates
    assert len(metrics['history']) == additional_updates
    assert [r['update'] for r in metrics['history']] == list(range(initial_update + 1, initial_update + additional_updates + 1))
    assert metrics['history'][-1]['actual_control_env_steps'] == steps
    assert metrics['counts']['numerical_failures'] == 0
    assert finite(metrics['history'])
    policy = torch.load(output / 'policy.pt', map_location='cpu', weights_only=False)
    native = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == initial_update + additional_updates
    assert native['native']['canonical_restart']
    assert native['native']['environment']['num_envs'] == 128
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    assert all(torch.isfinite(v).all() for v in native['model_state_dict'].values() if torch.is_tensor(v))
    for state in native['optimizer_state_dict']['state'].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    assert policy['schema'] == 'full-racket-residual-ppo-v4' and policy['native_execution']
    assert policy['asset_sha256'] == 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    assert metrics['policy_sha256'] == sha(output / 'policy.pt')
    meta = policy['residual_training_metadata']
    assert meta['completed_updates'] == native['iter']
    assert meta['actual_control_env_steps'] == steps
    assert meta['base_policy_sha256'] == sha(BASE) == '51a8b19b17e19e60bd7482c7bcb7ed1d4e8b7f50b9317cac3c49007089e8b62d'
    assert math.isfinite(meta['export_actor_max_error']) and meta['export_actor_max_error'] <= 2e-5
    assert hashlib.sha256(policy['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    return dict(policy_sha256=sha(output / 'policy.pt'), checkpoint_sha256=sha(output / 'checkpoint.pt'),
                iteration=native['iter'], actual_control_env_steps=steps,
                changing_policy_counts=metrics['counts'], frozen_performance=False)


def run(output, seconds, steps, script, arguments, state, hashes):
    global guard
    if stopping:
        raise InterruptedError('Queue cancelled before launch')
    assert source_hashes() == hashes, 'Source changed; review before continuing'
    assert not output.exists(), 'Use a fresh output; never restart a queue into existing evidence'
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07',
        '--budget-window', 'g07-full-racket-unlimited-20261024',
        '--max-seconds', str(seconds), '--env-steps-cap', str(steps),
        '--output', str(output), '--require-artifact', 'completed.json', '--',
        PYTHON, script, '--output', str(output), *arguments, '--headless']
    with (HERE / (output.name + '.guard.log')).open('w') as log:
        guard = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(state, guard_pid=guard.pid, command=command, source_hashes=hashes)
        code = guard.wait()
    if stopping:
        raise InterruptedError('Queue cancelled during job; inspect its checkpoint and receipt')
    assert code == 0, f'Guard failed with exit {code}; do not restart automatically'
    verify_job(output, steps)


def stop(*_):
    global stopping
    stopping = True
    if guard is not None and guard.poll() is None:
        guard.terminate()


def main():
    assert not STATE.exists() and not TRAIN.exists() and not FINAL_EVAL.exists()
    assert live(PRIOR_PID, str(PRIOR_QUEUE.relative_to(ROOT) / 'run.py'))
    hashes = source_hashes()
    write('WAITING_FOR_PILOT_AND_FIXED_TRAIN_EVALUATION', verified_previous_queue_pid=PRIOR_PID,
          source_hashes=hashes,
          continuation_protection=dict(minimum_air_contacts=MIN_AIR, minimum_contact_recoveries=MIN_RECOVERY,
            numerical_failures=0, scope='TRAIN continuation decision only; original formal thresholds unchanged'))
    while not stopping and live(PRIOR_PID, str(PRIOR_QUEUE.relative_to(ROOT) / 'run.py')):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Queue cancelled while waiting; predecessor remains untouched')
    assert source_hashes() == hashes, 'Source changed before continuation decision'
    assert read(PRIOR_QUEUE / 'state.json')['state'] == 'SUCCEEDED'
    pilot = verify_training(PILOT, 8, 0)
    summary = verify_evaluation(PRIOR_EVAL, PILOT / 'policy.pt')
    continue_training = summary['airborne_contacts'] >= MIN_AIR and summary['conditional_recovery_count'] >= MIN_RECOVERY
    atomic(HERE / 'DECISION.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        independent_fixed_TRAIN128=summary, pilot=pilot, continue_training=continue_training,
        continuation_protection=dict(minimum_air_contacts=MIN_AIR, minimum_contact_recoveries=MIN_RECOVERY),
        rationale='Six learned actor updates are a pilot. Continue only if fixed-policy contact/recovery has not collapsed; reassess after another24updates.',
        formal_thresholds_unchanged=True, formal_cases_read=False, formal_acceptance=False))
    if not continue_training:
        write('STOPPED_FOR_TRAIN_PERFORMANCE_REVIEW', fixed_TRAIN128=summary)
        return
    run(TRAIN, 10800, 393216, 'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint', str(BASE), '--resume', str(PILOT / 'checkpoint.pt'),
         '--num-envs', '128', '--updates', '24', '--seed', '748501'],
        'RUNNING_NATIVE_RESUME24', hashes)
    trained = verify_training(TRAIN, 24, 8)
    atomic(HERE / 'TRAINING_RECEIPT.json', trained)
    run(FINAL_EVAL, 2700, 102400, 'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint', str(TRAIN / 'policy.pt'), '--noise', 'none'],
        'RUNNING_FIXED_TRAIN128_ITER32', hashes)
    final_summary = verify_evaluation(FINAL_EVAL, TRAIN / 'policy.pt')
    atomic(HERE / 'FIXED_TRAIN128_RESULT.json', final_summary)
    write('SUCCEEDED', training=trained, independent_fixed_TRAIN128=final_summary,
          task_status='Original G07 and full deliverables remain unproven until separately verified')


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
