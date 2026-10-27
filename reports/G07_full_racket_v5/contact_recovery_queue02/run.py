"""Use the Main GPU0 guard for a pinned recovery revision in its own worktree."""
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
BASE = ROOT / 'reports/G07_full_racket_v4'
OUTPUTS = ROOT / 'reports/G07_full_racket_v5'
COLD_SOURCE = OUTPUTS / 'teacher_contact_recovery_cold128_02'
TRAIN = OUTPUTS / 'residual_contact_recovery_n128_updates15_02'
COLD_NEW = OUTPUTS / 'residual_contact_recovery15_cold128_02'
TEACHER = BASE / 'gain_body8_sequential_n128_01/best_candidate.pt'
PRECISION = BASE / 'origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
child = None
stopping = False


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def save(path, value):
    assert not path.exists(), str(path)
    atomic(path, value)


def state(name, **extra):
    atomic(HERE / 'state.json', dict(state=name, owner_pid=os.getpid(),
        observed_at=datetime.now(timezone.utc).isoformat(), formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, **extra))


def check(expected):
    for name, digest in expected.items():
        assert sha(Path(name)) == digest, name


def prepare():
    old = read(BASE / 'recovery_sequence_observer_20261027/state.json')
    assert old['state'] == 'COMPLETED_READ_ONLY_SEQUENCE_REPLAY'
    assert read(OUTPUTS / "contact_recovery_queue01/state.json")["state"] == "FAILED_PRESERVED_FOR_REVIEW"
    assert read(OUTPUTS / "teacher_contact_recovery_cold128_01/job.json")["status"] == "FAILED"
    for pid in [1930543, 1932397, 1932411, 1939051, 1939053, 1939067]:
        assert not Path('/proc', str(pid)).exists()
    expected = dict(read(BASE / 'recovery_sequence_observer_20261027/QUEUE_PLAN.json')['source_sha256'])
    for pattern in ['quadsmash/**/*.py', 'scripts/*.py', 'configs/*.json', 'assets/*.usd']:
        for path in WORK.glob(pattern):
            expected[str(path)] = sha(path)
    check_dir = WORK / 'reports/G07_full_racket_v5/contact_recovery_revision_20261027'
    for path in [Path(__file__), check_dir / 'check_recorded_samples.py',
                 check_dir / 'ACTUAL_RECORDED_INPUT_CPU_CHECK.json', check_dir / 'PRE_NATIVE_CHECK.json',
                 check_dir / 'check_controller_runtime.py', check_dir / 'CONTROLLER_RUNTIME_CPU_CHECK.json',
                 WORK / 'assets/vendor/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf',
                 WORK / 'assets/vendor/go2_flat_baryn20251217/policy.pt',
                 ROOT / 'scripts/run_gpu_job.py', TEACHER, PRECISION]:
        expected[str(path)] = sha(path)
    check(expected)
    save(HERE / 'PLAN.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        execution_worktree=str(WORK), source_sha256=expected,
        original_teacher_sha256=sha(TEACHER), explicit_scoring_revision='contact-flight-recovery-v1',
        original_physics_cases_target_and_thresholds_preserved=True,
        sequence=[dict(output=str(COLD_SOURCE), cap=102400, seconds=5400),
                  dict(output=str(TRAIN), cap=245760, seconds=10800),
                  dict(output=str(COLD_NEW), cap=102400, seconds=5400)],
        automatic_retry=False, formal_cases_read=False, formal_acceptance=False))
    print(json.dumps(dict(prepared=True, bound_inputs=len(expected))))


def job(output, entry, arguments, cap, seconds, label, expected):
    global child
    assert not stopping and not output.exists()
    check(expected)
    assert not any(row['status'] == 'RUNNING' for row in read(ROOT / 'status/GPU_LEDGER.json')['runs'])
    command = [PYTHON, str(ROOT / 'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
        'g07-full-racket-unlimited-20261024', '--max-seconds', str(seconds), '--env-steps-cap', str(cap),
        '--output', str(output), '--require-artifact', 'completed.json', '--', PYTHON,
        str(WORK / entry), '--output', str(output), *arguments, '--headless']
    save(HERE / (label + '_COMMAND.json'), dict(command=command, execution_worktree=str(WORK)))
    with (HERE / (label + '.guard.log')).open('x') as log:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2')
        env.pop('OPENBLAS_NUM_THREADS', None)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state('RUNNING_' + label, guard_pid=child.pid, output=str(output))
        while child.poll() is None:
            if stopping:
                child.send_signal(signal.SIGTERM)
            time.sleep(5)
    assert child.returncode == 0 and not stopping, label
    record = read(output / 'job.json')
    assert record['status'] == 'SUCCEEDED' and record['exit_code'] == 0
    assert not Path('/proc', str(record['child_pid'])).exists()
    assert record['charged_env_steps'] == read(output / 'completed.json')['env_steps'] == cap
    check(expected)


def assess(output, checkpoint, label):
    import numpy as np
    import torch
    from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL
    protocol = read(WORK / PROTOCOL)
    protocol['_root'] = str(WORK)
    cases, indices = training_cases(WORK, protocol)
    summary = summarize_training(read(output / 'episodes.json'), cases, indices, protocol)
    metrics = read(output / 'metrics.json')
    assert all(metrics[key] == value for key, value in summary.items())
    assert summary['numerical_failures'] == 0
    assert read(output / 'INPUTS.json')['policy_sha256'] == sha(checkpoint)
    save(HERE / (label + '_RESULT.json'), summary)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    controller = make_rally_controller(source, 2, 'cpu')
    inputs = read(output / 'native_command_inputs.json')
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(output / 'public_and_actions_first2.npz') as data:
        observations, actions, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    maximum = 0.
    previous = None
    with torch.inference_mode():
        for observation, action, keep in zip(observations, actions, valid):
            public = torch.from_numpy(observation)
            if previous is not None:
                ids = torch.where(public[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    controller.reset(ids)
            result = controller.action({'policy': public}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(result[keep] - action[keep]).max()))
            previous = public[:, 775].clone()
    assert maximum <= 5e-4
    save(HERE / (label + '_CPU_COMMANDS.json'), dict(actual_valid_public_rows=int(valid.sum()),
        CPU_vs_actual_GPU_max_action_error=maximum, original_tolerance=5e-4,
        policy_sha256=sha(checkpoint), new_native_execution=False))
    return summary


def main():
    sys.path.insert(0, str(WORK))
    plan = read(HERE / 'PLAN.json')
    expected = dict(plan['source_sha256'])
    expected[str(HERE / 'PLAN.json')] = sha(HERE / 'PLAN.json')
    entry = 'scripts/evaluate_full_racket_contact_recovery_origin0_train_v5.py'
    common = ['--precision-receipt', str(PRECISION)]
    job(COLD_SOURCE, entry, ['--checkpoint', str(TEACHER), '--noise', 'none', '--seed', '724201', *common],
        102400, 5400, 'TEACHER_REVISED_COLD128', expected)
    assess(COLD_SOURCE, TEACHER, 'TEACHER_REVISED_COLD128')
    import torch
    from quadsmash.rl.contact_recovery_source_v5 import verify_recovery_source
    from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL
    source = torch.load(TEACHER, map_location='cpu', weights_only=False)
    verification = verify_recovery_source(TEACHER, source, COLD_SOURCE, WORK, sha(WORK / PROTOCOL))
    save(HERE / 'ACTUAL_REVISED_SOURCE_VERIFICATION.json', verification)
    job(TRAIN, 'scripts/train_full_racket_contact_recovery_origin0_v5.py',
        ['--base-checkpoint', str(TEACHER), '--body-source-evaluation', str(COLD_SOURCE), *common,
         '--config', str(WORK / 'configs/g07_full_racket_contact_recovery_residual_v5.json'),
         '--num-envs', '128', '--updates', '15', '--seed', '751001'],
        245760, 10800, 'CONTACT_RECOVERY_TRAIN15', expected)
    checkpoint = torch.load(TRAIN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['iter'] == 15 and checkpoint['native']['canonical_restart'] is True
    assert not checkpoint['native']['environment']['tensors']['episode_length_buf'].any()
    assert read(TRAIN / 'canonical_contact_flight_state.json')['all_contact_flight_buffers_zero']
    assert checkpoint['infos']['residual_contract']['protocol_sha256'] == sha(WORK / PROTOCOL)
    save(HERE / 'ACTUAL_TRAINING_RECEIPT.json', dict(canonical_checkpoint_sha256=sha(TRAIN / 'checkpoint.pt'),
        policy_sha256=sha(TRAIN / 'policy.pt'), metrics=read(TRAIN / 'metrics.json'),
        canonical_ground_contact_flight_state_zero=True, formal_acceptance=False))
    job(COLD_NEW, entry, ['--checkpoint', str(TRAIN / 'policy.pt'), '--noise', 'none', '--seed', '724201', *common],
        102400, 5400, 'CONTACT_RECOVERY15_COLD128', expected)
    summary = assess(COLD_NEW, TRAIN / 'policy.pt', 'CONTACT_RECOVERY15_COLD128')
    state('COMPLETED_REVISED_SOURCE_TRAIN15_AND_COLD128', final_summary=summary)


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
