"""Confirm finalized native TRAIN contacts after the actual training guard exits.

CPU evidence work only. Never launch or stop a GPU job or reconstruct motion.
"""
from datetime import datetime, timezone
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
NATIVE = REPORTS / 'residual_resize2048_queue01'
BUILDER = REPORTS / 'residual_iter96_delivery_20261025/build.py'
MILESTONE = REPORTS / 'residual_n2048_first_complete_20261025'
CONFIRMATION = MILESTONE / 'FINAL_CONTACT_CONFIRMATION.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
GUARD_PID, TRAINER_PID, PARENT_PID = 1582969, 1582980, 1576853
STATE = HERE / 'state.json'
spec = importlib.util.spec_from_file_location('native_contact_queue_checks', NATIVE / 'run.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
core = native.core
child = None
stopping = False


def hashes():
    result = native.hashes()
    for path in [Path(__file__).resolve(), BUILDER, BUILDER.with_name('contact_checks.py'),
                 MILESTONE / 'REPORT.json', MILESTONE / 'SUPPLEMENT_UPDATE48.json',
                 MILESTONE / 'complete_episodes_update48.json']:
        result[str(path)] = core.sha(path)
    return result


def write(state, **extra):
    value = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        original_training_guard_pid=GUARD_PID, original_trainer_pid=TRAINER_PID,
        confirmation=str(CONFIRMATION), renderer_uses_gpu=False, GPU_jobs_started=False,
        formal_acceptance=False, goal_complete=False, **extra)
    core.atomic(STATE, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'source_hashes'}), flush=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    global child
    assert not STATE.exists() and not CONFIRMATION.exists()
    assert core.live(GUARD_PID, str(native.TRAIN))
    assert core.live(TRAINER_PID, str(native.TRAIN))
    assert core.live(PARENT_PID, 'residual_resize2048_queue01/run.py')
    expected = hashes()
    command = [PYTHON, str(BUILDER), '--confirm-training-contacts-only']
    core.atomic(HERE / 'PREDECLARED_PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        actual_live_guard_pid=GUARD_PID, actual_live_trainer_pid=TRAINER_PID,
        actual_live_parent_pid=PARENT_PID, source_hashes=expected, CPU_command=command,
        qualification='Wait for the actual guard exit; require its SUCCEEDED receipt, exact16777216 steps, exited trainer and unchanged sources. The existing builder then independently verifies canonical96/export and matches every full native TRAIN episode to its finalized actual contact.',
        CPU_environment=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'),
        CPU_timeout_s=1800, automatic_retries=False,
        preserved_creation_reports='Original pending flags remain unchanged; only a new FINAL_CONTACT_CONFIRMATION.json may confirm flushed records.',
        timing='May run alongside the original fixed128 GPU assessment after training finishes; no new GPU resources or scheduling.',
        full_motion_scope='No full50Hz training trajectory was saved; this queue cannot create a training-success motion video.',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False))
    write('WAITING_FOR_ACTUAL_TRAINING_GUARD_EXIT', source_hashes=expected,
          note='Original GPU owner, its later fixed assessment and both CPU video queues remain untouched.')
    while not stopping and core.live(GUARD_PID, str(native.TRAIN)):
        time.sleep(10)
    if stopping:
        raise InterruptedError('Cancelled while waiting; original GPU workflow untouched')
    assert hashes() == expected, 'Source changed; inspect before confirming any record'
    job = core.verify_job(native.TRAIN, native.STEPS)
    assert job['pid'] == GUARD_PID and job['child_pid'] == TRAINER_PID
    assert not core.live(TRAINER_PID, str(native.TRAIN))
    with (HERE / 'confirmation.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)
        write('MATCHING_FINALIZED_NATIVE_CONTACTS_CPU', source_hashes=expected,
              owned_CPU_child_pid=child.pid, command=command)
        code = child.wait(timeout=1800)
    if stopping:
        raise InterruptedError('Cancelled the owned CPU step; finalized files remain for inspection')
    assert code == 0, 'Owned CPU confirmation failed; no automatic retry'
    assert hashes() == expected
    result = core.read(CONFIRMATION)
    counts = core.read(native.TRAIN / 'metrics.json')['counts']
    assert result['exact_contact_record_matches']
    assert result['matched_complete_episodes'] == counts['complete'] >= 3
    assert result['independently_recomputed_episode_counts'] == counts
    assert result['original_contacts_sha256'] == core.sha(native.TRAIN / 'contacts.json')
    assert result['original_episodes_sha256'] == core.sha(native.TRAIN / 'episodes.json')
    assert result['original_snapshot_pending_flags_retained_as_history']
    assert not result['full_training_motion_trajectory_available'] and not result['formal_acceptance']
    write('SUCCEEDED', finalized_confirmation_sha256=core.sha(CONFIRMATION),
        independently_recomputed_training_counts=counts, matched_complete_episodes=result['matched_complete_episodes'],
        note='Final native contact identities and rigid point-velocity records confirmed. Frozen evaluation, videos, original A/B and full delivery remain separately required.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except InterruptedError as error:
        write('CANCELLED', error=str(error))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        write('FAILED_VALIDATION_OR_CONTACT_CONFIRMATION', error=repr(error))
        raise
