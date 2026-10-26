"""Wait for real completed07540/frozen data, archive and check a fresh extraction.

CPU only. No policy execution, native job control, automatic upload or changing
held main runtime. This workflow stops with a reviewable sealed data package.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_archive as builder
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PLAN, STATE = HERE/'PLAN.json', HERE/'state.json'
stopping = False
child = None


def atomic(path, value):builder.core.atomic(path, value)
def read(path):return builder.read(path)
def sha(path):return builder.sha(path)
def write(state, **extra):
    atomic(STATE, dict(state=state, observed_at=datetime.now(timezone.utc).isoformat(),
        observer_pid=os.getpid(), CPU_only=True, new_native_execution=False,
        automatic_upload=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False, **extra))


def prepare():
    assert not PLAN.exists() and not STATE.exists() and not builder.ARCHIVE.exists()
    assert builder.core.live(1734658, 'centered_placement075_resume32_queue02/run.py')
    active = [row for row in read(ROOT/'status/GPU_LEDGER.json')['runs'] if row['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) in [builder.TRAIN, builder.FROZEN]
    assert builder.core.live(active[0]['child_pid'], Path(active[0]['output']).name)
    train_complete = not builder.core.live(1734680, builder.TRAIN.name)
    if train_complete:builder.core.verify_job(builder.TRAIN, 8388608)
    held = {str(ROOT/name): digest for name, digest in read(builder.SELECTOR/'PREDECLARED_PLAN.json')['source_hashes'].items()}
    held[str(builder.SELECTOR/'PREDECLARED_PLAN.json')] = sha(builder.SELECTOR/'PREDECLARED_PLAN.json')
    assert all(sha(Path(name)) == digest for name, digest in held.items())
    archives = []
    for folder in ['placement075_delivery_20261026', 'placement075_intermediate_CPU_delivery_20261026']:
        record = read(builder.REPORTS/folder/'ARCHIVE_CHECK.json')
        path = Path(record['archive'])
        path = path if path.is_absolute() else ROOT/path
        assert sha(path) == record['sha256']
        archives.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    old = read(HERE/'ACTUAL_OLD025_COMPLETED_LAYOUT_CHECK.json')
    assert old['actual_training_run'] == 'residual_ppo_centered_base8_n2048_resume32_01'
    assert old['canonical_vs_public_actor_max_error'] == 0. and not old['new_native_execution']
    static = [Path(__file__), HERE/'build_archive.py', HERE/'check_data.py', HERE/'ACTUAL_OLD025_COMPLETED_LAYOUT_CHECK.json']
    atomic(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), static_sources={str(p):sha(p) for p in static},
        main_inputs=held, stage8_archive=archives[0], CPU16_24_archive=archives[1],
        actual_parent_pid=1734658, historical_train_native_pid=1734680,
        actual_current_native_pid=active[0]['child_pid'], training_already_complete_at_registration=train_complete,
        launch_condition='Actual parent exits READY, native32 additional updates/canonical40 and original TRAIN128 successful, selected training traces audited and original285 sources match.',
        scope='Completed native training plus all frozen128 data/assets. Current unreviewed40 video excluded; only previously reviewed stage8 video inherited.',
        source_custody='Main inputs checked at admission and sealing only; after sealing all checks use archive/extracted bytes. Does not prolong main runtime ownership during later artifact review/delivery.',
        actual_new07540_results_not_available=not (builder.FROZEN/'metrics.json').exists(),
        old025_completed_witness_is_layout_check_only=True, automatic_upload=False,
        new_native_execution=False, formal_acceptance=False, goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN), main_inputs_verified=len(held), new_GPU_job_started=False)))


def cpu(arguments, log_name):
    global child
    with (HERE/log_name).open('x') as log:
        child = subprocess.Popen([PYTHON, *arguments], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT)
        assert child.wait() == 0 and not stopping


def main():
    assert PLAN.exists() and not STATE.exists()
    plan = read(PLAN)
    expected = {**plan['static_sources'], str(PLAN):sha(PLAN)}
    assert all(sha(Path(n)) == h for n,h in expected.items())
    write('WAITING_FOR_ACTUAL_COMPLETED07540_DATA', parent_pid=1734658)
    while builder.core.live(1734658, 'centered_placement075_resume32_queue02/run.py'):
        assert not stopping
        time.sleep(10)
    assert not stopping and all(sha(Path(n)) == h for n,h in expected.items())
    write('VERIFYING_COMPLETED_NATIVE_AND_FROZEN128_DATA')
    builder.verify_completed()
    write('BUILDING_COMPLETE_DATA_ARCHIVE_ON_CPU')
    archive = builder.build()
    assert not stopping and all(sha(Path(n)) == h for n,h in expected.items())
    restored = Path(tempfile.mkdtemp(prefix='g07_placement075_completed40_restore_20261026_'))
    write('CHECKING_FRESH_EXTRACTED_DATA_ON_CPU', archive=archive, extracted_root=str(restored))
    with zipfile.ZipFile(builder.ARCHIVE) as bundle:
        assert all((restored/name).resolve().is_relative_to(restored) for name in bundle.namelist())
        bundle.extractall(restored)
    checker = restored/'reports/G07_full_racket_v4/placement075_completed40_data_queue01/check_data.py'
    cpu([str(checker), '--root', str(restored), '--output', str(HERE/'FRESH_RESTORED_DATA_CHECK.json'), '--require-bundled'], 'fresh_data_check.log')
    cpu([str(restored/'scripts/check_full_racket_training_traces_v4.py'), '--run', str(restored/builder.TRAIN.relative_to(ROOT)),
        '--output', str(HERE/'FRESH_RESTORED_TRAINING_TRACES_CHECK.json')], 'fresh_trace_check.log')
    assert sha(builder.ARCHIVE) == archive['sha256']
    write('READY_FOR_AUTHORIZED_DATA_DELIVERY', archive=archive, extracted_root=str(restored),
        fresh_extraction_CPU_checks_passed=True, current40_video_not_uploaded=True,
        full_original_goal_unproven=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:child.terminate()
    raise RuntimeError('CPU archive observer cancelled; native jobs untouched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        live = builder.core.live(1734658, 'centered_placement075_resume32_queue02/run.py')
        print(json.dumps(dict(actual_parent_live=live, readiness='WAIT' if live else 'CHECK_ACTUAL_TERMINAL_RECEIPTS', archive_created=builder.ARCHIVE.exists(), new_native_execution=False)))
    else:
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.prepare_only:prepare()
            else:
                signal.signal(signal.SIGTERM, stop)
                signal.signal(signal.SIGINT, stop)
                try:main()
                except BaseException as error:
                    (HERE/'error.txt').write_text(traceback.format_exc())
                    write('FAILED_DATA_ARCHIVE_ONLY_NATIVE_UNTOUCHED', error=repr(error))
                    raise
