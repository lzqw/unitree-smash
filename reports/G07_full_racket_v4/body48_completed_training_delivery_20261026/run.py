"""Wait for completed native48 proofs, then seal and check its training archive.

CPU only; never controls GPU jobs or retries a failed native dependency. The
existing owner may run frozen assessments while immutable training is archived.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

import build_archive as builder

ROOT,HERE = builder.ROOT,builder.HERE
PLAN,STATE = HERE/'PLAN.json',HERE/'state.json'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PARENT_PID,NATIVE_PID = 1814619,1818808
stopping,child = False,None


def live(pid,marker):
    process = Path('/proc')/str(pid)
    try:
        command = (process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return marker in command and (process/'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def write(state,**extra):
    builder.atomic(STATE,dict(state=state,observed_at=datetime.now(timezone.utc).isoformat(),
        observer_pid=os.getpid(),CPU_only=True,new_native_execution=False,
        automatic_upload=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False,**extra))


def check(expected):
    for name,digest in expected.items():
        assert builder.sha(Path(name)) == digest,name


def prepare():
    assert not any(path.exists() for path in [PLAN,STATE,builder.ARCHIVE])
    assert live(PARENT_PID,'body_residual48_after_grid16_queue01/run.py')
    assert live(NATIVE_PID,'train_full_racket_body_residual_ppo_v4.py')
    admission = builder.read(builder.PARENT/'ADMISSION_DECISION.json')
    expected = {str(ROOT/name):digest for name,digest in admission['source_hashes'].items()}
    check(expected)
    source = admission['source']
    inputs = dict(source_policy_relative_path=str(Path(source['policy']).relative_to(ROOT)),
        source_evaluation_relative_path=str(Path(source['evaluation']).relative_to(ROOT)),
        source_policy_sha256=source['policy_sha256'])
    builder.atomic(HERE/'CHECK_INPUTS.json',inputs)
    # Inspect real input snapshots and bundled code now; completed48 is not
    # fabricated or claimed before the canonical job and audit exist.
    import zipfile
    hashes = builder.read(ROOT/builder.TRAIN/'source_hashes.json')
    with zipfile.ZipFile(ROOT/builder.TRAIN/'source_snapshot.zip') as snapshot:
        assert set(snapshot.namelist()) == set(hashes)
        import hashlib
        for name,digest in hashes.items():
            assert hashlib.sha256(snapshot.read(name)).hexdigest() == digest
    companion = builder.REPORTS/builder.COMPANION_NAME
    assert builder.sha(companion) == builder.COMPANION_SHA
    with zipfile.ZipFile(companion) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        names = [name for name in manifest['source_files'] if name.startswith('vendor/rsl_rl/')]
        assert len(names) == 30
        for name in names:
            assert hashlib.sha256(archive.read(name)).hexdigest() == manifest['source_files'][name]['sha256']
    suite = builder.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')['training_suite']
    assert builder.sha(ROOT/suite['path']) == suite['sha256']
    assert len(builder.read(ROOT/suite['path'])['cases']) == 128
    builder.atomic(HERE/'PREFLIGHT_CHECK.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_native_source_snapshot_members=len(hashes),all_snapshot_hashes_match=True,
        main_held_inputs_verified=len(expected),companion_SHA_verified=True,bundled_RSL_members_verified=len(names),
        original_training_cases_verified=128,completed48_not_available_at_preflight=True,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False))
    static = [HERE/name for name in ['check_data.py','build_archive.py','run.py','CHECK_INPUTS.json','PREFLIGHT_CHECK.json']]
    builder.atomic(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),
        static_sources={str(path):builder.sha(path) for path in static},main_inputs=expected,
        actual_parent_pid=PARENT_PID,actual_training_native_pid=NATIVE_PID,
        condition='Actual native child exits SUCCEEDED with canonical48, parent TRAINING_RECEIPT and TRAINING_TRACE_AUDIT; do not wait for later frozen assessments.',
        scope='All completed48 training data/weights plus measured source128 evidence. New frozen assessments are separate.',
        source_custody='Check main hashes at admission and sealing only. After sealing use archive/extraction bytes; no extended main hold.',
        archive=str(builder.ARCHIVE),GPU_jobs_started=0,automatic_upload=False,automatic_retry=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),main_inputs_verified=len(expected),new_GPU_job_started=False)),flush=True)


def cpu(arguments,log_name):
    global child
    with (HERE/log_name).open('x') as log:
        child = subprocess.Popen([PYTHON,*arguments],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        assert child.wait() == 0 and not stopping,'Preserve actual CPU failure: '+log_name


def main():
    assert PLAN.exists() and not STATE.exists()
    plan = builder.read(PLAN)
    static = {**plan['static_sources'],str(PLAN):builder.sha(PLAN)}
    check(static)
    write('WAITING_FOR_ACTUAL_COMPLETED_NATIVE48_PROOFS',parent_pid=PARENT_PID,native_pid=NATIVE_PID)
    while True:
        assert not stopping
        if not live(NATIVE_PID,'train_full_racket_body_residual_ppo_v4.py'):
            job_path = ROOT/builder.TRAIN/'job.json'
            job = builder.read(job_path) if job_path.exists() else None
            assert job is None or job['status'] in ['RUNNING','SUCCEEDED'],'Failed native dependency; do not retry'
            if job and job['status'] == 'SUCCEEDED' and all((builder.PARENT/name).exists()
                    for name in ['TRAINING_RECEIPT.json','TRAINING_TRACE_AUDIT.json']):
                break
            assert live(PARENT_PID,'body_residual48_after_grid16_queue01/run.py'),'Owner exited before completed training proofs'
        time.sleep(10)
    check(static)
    check(plan['main_inputs'])
    write('EXPORTING_ACTUAL_COMPLETED48_EPISODE_CSV_ON_CPU')
    cpu([str(ROOT/'scripts/export_full_racket_train_csv_v4.py'),'--run',str(ROOT/builder.TRAIN),
        '--output-prefix',str(HERE/'NATIVE48_EPISODES')],'training_csv.log')
    write('CHECKING_ACTUAL_COMPLETED48_DATA_ON_CPU')
    cpu([str(HERE/'check_data.py'),'--root',str(ROOT),
        '--output',str(HERE/'ACTUAL_COMPLETED_DATA_CHECK.json')],'actual_data_check.log')
    write('SEALING_COMPLETED48_TRAINING_ARCHIVE_ON_CPU')
    archive = builder.build()
    check(static)
    restored = Path(tempfile.mkdtemp(prefix='g07_body48_completed_training_restore_20261026_'))
    write('CHECKING_FRESH_EXTRACTED48_DATA_ON_CPU',archive=archive,extracted_root=str(restored))
    import zipfile
    with zipfile.ZipFile(builder.ARCHIVE) as bundle:
        assert all((restored/name).resolve().is_relative_to(restored) for name in bundle.namelist())
        bundle.extractall(restored)
    cpu([str(restored/builder.HERE_REL/'check_data.py'),'--root',str(restored),
        '--output',str(HERE/'FRESH_RESTORED_DATA_CHECK.json'),'--require-bundled'],'fresh_data_check.log')
    cpu([str(restored/'scripts/check_full_racket_training_traces_v4.py'),
        '--run',str(restored/builder.TRAIN),'--output',str(HERE/'FRESH_RESTORED_TRAINING_TRACES_CHECK.json')],
        'fresh_trace_check.log')
    assert builder.sha(builder.ARCHIVE) == archive['sha256']
    write('READY_FOR_DATA_UPLOAD',archive=archive,extracted_root=str(restored),
        fresh_extraction_CPU_checks_passed=True,original_full_goal_unproven=True,
        upload_authorized_by_existing_user_instructions=True)


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()
    raise RuntimeError('CPU archive worker cancelled; native GPU jobs untouched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    with (HERE/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
        else:
            signal.signal(signal.SIGTERM,stop)
            signal.signal(signal.SIGINT,stop)
            try:
                main()
            except BaseException as error:
                (HERE/'error.txt').write_text(traceback.format_exc())
                write('FAILED_ARCHIVE_ONLY_NATIVE_UNTOUCHED',error=repr(error))
                raise
