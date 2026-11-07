#!/usr/bin/env python3
"""One audited correction export, finite fit and teacher-off native evaluation.

Waits for the actual collection worker to finish and exit. Uses the existing
GPU0 guard; does not change live sources or automatically promote a checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    try: return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: return None


def validate_teacher_off_command(command, checkpoint, output):
    if '--' not in command: raise ValueError('GPU guard required')
    guard, native = command[:command.index('--')], command[command.index('--')+1:]
    if ('scripts/run_gpu_job.py' not in guard or '--max-seconds' not in guard or
            '--env-steps-cap' not in guard): raise ValueError('Finite guarded execution required')
    if '--learned-recovery-mpc-correction' in native or any(x.startswith('--landing-mpc-') for x in native):
        raise ValueError('Teacher-off evaluation must not enable MPC')
    if (native[native.index('--learned-recovery-command-head')+1] != str(checkpoint) or
            native[native.index('--output')+1] != str(output) or
            native[native.index('--device')+1] != 'cuda:0'):
        raise ValueError('Wrong candidate, output or GPU')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args(); plan = json.loads(a.plan.read_text()); out = a.plan.parent
    with (out/'queue_owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), plan_sha256=sha(a.plan)), f)
    def write(name, value):
        path = out/name; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)
    def status(state, **kw):
        write('status.json', dict(state=state, pid=os.getpid(), time=datetime.now(timezone.utc).isoformat(), **kw))
    def check_sources():
        for name, expected in plan['source_sha256'].items():
            if sha(ROOT/name) != expected: raise ValueError('Pinned source changed: '+name)
    def run(command, log):
        env = os.environ.copy(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'
        with (out/log).open('x') as f:
            subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        validate_teacher_off_command(plan['evaluation_command'], plan['checkpoint'], plan['evaluation_output'])
        check_sources(); status('WAITING_FOR_VERIFIED_PAIRED_COLLECTION')
        deadline = time.monotonic()+plan['wait_limit_s']
        parent_dir = Path(plan['parent_worker'])
        while True:
            parent = json.loads((parent_dir/'status.json').read_text())
            identity = process_identity(plan['parent_pid'])
            if parent['pid'] != plan['parent_pid'] or (identity is not None and identity != plan['parent_start_ticks']):
                raise ValueError('Collection worker identity changed')
            if parent['state'] == 'FAILED': raise ValueError('Collection failed; inspect before further training')
            if identity is None:
                if parent['state'] != 'COMPLETED': raise ValueError('Collection disappeared without completion')
                break
            if time.monotonic()>deadline: raise ValueError('Wait expired; no training or GPU evaluation started')
            time.sleep(5)
        check_sources()
        source = Path(plan['collection_output'])
        job = json.loads((source/'job.json').read_text())
        audit = json.loads((parent_dir/'paired_command_audit.json').read_text())
        if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or not (source/'completed.json').exists() or
                not audit['passed'] or not audit['extra_teacher_enabled'] or Path(audit['run']).resolve()!=source.resolve()):
            raise ValueError('Completed native paired execution and matching audit required')
        status('EXPORTING_VERIFIED_SUCCESSFUL_CORRECTIONS')
        run(plan['export_command'], 'export.log')
        validation = json.loads((Path(plan['dataset'])/'validation.json').read_text())
        if not validation['passed'] or not validation['trainable']:
            raise ValueError('No verified trainable corrections; do not fit failures')
        status('FITTING_CORRECTIONS_WITH_SUCCESS_RETENTION')
        run(plan['training_command'], 'training.log'); check_sources()
        checkpoint_sha = sha(plan['checkpoint'])
        write('training_receipt.json', dict(checkpoint_sha256=checkpoint_sha,
            dataset_manifest_sha256=sha(Path(plan['dataset'])/'manifest.json'),
            fit_report_sha256=sha(Path(plan['checkpoint']).parent/'fit_report.json'), native_evaluated=False))
        candidate = Path(plan['evaluation_output'])
        if candidate.exists(): raise ValueError('Preserve previous evaluation')
        status('RUNNING_LEARNED_HEAD_WITHOUT_EXTRA_TEACHER')
        run(plan['evaluation_command'], 'guard.log'); check_sources()
        if sha(plan['checkpoint']) != checkpoint_sha: raise ValueError('Candidate changed during evaluation')
        status('AUDITING_TEACHER_OFF_EXECUTION')
        for i, command in enumerate(plan['post_commands']): run(command, f'audit{i}.log')
        audit = json.loads((out/'command_audit.json').read_text())
        metrics = json.loads((candidate/'metrics.json').read_text())
        if (not audit['passed'] or metrics['numerical_failures'] or
                metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']):
            raise ValueError('Teacher-off execution validation failed')
        write('result.json', dict(complete5=metrics['complete5'], air=metrics['airborne_contacts'],
            recovery=metrics['conditional_recovery_count'], numerical_failures=metrics['numerical_failures'],
            automatic_promotion=False, formal_acceptance=False))
        status('COMPLETED', complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED', error=repr(error)); raise


if __name__ == '__main__': main()
