#!/usr/bin/env python3
"""One finite teacher-off evaluation after the active controller experiment.

No source mutation, repeat training, automatic promotion or overlapping GPU job.
The successful controller and original student remain separate comparisons.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.plan.parent
    # Exclusive creation also protects against two simultaneous queue launches.
    with (output / 'queue_owner.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'plan_sha256': sha(args.plan)}, stream)

    def save(name, value):
        path = output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    def status(state, **details):
        save('status.json', dict(state=state, pid=os.getpid(),
             time=datetime.now(timezone.utc).isoformat(), **details))

    def check_sources():
        for name, expected in plan['required_source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Pinned input changed: ' + name)

    def run(command, name):
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        with (output / name).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)

    try:
        check_sources()
        status('WAITING_FOR_ACTIVE_SET_CONTROLLER')
        deadline = time.monotonic() + plan['parent_wait_limit_s']
        while True:
            parent = json.loads((Path(plan['parent_worker']) / 'status.json').read_text())
            if parent['pid'] != plan['parent_pid']:
                raise ValueError('Parent worker identity changed')
            if parent['state'] == 'FAILED':
                raise ValueError('Parent failed; inspect before further GPU use')
            if parent['state'] == 'COMPLETED' and not Path('/proc', str(plan['parent_pid'])).exists():
                break
            if time.monotonic() > deadline:
                raise ValueError('Parent wait expired; no GPU job started')
            time.sleep(5)
        check_sources()
        candidate = Path(plan['candidate_output'])
        if candidate.exists():
            raise ValueError('Candidate output already exists')
        status('RUNNING_EVENT_REFINED_HEAD_TEACHER_OFF', output=str(candidate))
        run(plan['candidate_command'], 'guard.log')
        check_sources()
        run(['bash', 'scripts/python.sh', 'scripts/audit_native_learned_recovery_v1.py',
             '--run', str(candidate), '--checkpoint', plan['checkpoint'],
             '--output', str(output / 'command_audit.json')], 'command_audit.log')
        for label, before in plan['comparisons'].items():
            run(['bash', 'scripts/python.sh', 'scripts/compare_frozen_complete5_v2.py',
                 '--before', before, '--after', str(candidate),
                 '--output', str(output / (label + '.json'))], label + '.log')
        run(['bash', 'scripts/python.sh', 'scripts/summarize_native_recovery_learning_v1.py',
             '--run', str(candidate), '--command-audit', str(output / 'command_audit.json'),
             '--student-comparison', str(output / 'student27.json'),
             '--teacher-comparison', str(output / 'controller35.json'),
             '--output', str(output / 'execution_summary.json')], 'execution_summary.log')
        metrics = json.loads((candidate / 'metrics.json').read_text())
        audit = json.loads((output / 'command_audit.json').read_text())
        if (not audit['passed'] or metrics['numerical_failures'] or
                metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']):
            raise ValueError('Numerical/interface/teacher-off validation failed')
        save('result.json', dict(complete5=metrics['complete5'],
             air=metrics['airborne_contacts'], recovery=metrics['conditional_recovery_count'],
             numerical_failures=metrics['numerical_failures'], automatic_promotion=False))
        status('COMPLETED', complete5=metrics['complete5'])
    except BaseException as error:
        status('FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
