#!/usr/bin/env python3
"""Wait for one identified training job, then run guarded native comparisons.

This process consumes no GPU while waiting. Every native child still goes
through the existing single-GPU ledger and lock. It never stops its predecessor.
"""
import argparse
import datetime
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_matches(record):
    path = Path('/proc')/str(record['pid'])
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == record['start_ticks']
    except FileNotFoundError:
        return False


def verify_inputs(plan):
    for path, expected in plan['held_inputs'].items():
        if sha(path) != expected:
            raise ValueError('Planned native-trial input changed: '+path)


def verify_completed(directory):
    directory = Path(directory)
    job = json.loads((directory/'job.json').read_text())
    if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or (directory/'error.txt').exists():
        raise ValueError('Native predecessor/trial did not finish successfully: '+str(directory))
    return json.loads((directory/'completed.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.plan.parent
    if (output/'state.json').exists():
        raise ValueError('Use a fresh sequence output; preserve prior lifecycle records')
    lock = (output/'sequence.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    stopping = False
    state = dict(pid=os.getpid(), state='WAITING_FOR_IDENTIFIED_TRAINING',
                 plan_sha256=sha(args.plan), completed_trials=[], goal_complete=False)

    def save():
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(output/'state.json', state)

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        verify_inputs(plan)
        if not all(process_matches(r) for r in plan['predecessor_processes']):
            raise ValueError('Expected training processes were not live at sequence registration')
        while any(process_matches(r) for r in plan['predecessor_processes']):
            if stopping:
                state['state'] = 'STOPPED_WHILE_WAITING';save();return 2
            state['live_predecessor_pids'] = [r['pid'] for r in plan['predecessor_processes'] if process_matches(r)]
            save()
            time.sleep(5)
        predecessor = Path(plan['predecessor_output'])
        completed = verify_completed(predecessor)
        metrics = json.loads((predecessor/'metrics.json').read_text())
        if (completed['env_steps'] != plan['expected_training_env_steps']
                or metrics['completed_updates'] != plan['expected_new_updates']
                or metrics['history'][-1]['update'] != plan['expected_total_updates']
                or sha(predecessor/'policy.pt') != metrics['policy_sha256']):
            raise ValueError('Training completion or policy export differs from the planned continuation')
        state['training_policy_sha256'] = metrics['policy_sha256'];save()
        from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
        protocol = json.loads(Path(plan['protocol']).read_text());protocol['_root'] = str(ROOT)
        cases, indices = training_cases(ROOT, protocol)
        for trial in plan['trials']:
            if stopping:
                state['state'] = 'STOPPED_BETWEEN_TRIALS';save();return 2
            verify_inputs(plan)
            target = Path(trial['output'])
            if target.exists():
                raise ValueError('Never overwrite an existing native trial: '+str(target))
            state.update(state='RUNNING_NATIVE_TRIAL', current_trial=trial['name'], current_output=str(target))
            with (output/(trial['name']+'.guard.log')).open('x') as log:
                child = subprocess.Popen(trial['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_guard_pid'] = child.pid;save()
                code = child.wait()
            child = None
            state.pop('child_guard_pid', None)
            if code != 0:
                raise ValueError('Guarded native trial exited '+str(code)+': '+trial['name'])
            verify_completed(target)
            rows = json.loads((target/'episodes.json').read_text())
            checked = summarize_training(rows, cases, indices, protocol)
            reported = json.loads((target/'metrics.json').read_text())
            if any(reported[key] != value for key, value in checked.items()):
                raise ValueError('Native and independently recomputed trial metrics disagree')
            if checked['numerical_failures']:
                raise ValueError('Numerical failure requires diagnosis before further trials')
            atomic(target/'sequence_independent_summary.json', checked)
            state['completed_trials'].append(dict(name=trial['name'], output=str(target),
                complete=checked['complete'], air=checked['airborne_contacts'],
                recovery=checked['conditional_recovery_count'], numerical_failures=checked['numerical_failures']))
            save()
        verify_inputs(plan)
        comparisons=plan.get('comparison_commands')
        if comparisons is None:comparisons=[plan['comparison_command']]
        for command in comparisons:
            subprocess.run(command, cwd=ROOT, check=True)
        state['state'] = 'TRIALS_AND_COMPARISON_COMPLETE';save()
        return 0
    except BaseException:
        state['state'] = 'SEQUENCE_FAILED';state['error'] = traceback.format_exc();save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
