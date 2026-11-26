"""Run frozen Recovery baselines during idle GPU intervals."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
APPROACH_ROOT = ROOT.with_name('supermimic_e1_v2_approach_scale')
GUI = {'Xorg', 'Xwayland', 'gnome-shell', 'chrome', 'firefox', 'antigravity', 'awesun_desktop'}


def dump(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def gpu_state():
    def query(spec):
        command = ['nvidia-smi', '-i', '0', spec, '--format=csv,noheader,nounits']
        return subprocess.check_output(command, text=True, timeout=8).strip()

    free = int(query('--query-gpu=memory.free'))
    pids = set()
    for line in query('--query-compute-apps=pid').splitlines():
        pid = int(line)
        try:
            executable = Path(os.readlink(f'/proc/{pid}/exe')).name
        except FileNotFoundError:
            continue
        except PermissionError:
            try:
                executable = Path(f'/proc/{pid}/comm').read_text().strip()
            except OSError:
                executable = ''
        if executable not in GUI:
            pids.add(pid)
    return free, pids


def approach_jobs():
    """Also yield during Approach launch or between a wrapper's native stages."""
    prefixes = ('train_', 'run_', 'eval_', 'evaluate_')
    jobs = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if Path(os.readlink(proc / 'cwd')) != APPROACH_ROOT:
                continue
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
            arguments = [arg.strip("'\"") for arg in command.decode(errors='replace').split()]
        except OSError:
            continue
        if any(Path(arg).name.startswith(prefixes) and arg.endswith('.py') for arg in arguments):
            jobs.add(int(proc.name))
    return jobs


def stop_native(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def verify(checkpoints):
    import torch
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
    from quadsmash.rl.supermimic_policy_execution_v1 import POSTHIT, resolve_contract

    reference = None
    contract = None
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        execution = resolve_contract(payload)[0]
        if payload['algorithm'] != 'supermimic_v1' or execution['recovery_gate'] != POSTHIT:
            raise ValueError(f'Expected causal SuperMimic checkpoint: {checkpoint}')
        state = payload['actor']
        protected = {k: v for k, v in state.items() if not k.startswith('actor.approach.')}
        if not set(RECOVERY_NAMES).issubset(protected):
            raise ValueError(f'Missing Recovery tensors: {checkpoint}')
        identity = (execution, payload['metadata']['task_contract'])
        if reference is None:
            reference = {k: v.clone() for k, v in protected.items()}
            contract = identity
        elif (reference.keys() != protected.keys() or identity != contract
              or any(not torch.equal(value, protected[k]) for k, value in reference.items())):
            raise ValueError(f'Checkpoints differ outside Approach or in the execution contract: {checkpoint}')
        del payload, state, protected


def summarize(folder):
    metrics = json.loads((folder / 'metrics.json').read_text())
    rows = json.loads((folder / 'episodes_summary.json').read_text())
    if len(rows) != 128 or metrics['numerical_failures'] or any(metrics['optimizer_updates'].values()):
        raise ValueError('Incomplete or numerically invalid fixed128 baseline')
    contact = [row for row in rows if row['qualified_airborne_string_contact']]
    faults = [row for row in contact if row['body_ground_seen']]
    ungrounded = [row for row in contact if not row['body_ground_seen'] and not row['strict_physical_recovery']]
    diagnostics = [row['recovery_baseline_diagnostic'] for row in ungrounded]
    observed = [row for row in diagnostics if row['qualified_contact_landing']]
    from quadsmash.tasks.stand.recovery_baseline_diagnostics import STRICT_FIELDS
    return dict(
        episodes=len(rows), qualified_contacts=len(contact), body_ground_all=sum(row['body_ground_seen'] for row in rows),
        body_ground_given_contact=len(faults), robot_net=sum(row['robot_net_fault'] for row in rows),
        strict_physical_recovery=sum(row['strict_physical_recovery'] for row in contact),
        relaxed_physical_recovery=sum(row['physical_recovery_success'] for row in contact),
        strict_scored_recovery=sum(row['strict_observed']['recovery_success'] is True for row in contact),
        relaxed_scored_recovery=sum(row['recovery_success'] for row in contact),
        relaxed_ball_side=sum(row['ball_side_complete_except_recovery'] for row in rows),
        strict_complete5=metrics['strict_observed']['complete5']['success'],
        relaxed_complete5=metrics['complete5'], strict_failed_without_body=len(ungrounded),
        strict_failed_without_qualified_landing=len(diagnostics) - len(observed),
        strict_threshold_failed_seconds={
            name: sum(row['strict_failed_seconds'][name] for row in observed) for name in STRICT_FIELDS},
        strict_threshold_interrupts={
            name: sum(row['strict_window_interrupts'][name] for row in observed) for name in STRICT_FIELDS},
        strict_arm_only_failed_s=sum(row['strict_arm_only_failed_s'] for row in observed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    parser.add_argument('--max-idle-wait-minutes', type=float, default=0.)
    args = parser.parse_args()
    checkpoints = [path.resolve(strict=True) for path in args.checkpoint]
    verify(checkpoints)
    output = args.output.resolve()
    output.mkdir(parents=True)
    status = output / 'status.json'
    results = []
    deadline = time.monotonic() + 60. * args.max_idle_wait_minutes
    for index, checkpoint in enumerate(checkpoints):
        if shutil.disk_usage(output).free < 15 * 2**30:
            dump(status, dict(state='disk_reserve', results=results))
            return
        with args.gpu_lock.open('a') as lock:
            while True:
                free, competitors = gpu_state()
                competitors |= approach_jobs()
                if not competitors and free >= 10000:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        break
                dump(status, dict(state='waiting_for_approach', gpu_free_mib=free,
                                  gpu_occupants=sorted(competitors), results=results))
                if time.monotonic() >= deadline:
                    return
                time.sleep(10)
            folder = output / f'{index:02d}_{checkpoint.stem}'
            command = [sys.executable, '-u', 'scripts/evaluate_supermimic_v1.py', '--checkpoint', str(checkpoint),
                       '--output', str(folder), '--research-fixed128', '--recovery-full-tail', '--headless',
                       '--physics-device', 'cuda:0', '--device', 'cuda:0',
                       '--kit_args=--renderer/multiGpu/enabled=false']
            with folder.with_suffix('.log').open('wb') as log:
                child = subprocess.Popen(command, cwd=ROOT,
                                         env=dict(os.environ, PYTHONPATH='.:scripts'),
                                         start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                try:
                    until = time.monotonic() + 45 * 60
                    dump(status, dict(state='running', checkpoint=str(checkpoint), pid=child.pid,
                                      gpu_free_mib=free, results=results))
                    while child.poll() is None:
                        free, competitors = gpu_state()
                        other = (competitors - {child.pid}) | approach_jobs()
                        if other or free < 2500 or time.monotonic() >= until:
                            dump(status, dict(state='stopped_for_gpu_or_wall_limit', gpu_free_mib=free,
                                              gpu_occupants=sorted(other), results=results))
                            return
                        time.sleep(3)
                finally:
                    stop_native(child)
            if child.returncode:
                dump(status, dict(state='native_failed', returncode=child.returncode, results=results))
                raise subprocess.CalledProcessError(child.returncode, command)
            result = dict(checkpoint=str(checkpoint), **summarize(folder))
            dump(folder / 'recovery_baseline_summary.json', result)
            results.append(result)
    dump(status, dict(state='complete', results=results))


if __name__ == '__main__':
    main()
