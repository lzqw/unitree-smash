"""Export original per-episode CSVs after the live512-env learning/assessment workflow."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
PARENT = REPORTS / 'residual_resize512_queue01'
PARENT_PID = 1566225
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
STATE = HERE / 'state.json'
stopping = False
child = None


def read(path):
    return json.loads(path.read_text())


def write(state, **extra):
    data = dict(state=state, queue_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        formal_acceptance=False, gpu_used=False, **extra)
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(data), flush=True)


def live():
    proc = Path(f'/proc/{PARENT_PID}')
    try:
        status = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    if status in ('Z', 'X'):
        return False
    if 'residual_resize512_queue01/run.py' not in command:
        raise RuntimeError('Parent PID identity changed; refusing PID reuse')
    return True


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    global child
    assert not STATE.exists() and live()
    paths = [ROOT / 'scripts/export_full_racket_train_csv_v4.py', Path(__file__).resolve(),
        ROOT / 'quadsmash/evaluation_v4.py', ROOT / 'quadsmash/control/full_racket_train_evaluation_v4.py',
        ROOT / 'quadsmash/physics/rally_rules_v4.py', ROOT / 'configs/g07_full_racket_eval_v4_r3.json']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write('WAITING_FOR_NATIVE_WORKFLOW', verified_parent_pid=PARENT_PID, source_hashes=hashes)
    while not stopping and live():
        time.sleep(10)
    if stopping:
        write('CANCELLED', note='No GPU job interrupted')
        return
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    parent = read(PARENT / 'state.json')
    if parent['state'] != 'SUCCEEDED':
        write('SKIPPED_PARENT_NOT_SUCCEEDED', parent_state=parent['state'])
        return
    exports = []
    for source, stem in [('residual_ppo_n512_resume24_01', 'iter32_training_episodes'),
                         ('residual_ppo_fixed_train128_iter32_n512_01', 'iter32_fixed_episodes')]:
        prefix = HERE / stem
        command = [PYTHON, 'scripts/export_full_racket_train_csv_v4.py', '--run', str(REPORTS / source),
            '--output-prefix', str(prefix)]
        write('EXPORTING_CPU_CSV', source=source, command=command)
        with (HERE / (stem + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=log, stderr=subprocess.STDOUT)
            code = child.wait(timeout=300)
        assert code == 0 and not stopping, 'CPU export stopped or failed; inspect its log'
        manifest = read(prefix.with_suffix('.json'))
        assert manifest['original_runtime_metrics_match'] and not manifest['formal_acceptance']
        exports.append(manifest)
    write('SUCCEEDED', exports=exports, task_status='CSV export only; no new native performance or G07 acceptance')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        write('FAILED_EXPORT_OR_VALIDATION', error=repr(error))
        raise
