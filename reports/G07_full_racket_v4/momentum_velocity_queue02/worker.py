"""Run one momentum-informed TRAIN search after the current GPU guard exits."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/autodrive/LZQW/unitree_smash')
HERE = Path(__file__).resolve().parent
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
PRIOR = ROOT / 'reports/G07_full_racket_v4/ball_pitch_blend_cem03_resume01'
OUTPUT = ROOT / 'reports/G07_full_racket_v4/ball_momentum_velocity_cem04'
COMMAND = [PYTHON, 'scripts/run_gpu_job.py', '--goal', 'G07', '--budget-window',
           'g07-full-racket-unlimited-20261024', '--max-seconds', '5400', '--env-steps-cap',
           '204800', '--output', str(OUTPUT.relative_to(ROOT)), '--require-artifact', 'completed.json',
           '--', PYTHON, 'scripts/search_full_racket_ball_v4.py', '--output', str(OUTPUT.relative_to(ROOT)),
           '--warm-start', 'reports/G07_full_racket_v4/momentum_velocity_warm10/policy.pt',
           '--candidates', '64', '--generations', '2', '--sigma-fraction', '.05',
           '--training-indices', '1', '65', '--continue-after-miss', '--contact-curriculum',
           '--contact-grid', '--recovery-leg-bank',
           'reports/G07_full_racket_v4/momentum_velocity_warm10/RECOVERY_LEG_BANK_FROM_CEM03_GEN0.json',
           '--push-search-range', '.30', '--headless']
state = dict(worker_pid=os.getpid(),prior_guard_pid=1360158,prior_child_pid=1360170,
             prior_output=str(PRIOR.relative_to(ROOT)),output=str(OUTPUT.relative_to(ROOT)),
             command=COMMAND,formal_acceptance=False)


def save(status, **extra):
    state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
    temporary = HERE / 'state.json.tmp'
    temporary.write_text(json.dumps(state,indent=2) + '\n')
    temporary.replace(HERE / 'state.json')
    print(json.dumps(dict(status=status,worker_pid=os.getpid(),**extra)),flush=True)


def live(pid, required):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return False
    if path.joinpath('stat').read_text().split()[2] == 'Z':
        return False
    args = path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    if path.stat().st_uid != os.getuid() or required not in args or str(PRIOR.relative_to(ROOT)) not in args:
        raise RuntimeError('Predecessor PID identity changed; do not launch a replacement')
    return True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    if OUTPUT.exists():
        raise ValueError('Refuse to overwrite native evidence')
    sources = list(ROOT.glob('quadsmash/**/*.py')) + [ROOT / name for name in [
        'scripts/search_full_racket_ball_v4.py', 'scripts/run_gpu_job.py',
        'configs/g07_full_racket_eval_v4_r3.json', 'configs/full_racket_v4_r3.json',
        'configs/court_net_v4.json', 'configs/full_racket_usd_dependencies_v4.json',
        'reports/G07_full_racket_v4/momentum_velocity_warm10/policy.pt',
        'reports/G07_full_racket_v4/momentum_velocity_warm10/RECOVERY_LEG_BANK_FROM_CEM03_GEN0.json']]
    inertials = json.loads((ROOT / 'reports/G07_full_racket_v4/CENTROIDAL_USD_INERTIALS_0715.json').read_text())
    sources += [Path(p) for p in inertials['source_layers']]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):digest(p) for p in sources}
    (HERE / 'launch_source_hashes.json').write_text(json.dumps(hashes,indent=2) + '\n')
    started = time.monotonic()
    save('WAITING_FOR_VERIFIED_PREDECESSOR')
    while True:
        guard = live(1360158,'scripts/run_gpu_job.py')
        child = live(1360170,'scripts/search_full_racket_ball_v4.py')
        if not guard and not child:
            break
        if time.monotonic() - started > 5400:
            raise TimeoutError('Observation bound reached; no replacement launched')
        time.sleep(15)
    receipt = json.loads((PRIOR / 'job.json').read_text())
    if receipt['status'] != 'SUCCEEDED' or not receipt['receipt_present'] or not (PRIOR / 'completed.json').exists():
        raise RuntimeError('Predecessor did not complete; inspect before dependent training')
    rows = [row for p in sorted(PRIOR.glob('episodes_[0-9][0-9][0-9].json')) for row in json.loads(p.read_text())]
    if not rows or any(row['numerical_failures'] for row in rows):
        raise RuntimeError('Missing or numerically faulty native results require inspection')
    attacks = sum(row['attack_outgoing'] for row in rows)
    if attacks:
        save('SKIPPED_NATIVE_ATTACK_PROGRESS_REQUIRES_REVIEW',native_attacks=attacks)
        sys.exit(0)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Queued control, weights or geometry changed: ' + name)
    if OUTPUT.exists():
        raise ValueError('Another process already created the output; do not duplicate')
    with (HERE / 'guard_stdout.log').open('x') as log:
        process = subprocess.Popen(COMMAND,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save('RUNNING_GUARDED_JOB',guard_pid=process.pid,predecessor_native_attacks=0)
        code = process.wait()
    result = json.loads((OUTPUT / 'job.json').read_text())
    success = code == 0 and result['status'] == 'SUCCEEDED' and result['receipt_present']
    save('SUCCEEDED' if success else 'FAILED',exit_code=code)
    sys.exit(0 if success else 1)
except Exception:
    (HERE / 'error.txt').write_text(traceback.format_exc())
    save('FAILED_BEFORE_OR_DURING_CONTINUATION')
    raise
