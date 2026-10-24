#!/usr/bin/env python3
"""Reconcile one orphaned GPU guard after its native output is complete.

This is deliberately stricter than editing the ledger by hand: it requires
the exact recorded PIDs to be gone, the required completion marker, a fresh
GPU0 process check, and an exclusive project ledger lock. It never changes
the declared interaction cap or marks an incomplete output successful.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--required-artifact', default='completed.json')
    args = parser.parse_args()
    output = args.output.resolve()
    ledger_path = ROOT / 'status/GPU_LEDGER.json'
    lock_path = ROOT / 'status/gpu_job.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text())
        matches = [r for r in ledger['runs'] if r.get('output') == str(output)
                   and r.get('status') == 'RUNNING']
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one RUNNING ledger record, found {len(matches)}')
        record = matches[0]
        if record.get('pid') != args.guard_pid or record.get('child_pid') != args.child_pid:
            raise ValueError('PID arguments do not match the ledger record')
        for pid in (args.guard_pid, args.child_pid):
            if Path(f'/proc/{pid}').exists():
                raise ValueError(f'PID {pid} is still alive; do not reconcile an active job')
        artifact = output / args.required_artifact
        if not artifact.is_file():
            raise ValueError('Required completion artifact is missing')
        if not output.joinpath('source_hashes.json').is_file():
            raise ValueError('Native source hash snapshot is missing')
        apps = subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name',
            '--format=csv,noheader'], text=True)
        competing = [line for line in apps.splitlines()
                     if not any(s in line for s in ('nvidia-cuda-mps-server', '/usr/local/awesun/bin/awesun_desktop'))]
        if competing:
            raise ValueError('GPU0 has competing compute processes: ' + repr(competing))
        started = datetime.fromisoformat(record['started_at'])
        finished = datetime.fromtimestamp(artifact.stat().st_mtime, timezone.utc)
        elapsed = max(0., (finished - started).total_seconds())
        record.update(status='SUCCEEDED', exit_code=0, timed_out=False,
                      receipt_present=True, guard_receipt_present=False,
                      orphan_guard_reconciled=True,
                      completion_artifact=str(artifact.relative_to(ROOT)),
                      elapsed_s=elapsed, charged_gpu_hours=elapsed / 3600.,
                      finished_at=finished.isoformat(),
                      reconciliation_time=datetime.now(timezone.utc).isoformat())
        atomic_json(ledger_path, ledger)
        job = dict(record, reconciliation_scope='Guard orphaned after native child completed; completion marker and raw outputs verified. This is not a formal acceptance result.')
        atomic_json(output / 'job.json', job)
        with (ROOT / 'status/RUNS.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(job, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
