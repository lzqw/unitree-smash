"""Finish only budget-incomplete paired model diagnostics, never physical failures.

Each worker receives its own compute allowance when it actually starts. A
separate overall deadline remains. Replay starts at the same recorded actual
release state, not an incomplete model state. This does not launch native
physics, train a policy or label a model path as an expert demonstration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_state_settling_feedback_v2 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run_one(cfg, nominal, path, mode, seconds, overall_deadline):
    started = time.monotonic()
    result = evaluate(cfg, nominal, path, mode, min(started + seconds, overall_deadline))
    result['worker_elapsed_s'] = time.monotonic() - started
    result['worker_compute_allowance_s'] = min(seconds, max(0., overall_deadline - started))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['previous', 'configuration', 'run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--worker-seconds', type=int, default=2400)
    p.add_argument('--total-seconds', type=int, default=5400)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if not 0 < a.workers <= 4 or not 0 < a.worker_seconds <= a.total_seconds <= 7200:
        raise ValueError('Explicit positive bounded CPU allowances required')
    read = lambda path: json.loads(path.read_text())
    previous = read(a.previous / 'result.json')
    if previous['state'] != 'COMPLETED_FINE_SELECTOR_SETTLING_FEEDBACK':
        raise ValueError('Wait for a terminal preceding diagnostic')
    if read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Require completed original native state evidence')
    inputs = read(a.previous / 'inputs.json')
    pins = dict(inputs['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed preceding diagnostic source ' + path)
    if pins.get(str(a.configuration.resolve())) != sha(a.configuration):
        raise ValueError('Keep the same controller and hardware configuration')
    cfg = read(a.configuration)
    nominal = read(a.run / 'native_command_inputs.json')['default_joint_position_rad']
    jobs, retained, all_keys = [], [], set()
    for row in previous['results']:
        key = (row['state_key'], row['mode'])
        if key in all_keys or row['mode'] not in ['plan_end', 'state_settling']:
            raise ValueError('Invalid paired result membership')
        all_keys.add(key)
        failure = row.get('failure') or {}
        if failure.get('reason') != 'compute_budget':
            retained.append(row)
            continue
        snapshot = a.previous / 'candidate_snapshots' / ('state_' + row['state_key'].replace(':', '_') + '.json')
        if pins.get(str(snapshot.resolve())) != sha(snapshot):
            raise ValueError('Candidate must be the exact preceding frozen snapshot')
        data = read(snapshot)
        if data['native_success'] or data['valid_expert_labels'] or data['fine']['substeps'] != 160:
            raise ValueError('Expected unverified fine-model reference')
        jobs.append((snapshot, row['mode']))
    for f in [Path(__file__), a.previous / 'result.json', a.previous / 'inputs.json', a.run / 'job.json']:
        pins[str(f.resolve())] = sha(f)
    summary = dict(retry_count=len(jobs), retained_count=len(retained),
        retry_keys=[(read(path)['state_key'], mode) for path, mode in jobs],
        worker_seconds=a.worker_seconds, total_seconds=a.total_seconds,
        control_or_physical_changes=False, native_execution=False, training=False)
    if a.check_only:
        print(json.dumps(summary))
        return
    if a.output.exists() or not jobs:
        raise ValueError('Require fresh output and budget-incomplete work')
    a.output.mkdir(parents=True)
    write(a.output / 'inputs.json', dict(summary, source_sha256=pins,
        replay_from_original_actual_state=True, workers=a.workers))
    started = time.monotonic()
    results = list(retained)
    write(a.output / 'status.json', dict(state='RUNNING_BUDGET_ONLY_REPLAY', pid=os.getpid(),
        completed_retries=0, total_retries=len(jobs), retained_count=len(retained)))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_one, cfg, nominal, str(path), mode,
            a.worker_seconds, started + a.total_seconds) for path, mode in jobs]
        for future in as_completed(futures):
            result = future.result()
            write(a.output / (result['state_key'].replace(':', '_') + '_' + result['mode'] + '.json'), result)
            results.append({k: v for k, v in result.items() if k not in ['trajectory', 'initial_actual_state']})
            write(a.output / 'status.json', dict(state='RUNNING_BUDGET_ONLY_REPLAY', pid=os.getpid(),
                completed_retries=len(results) - len(retained), total_retries=len(jobs), results=results))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed source during diagnostic ' + path)
    if {(r['state_key'], r['mode']) for r in results} != all_keys:
        raise ValueError('Paired diagnostic membership changed')
    result = dict(state='COMPLETED_BUDGET_ONLY_REPLAY', results=results,
        wall_seconds=time.monotonic() - started, native_success=False,
        valid_expert_labels=False, training=False)
    write(a.output / 'result.json', result)
    write(a.output / 'status.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
