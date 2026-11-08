"""Isolate state-driven settling from the cost of fine candidate ranking.

Reuse two immutable complete-model references from the existing paired test.
Only evaluate its missing coarse-selector/state-settling cells. Physical
rollout remains 160 substeps, original limits and full original episode.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_state_settling_feedback_v1 import evaluate
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def run_one(cfg, nominal, path, deadline):
    started = time.monotonic()
    result = evaluate(cfg, nominal, path, 'state_settling', min(started + 1800, deadline))
    result['worker_elapsed_s'] = time.monotonic() - started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'configuration', 'run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    inputs = read(a.source / 'inputs.json')
    pins = dict(inputs['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed source ' + path)
    if pins.get(str(a.configuration.resolve())) != sha(a.configuration):
        raise ValueError('Keep original controller configuration')
    if read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native state evidence required')
    for name in ['native_command_inputs.json']:
        path = a.run / name
        if pins.get(str(path.resolve())) != sha(path):
            raise ValueError('Native source differs from the paired experiment')
    paths = sorted((a.source / 'candidate_snapshots').glob('state_*.json'))
    if len(paths) != 2:
        raise ValueError('Expected the two declared immutable complete references')
    descriptions = []
    for path in paths:
        if pins.get(str(path.resolve())) != sha(path):
            raise ValueError('Unpinned reference snapshot')
        data = read(path)
        required = 800 - data['initial_actual_state']['tick']
        if (len(data['fine']['trajectory']) != required or data['fine']['substeps'] != 160
                or data['native_success'] or data['valid_expert_labels']):
            raise ValueError('Require full-episode unverified fine-model reference')
        descriptions.append(dict(state_key=data['state_key'], required_steps=required))
    # Recheck actual state equality rather than assuming an inherited manifest
    # includes the large raw command archive (the preceding one did not).
    with np.load(a.run / 'pure_control_commands.npz') as data:
        for path in paths:
            record = read(path)['initial_actual_state']
            ids = np.flatnonzero((data['lane'] == record['case']) & (data['tick'] == record['tick']))
            if len(ids) != 1 or any(not np.array_equal(data[k][ids[0]], record[k])
                    for k in ['q', 'v', 'previous', 'ground_force']):
                raise ValueError('Candidate state differs from recorded native observation')
    files = [Path(__file__), a.source / 'inputs.json', a.run / 'job.json',
        a.run / 'pure_control_commands.npz']
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
        if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(path.resolve()): sha(path) for path in files})
    summary = dict(cases=descriptions, feedback_candidate_substeps=16,
        physical_substeps=160, workers=2, worker_seconds=1800, total_seconds=1900,
        native_execution=False, hardware_changed=False, training=False)
    if a.check_only:
        print(json.dumps(summary))
        return
    if a.output.exists():
        raise ValueError('Fresh result directory required')
    a.output.mkdir(parents=True)
    write(a.output / 'inputs.json', dict(summary, source_sha256=pins))
    cfg = read(a.configuration)
    nominal = read(a.run / 'native_command_inputs.json')['default_joint_position_rad']
    started = time.monotonic()
    results = []
    write(a.output / 'status.json', dict(state='RUNNING_COARSE_SETTLING_COUNTERFACTUAL',
        pid=os.getpid(), completed=0, total=2))
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_one, cfg, nominal, str(path), started + 1900) for path in paths]
        for future in as_completed(futures):
            result = future.result()
            write(a.output / (result['state_key'].replace(':', '_') + '.json'), result)
            results.append({k: v for k, v in result.items() if k not in ['trajectory', 'initial_actual_state']})
            write(a.output / 'status.json', dict(state='RUNNING_COARSE_SETTLING_COUNTERFACTUAL',
                pid=os.getpid(), completed=len(results), total=2, results=results))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed diagnostic source ' + path)
    result = dict(state='COMPLETED_COARSE_SETTLING_COUNTERFACTUAL', results=results,
        wall_seconds=time.monotonic() - started, native_success=False, valid_expert_labels=False)
    write(a.output / 'result.json', result)
    write(a.output / 'status.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
