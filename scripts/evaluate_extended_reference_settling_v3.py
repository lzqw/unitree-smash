"""Test transient stabilization in a completed search's incomplete reference.

All paired runs begin at the same recorded native release. Only candidate
scoring fidelity and the support-transition rule differ. No native restart,
threshold change, learning or expert labels. Each worker gets its budget on
actual start, with a separate bounded overall deadline.
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
from scripts.evaluate_state_settling_feedback_v1 import evaluate as coarse_evaluate
from scripts.evaluate_state_settling_feedback_v2 import evaluate as fine_evaluate
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def run_one(cfg, nominal, path, mode, substeps, deadline):
    start = time.monotonic()
    evaluate = coarse_evaluate if substeps == 16 else fine_evaluate
    result = evaluate(cfg, nominal, path, mode, min(deadline, start + 2400))
    result['candidate_scoring_substeps'] = substeps
    result['physical_substeps'] = 160
    result['worker_wall_seconds'] = time.monotonic() - start
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['search', 'configuration', 'run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--state-key', required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists() or read(a.search / 'result.json')['state'] != 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Completed search and fresh output required')
    if read(a.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Require completed native initial-state evidence')
    pins = dict(read(a.search / 'inputs.json')['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed search input ' + path)
    cfg = read(a.configuration)
    if pins.get(str(a.configuration.resolve())) != sha(a.configuration):
        raise ValueError('Keep original configuration')
    candidate_path = a.search / ('state_' + a.state_key.replace(':', '_') + '.json')
    raw = candidate_path.read_bytes()
    candidate = json.loads(raw)
    initial, plan = candidate['initial_actual_state'], candidate['fine']
    if candidate['state_key'] != a.state_key or candidate['native_success'] or candidate['valid_expert_labels']:
        raise ValueError('Require a model-only reference from the declared state')
    if plan['substeps'] != 160 or not plan['trajectory']:
        raise ValueError('Fine physical reference required')
    if any(row['step'] != i for i, row in enumerate(plan['trajectory'])):
        raise ValueError('Discontinuous combined reference')
    archive = a.run / 'pure_control_commands.npz'
    with np.load(archive) as z:
        ids = np.flatnonzero((z['lane'] == initial['case']) & (z['tick'] == initial['tick']))
        if len(ids) != 1 or any(not np.array_equal(z[k][ids[0]], initial[k]) for k in ['q', 'v', 'previous', 'ground_force']):
            raise ValueError('Reference does not begin at the recorded actual state')
    nominal = read(a.run / 'native_command_inputs.json')['default_joint_position_rad']
    a.output.mkdir(parents=True)
    snapshots = a.output / 'candidate_snapshots'
    snapshots.mkdir()
    snapshot = snapshots / candidate_path.name
    snapshot.write_bytes(raw)
    files = [Path(__file__), candidate_path, snapshot, archive, a.search / 'result.json',
        a.search / 'inputs.json', a.run / 'job.json', a.run / 'native_command_inputs.json']
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
        if name.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(path.resolve()): sha(path) for path in files})
    write(a.output / 'inputs.json', dict(source_sha256=pins, state_key=a.state_key,
        actual_initial_state=initial, reference_steps=len(plan['trajectory']),
        original_reference_failure=plan['failure'], workers=4,
        worker_seconds=2400, total_seconds=2500, physical_substeps=160,
        scoring_substeps=[16, 160], transition_modes=['plan_end', 'state_settling'],
        native_execution=False, training=False, hardware_changed=False, acceptance_changed=False))
    start = time.monotonic()
    results = []
    def progress(state):
        write(a.output / 'status.json', dict(state=state, pid=os.getpid(), completed=len(results),
            total=4, results=results, elapsed_s=time.monotonic() - start,
            native_success=False, valid_expert_labels=False))
    progress('RUNNING_EXTENDED_REFERENCE_SETTLING')
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(run_one, cfg, nominal, str(snapshot), mode, substeps, start + 2500)
            for substeps in [16, 160] for mode in ['plan_end', 'state_settling']]
        for job in as_completed(jobs):
            row = job.result()
            filename = a.state_key.replace(':', '_') + '_' + str(row['candidate_scoring_substeps']) + '_' + row['mode'] + '.json'
            write(a.output / filename, row)
            results.append({key: value for key, value in row.items() if key not in ['trajectory', 'initial_actual_state']})
            progress('RUNNING_EXTENDED_REFERENCE_SETTLING')
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed diagnostic input ' + path)
    summary = dict(state='COMPLETED_EXTENDED_REFERENCE_SETTLING', results=results,
        elapsed_s=time.monotonic() - start, native_success=False, valid_expert_labels=False)
    write(a.output / 'result.json', summary)
    progress(summary['state'])
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
