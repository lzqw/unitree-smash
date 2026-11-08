"""Separate incomplete reference endings from feedback damage to a valid prefix.

Read immutable per-case model results. A running batch status is snapshotted
only to select finished entries, not treated as a completed batch or success.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'retries', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    if a.output.exists():
        raise ValueError('Fresh audit output required')
    raw_status = (a.retries / 'status.json').read_bytes()
    status = json.loads(raw_status)
    expected = read(a.source / 'result.json')['results']
    source_pins = read(a.source / 'inputs.json')['source_sha256']
    finished = status.get('results', [])
    entries, files = [], []
    for summary in finished:
        stem = summary['state_key'].replace(':', '_')
        name = stem + '_' + summary['mode'] + '.json'
        result_path = a.retries / name
        if not result_path.exists():
            result_path = a.source / name
        result = read(result_path)
        if any(result.get(k) != summary.get(k) for k in ['state_key', 'mode', 'steps', 'failure']):
            raise ValueError('Status and finished per-case artifact disagree')
        reference_path = a.source / 'candidate_snapshots' / ('state_' + stem + '.json')
        if source_pins.get(str(reference_path.resolve())) != sha(reference_path):
            raise ValueError('Changed immutable reference snapshot')
        ref = read(reference_path)
        if ref['initial_actual_state'] != result['initial_actual_state']:
            raise ValueError('Different initial state')
        planned, actual = ref['fine']['trajectory'], result['trajectory']
        common = min(len(planned), len(actual))
        du = [float(np.max(abs(np.asarray(planned[k]['target']) - actual[k]['target']))) for k in range(common)]
        changed = next((k for k, v in enumerate(du) if v > 1e-8), None)
        failure = result['failure']
        failure_step = failure['tick'] - ref['initial_actual_state']['tick'] if failure else None
        phase = ('not_failed' if failure is None else 'compute_incomplete' if failure['reason'] == 'compute_budget'
            else 'before_reference_end' if failure_step < len(planned) else 'after_reference_end')
        boundary = actual[len(planned)] if len(actual) > len(planned) else None
        entries.append(dict(state_key=ref['state_key'], mode=result['mode'], reference_steps=len(planned),
            executed_steps=len(actual), full_model_path=result['full_model_path'],
            common_reference_steps=common, maximum_command_difference_during_reference=max(du, default=0.),
            first_command_difference_step=changed, failure_location=phase, failure=failure,
            capture_tick=result['state_settling_capture_tick'],
            final_model_stable_s=result['final_model_stable_s'],
            first_after_reference_command_source=boundary['source'] if boundary else None,
            reference_end_stable_s=planned[-1]['phase']['control_sample_stable_duration_s']))
        files += [result_path, reference_path]
    keys = {(r['state_key'], r['mode']) for r in entries}
    pending = [(r['state_key'], r['mode']) for r in expected if (r['state_key'], r['mode']) not in keys]
    a.output.mkdir(parents=True)
    status_path = a.output / 'observed_status.json'
    status_path.write_bytes(raw_status)
    files += [Path(__file__), status_path, a.source / 'inputs.json', a.source / 'result.json']
    result = dict(entries=entries, pending=pending, source_sha256={str(f.resolve()): sha(f) for f in files},
        native_execution=False, valid_expert_labels=False, batch_complete=not pending,
        interpretation='An incomplete reference is not a complete landing demonstration.')
    write(a.output / 'result.json', result)
    print(json.dumps(dict(entries=entries, pending=pending)))


if __name__ == '__main__':
    main()
