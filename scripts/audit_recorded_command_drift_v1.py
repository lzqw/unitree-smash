"""Compare recorded-command predictions with measured next states.

This is an offline error audit, not new simulation or a causal attribution
to a particular contact parameter. Contact flags have different definitions
and are deliberately not scored as interchangeable ground-truth labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh audit output required')
    prediction = json.loads(args.prediction.read_text())
    with np.load(args.trace, allow_pickle=False) as source:
        trace = {k: source[k].copy() for k in source.files}
    if trace['q'].shape != (520, 25) or trace['v'].shape != (520, 24):
        raise ValueError('Original ticks 280 through 799 required')
    audits = []
    for row in prediction['rows']:
        result = row['result']
        initial = row['tick'] - 280
        if not (np.array_equal(result['initial_q'], trace['q'][initial])
                and np.array_equal(result['initial_v'], trace['v'][initial])):
            raise ValueError('Prediction did not start from measured state')
        errors = []
        for state in result['trajectory']:
            tick = state['control_tick']
            index = tick - 280
            if index + 1 == 520:
                continue  # final next state was not saved in this trace
            q = np.asarray(state['q']); v = np.asarray(state['v'])
            actual_q = trace['q'][index + 1]
            actual_v = trace['v'][index + 1]
            dot = abs(np.dot(q[3:7], actual_q[3:7]))
            dot /= np.linalg.norm(q[3:7]) * np.linalg.norm(actual_q[3:7])
            errors.append(dict(next_tick=tick + 1,
                command_error_rad=float(np.max(abs(np.asarray(state['target']) - trace['requested_target_rad'][index]))),
                root_position_error_m=float(np.linalg.norm(q[:3] - actual_q[:3])),
                root_rotation_error_rad=float(2 * np.arccos(np.clip(dot, 0, 1))),
                joint_position_error_rad=float(np.max(abs(q[7:] - actual_q[7:]))),
                joint_velocity_error_rad_s=float(np.max(abs(v[6:] - actual_v[6:])))))
        thresholds = {'root_position_error_m': .01, 'joint_position_error_rad': .1}
        audits.append(dict(start_tick=row['tick'], compared_steps=len(errors),
            maximum={k: max(e[k] for e in errors) for k in errors[0] if k != 'next_tick'},
            first_threshold_crossing={k: next((e['next_tick'] for e in errors if e[k] > value), None)
                                      for k, value in thresholds.items()},
            diagnostic_thresholds=thresholds, failure=result['failure'], rows=errors))
    paths = [args.prediction, args.trace, Path(__file__)]
    output = dict(scope=__doc__, audits=audits, native_execution=False,
                  valid_expert_labels=False, acceptance_thresholds_changed=False,
                  source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for row in audits:
        print(json.dumps({k: v for k, v in row.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
