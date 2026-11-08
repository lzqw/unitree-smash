"""Compare each accepted MPC prediction with its actual execution interval.

The prediction row at control tick t is the post-step state at t+1. Intervals
stop at the next replan; future predictions are never compared to execution
under a different plan. This is accumulated prediction error with feedback,
not an independent one-step model error or an expert-data admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def audit(run):
    read = lambda name: json.loads((run/name).read_text())
    job = read('job.json')
    native = read('sequence_native_result.json')
    if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0
            or not native.get('online_mpc') or not native['query_match']['passed']):
        raise ValueError('Completed native MPC execution required; offline queries do not qualify')
    planning = read('foot_space_current_state_planning.json')
    calls = planning['calls']
    if not calls or len(calls) != native['planning_calls']:
        raise ValueError('Recorded planning calls differ')
    with np.load(run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    ticks = [r['control_tick'] for r in native['records']]
    if ticks != list(range(280, 800)) or any(len(v) != len(ticks) for v in trace.values()):
        raise ValueError('Full consecutive original native remainder required')
    if not np.array_equal(trace['previous_target_rad'][1:], trace['final_target_rad'][:-1]):
        raise ValueError('Actual command history changed')
    model = LoadedPinocchioModelV1()
    intervals = []
    for ci, call in enumerate(calls):
        start = call['control_tick']
        end = calls[ci+1]['control_tick'] if ci+1 < len(calls) else 800
        if not 280 <= start < end <= 800 or not call['accepted'] or not call['independent_replay_exact']:
            raise ValueError('Invalid accepted planning interval')
        index = start-280
        for key, field in [('q', 'q'), ('v', 'v'), ('contacts', 'contacts'), ('target', 'previous_target_rad')]:
            if not np.array_equal(call['query'][key], trace[field][index]):
                raise ValueError('Planner query differs from actual trace: '+key)
        predicted = call['prediction']['trajectory']
        if len(predicted) != 800-start or [r['control_tick'] for r in predicted] != list(range(start, 800)):
            raise ValueError('Prediction does not cover the declared original clock')
        rows = []
        for row in predicted[:end-start]:
            tick = row['control_tick']
            index = tick-280
            # The trace stops at the pre-step state of tick799. Do not fabricate
            # a tick800 observation from the model or next episode's reset.
            if tick+1 >= 800:
                continue
            delta = pin.difference(model.model, trace['q'][index+1], np.asarray(row['q']))
            rows.append(dict(control_tick=tick, observed_state_tick=tick+1,
                root_position_error_m=float(np.linalg.norm(delta[:3])),
                root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
                maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
                maximum_velocity_error=float(np.max(abs(np.asarray(row['v'])-trace['v'][index+1]))),
                maximum_command_difference_rad=float(np.max(abs(np.asarray(row['target'])-trace['final_target_rad'][index]))),
                contact_flags_equal=bool(np.array_equal(row['contacts'], trace['contacts'][index+1]))))
        if not rows:
            raise ValueError('No observed execution interval')
        metrics = [k for k in rows[0] if 'error' in k or 'difference' in k]
        intervals.append(dict(start_tick=start, next_replan_or_end_tick=end,
            observed_steps=len(rows), missing_original_terminal_observation=end == 800,
            planner_wall_seconds=call['wall_seconds'],
            first_command_error_rad=float(np.max(abs(np.asarray(call['first_command'])-trace['final_target_rad'][start-280]))),
            summary={k: dict(maximum=max(r[k] for r in rows), median=float(np.median([r[k] for r in rows]))) for k in metrics},
            contact_flag_agreement_fraction=float(np.mean([r['contact_flags_equal'] for r in rows])), rows=rows))
    files = [run/name for name in ['job.json', 'sequence_native_result.json',
        'sequence_native_trace.npz', 'foot_space_current_state_planning.json']]
    files += [Path(__file__).resolve(), ROOT/'scripts/loaded_pinocchio_model_v1.py']
    return dict(case=native['case'], intervals=intervals,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        new_native_execution=False, valid_expert_labels=False,
        strict_isolation_audit_waived=False,
        limitations=['Accumulated planned-versus-feedback execution, not a fixed-command one-step error.',
            'Measured contact flags and modeled contacts have different definitions.',
            'The original tick800 pre-reset endpoint is absent from this trace.',
            'Native recovery scoring and all original isolation checks remain separate.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh audit output required')
    result = audit(args.run)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(args.output)
    print(json.dumps(dict(case=result['case'], planning_intervals=len(result['intervals']))))
