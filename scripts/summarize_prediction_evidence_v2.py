"""Report recorded prediction errors without turning solver success into admission."""
import argparse
import json
import math
from pathlib import Path


def summarize(record):
    groups = []
    for method in record['summary']:
        for lane in (0, 1):
            rows = [r for r in record['rows'] if r['method'] == method and r['lane'] == lane]
            solved = [r for r in rows if r['solved']]
            first = {}
            for key in ('joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                        'motor_torque_violation_nm', 'foot_penetration_m'):
                missing = [r['tick'] for r in solved if key not in r['diagnostics']
                           or not math.isfinite(r['diagnostics'][key])]
                violations = [dict(tick=r['tick'], value=r['diagnostics'][key]) for r in solved
                              if key in r['diagnostics'] and math.isfinite(r['diagnostics'][key])
                              and r['diagnostics'][key] > 1e-6]
                # 1e-6 is a diagnostic display threshold, NOT a new physical admission tolerance.
                first[key] = dict(first_above_display_threshold=next(iter(violations), None),
                                  missing_or_nonfinite_ticks=missing)
            groups.append(dict(method=method, lane=lane, solved=len(solved), requested=record['steps'],
                first_solver_failure=next((dict(tick=r['tick'], reason=r['error']) for r in rows if not r['solved']), None),
                first_racket_body_error_above_3mm=next((dict(tick=r['tick'], error_m=r['racket_position_error_m'])
                    for r in solved if r['racket_position_error_m'] > .003), None),
                diagnostic_evidence=first,
                physical_admission='NOT_EVALUATED', contact_point_accuracy='NOT_MEASURED'))
    return dict(mode=record['mode'], start_tick=record['start_tick'], steps=record['steps'], groups=groups,
                valid_teacher_reference=False, explanation='Solver convergence is not physical admission. '
                'Racket body origin is not the actual contact point. Missing checks remain unknown.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(summarize(json.loads(args.input.read_text())), stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
