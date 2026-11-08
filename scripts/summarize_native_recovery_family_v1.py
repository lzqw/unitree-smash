"""Summarize native reference-transfer outcomes without promoting the teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
from scripts.diagnose_frozen_trace_divergence_v1 import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'baseline', 'audit', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh evidence required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit = read(a.audit); meta = read(a.run / 'recovery_family_result.json')
    pins = dict(audit['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed executed evidence ' + path)
    old, new = [RecordedNonfootContactHistory(path) for path in [a.baseline, a.run]]
    with np.load(a.run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        q85, v85 = z['q'][0].copy(), z['v'][0].copy()
    outcomes = {r['training_index']: r for r in audit['family_outcomes']}
    rows = []
    for case in meta['cases']:
        if case == 85:
            q, v = q85, v85
        else:
            alignment = meta['alignments'][str(case)]
            q, v = np.asarray(alignment['initial_q']), np.asarray(alignment['initial_v'])
        r = pin.Quaternion(q[3:7]).matrix()
        original_time, actual_time = old.first_frame[case] * old.dt, new.first_frame[case] * new.dt
        rows.append(dict(case=case, complete5=outcomes[case]['complete5'],
            initial_base_height_m=float(q[2]), initial_tilt_rad=float(np.arccos(np.clip(r[2, 2], -1., 1.))),
            initial_linear_speed_m_s=float(np.linalg.norm(v[:3])), initial_angular_speed_rad_s=float(np.linalg.norm(v[3:6])),
            old_first_scored_nonfoot_s=float(original_time) if np.isfinite(original_time) else None,
            actual_first_scored_nonfoot_s=float(actual_time) if np.isfinite(actual_time) else None,
            first_nonfoot_delay_s=float(actual_time - original_time) if np.isfinite(original_time + actual_time) else None))
    lost = []
    for case in audit['complete5_changes']['lost']:
        row, more = compare(a.baseline, a.run, case); lost.append(row); pins.update(more)
    for path in [Path(__file__), a.audit, a.run / 'recovery_family_result.json']:
        pins[str(path.resolve())] = sha(path)
    for directory in [a.run, a.baseline]:
        for name in ['first_nonfoot_contact_details.json', 'contact_telemetry_metadata.json']:
            path = directory / name; pins[str(path.resolve())] = sha(path)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Evidence changed during summary ' + path)
    result = dict(scope=__doc__, complete5=audit['complete5'], changes=audit['complete5_changes'],
        ball5_changes=audit['ball5_changes'], selected_success=sum(r['complete5'] for r in rows),
        cases=rows, unselected_differing_cases=[r['case'] for r in audit['differing_untouched']],
        lost_case_divergence=lost, source_sha256=pins, accepted_for_generalization=False,
        student_updated=False, valid_expert_labels=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete5', 'changes', 'selected_success', 'cases']}))


if __name__ == '__main__':
    main()
