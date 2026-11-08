"""Compare a rejected model prefix with its actual continuous-feedback execution.

Model row t is the state AFTER command t; native row t is BEFORE command t.
This audit preserves that alignment and reports command divergence separately.
It neither extends a rejected model trajectory nor changes teacher admission.
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
from scripts.parallel_landing_solver_v1 import build_controller


def compare(model, predicted, native_ticks, trace):
    lookup = {int(t): i for i, t in enumerate(native_ticks)}
    if len(lookup) != len(native_ticks) or not np.all(np.diff(native_ticks) == 1):
        raise ValueError('Consecutive unique measured ticks required')
    rows = []
    for row in predicted:
        t = int(row['control_tick'])
        if t not in lookup or t + 1 not in lookup:
            raise ValueError('Measured post-command state missing; do not fabricate endpoint')
        i, j = lookup[t], lookup[t + 1]
        delta = pin.difference(model.model, trace['q'][j], np.asarray(row['q']))
        dv = np.asarray(row['v']) - trace['v'][j]
        target_delta = np.asarray(row['target']) - trace['final_target_rad'][i]
        rows.append(dict(command_tick=t, measured_endpoint_tick=t+1,
            root_position_error_m=float(np.linalg.norm(delta[:3])),
            root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
            maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
            maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
            maximum_command_difference_rad=float(np.max(abs(target_delta)))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['search', 'run', 'bundle', 'native-audit', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh evidence required')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    search, manifest, audit = read(a.search), read(a.bundle/'manifest.json'), read(a.native_audit)
    native = read(a.run/'sequence_native_result.json')
    if (read(a.run/'job.json')['status'] != 'SUCCEEDED' or not audit['passed_dispatch_integrity']
            or not native['query_match']['passed'] or native['model_accepted']
            or search['case'] != manifest['case'] or search['best']['plan'] != manifest['proposal']
            or search['best']['result']['accepted']):
        raise ValueError('Completed actual execution of the same rejected proposal required')
    pins = dict(audit['source_sha256'])
    for path, digest in search['source_sha256'].items():
        if path in pins and pins[path] != digest:
            raise ValueError('Evidence sources disagree '+path)
        pins[path] = digest
    for path in [a.search, a.bundle/'manifest.json', a.native_audit, Path(__file__).resolve()]:
        pins[str(path.resolve())] = sha(path)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Evidence changed '+path)
    model = build_controller(manifest['configuration']).qp.model
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {key: z[key].copy() for key in z.files}
    ticks = np.array([r['tick'] for r in native['records']], dtype=int)
    predicted = search['best']['result']
    if not np.allclose(trace['q'][0], predicted['initial_q'], atol=2e-5, rtol=0):
        raise ValueError('Initial query differs')
    rows = compare(model, predicted['trajectory'], ticks, trace)
    metrics = [k for k in rows[0] if 'error_' in k or 'difference_' in k]
    result = dict(scope=__doc__,case=manifest['case'],rows=rows,
        summary={k: dict(maximum=max(r[k] for r in rows), final=rows[-1][k]) for k in metrics},
        original_model_failure=predicted['failure'],
        original_model_completed_steps=predicted['completed_model_steps'],
        native_outcome=audit['selected_case_outcome'],
        strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],
        source_sha256=pins, valid_expert_labels=False, automatic_promotion=False,
        limitations=['Both paths use feedback: command differences contribute to accumulated state error.',
            'This comparison ends at the stored model prefix; no extrapolation beyond rejection.',
            'Model contact flags and measured force flags have different definitions and are not scored as equal labels.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(a.output)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
