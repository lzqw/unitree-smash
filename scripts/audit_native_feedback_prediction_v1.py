"""Compare one-step prediction with recorded native feedback execution.

Diagnostic only: no native control, no scoring changes, no expert promotion.
Unlike accumulated nominal tracking error, each prediction starts at the
actual measured state and uses the actual command dispatched at that tick.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def predict(job):
    w = _WORKER
    model = w['controller'].qp.model
    state = state_from_row(job['before'])
    actual = state_from_row(job['after'])
    gap, _ = w['geometry'].evaluate(state.q)
    result = dict(tick=job['tick'], actual_initial_nonfoot_gap_m=float(gap.min()),
                  measured_foot_normal_force_n=job['foot_force'][:, 2],
                  observed_contact_flags=state.contacts)
    try:
        prediction = predict_speed_limited_spheres(model, w['controller'].qp.interface,
            state, job['target'], w['spheres'], geometry=w['geometry'], substeps=160)
        following = prediction['state']
        delta = pin.difference(model.model, actual.q, following.q)
        result.update(completed=True, diagnostics=prediction['diagnostics'],
            first_predicted_ground_violation=prediction['first_ground_violation'],
            root_position_error_m=float(np.linalg.norm(delta[:3])),
            root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
            root_linear_velocity_error_m_s=float(np.linalg.norm(following.v[:3]-actual.v[:3])),
            root_angular_velocity_error_rad_s=float(np.linalg.norm(following.v[3:6]-actual.v[3:6])),
            maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
            maximum_joint_velocity_error_rad_s=float(np.max(abs(following.v[model.v_indices]-actual.v[model.v_indices]))))
    except (ValueError, np.linalg.LinAlgError) as error:
        result.update(completed=False, error=repr(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'run', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plan = read(args.plan)
    native = read(args.run/'sequence_native_result.json')
    if not native['native_execution'] or not native['query_match']['passed']:
        raise ValueError('Matching actual first-episode query required')
    source = read(plan['source'])
    _, _, _, _, context = source_context(source['configuration']['source'])
    with np.load(args.run/'sequence_native_trace.npz', allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    records = native['records']
    count = len(records)
    if count < 2 or any(len(value) != count for value in trace.values()):
        raise ValueError('Aligned consecutive native trace required')
    ticks = np.asarray([row['control_tick'] for row in records])
    if not np.all(np.diff(ticks) == 1):
        raise ValueError('Cannot predict across missing ticks or resets')
    if not np.isfinite(np.concatenate([trace['q'].ravel(), trace['v'].ravel(),
                                      trace['final_target_rad'].ravel()])).all():
        raise ValueError('Nonfinite native evidence')
    history_error = float(np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1])))
    if history_error > 5e-6:
        raise ValueError('Actual command history is discontinuous')
    # Dense first 0.5 s, then fixed 0.1 s spacing; selection is not based on error.
    indices = sorted(set(range(min(50, count-1))) | set(range(50, count-1, 10)))
    def state(index):
        return dict(q=trace['q'][index], v=trace['v'][index], contacts=trace['contacts'][index],
                    target=trace['previous_target_rad'][index])
    jobs = [dict(tick=int(ticks[i]), before=state(i), after=state(i+1),
                 target=trace['final_target_rad'][i], foot_force=trace['foot_force_world_n'][i])
            for i in indices]
    pins = dict(plan['source_sha256'])
    for path in [args.plan, args.run/'sequence_native_result.json', args.run/'sequence_native_trace.npz']:
        pins[str(path.resolve())] = sha(path)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve()); pins[path] = sha(path)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source or evidence changed: '+path)
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=context) as pool:
        rows = list(pool.map(predict, jobs))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source or evidence changed during diagnostic: '+path)
    good = [row for row in rows if row['completed']]
    metrics = [key for key in good[0] if '_error_' in key] if good else []
    summary = {key: dict(maximum=max(row[key] for row in good),
                        median=float(np.median([row[key] for row in good]))) for key in metrics}
    result = dict(case=native['case'], source_sha256=pins, intervals=len(rows),
        completed=len(good), actual_history_error_rad=history_error, rows=rows, summary=summary,
        seconds=time.monotonic()-began, new_native_execution=False, valid_expert_labels=False,
        limitations=['10 ms measured-state predictions, not full recovery verification.',
            'Foot contact and simulator speed limits modeled; nonfoot/self-contact impulses are not.',
            'Observation contact flags and predicted loaded feet use different definitions.',
            'An isolation-audit failure remains a failure; this diagnostic cannot waive it.'])
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(intervals=len(rows), completed=len(good), summary=summary)))


if __name__ == '__main__':
    main()
