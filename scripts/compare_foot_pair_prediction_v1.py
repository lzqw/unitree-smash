"""Paired physical-step evaluation of the foot-foot contact candidate.

Uses every recorded interval, the same actual start state and next-buffer
executed command. A numerical improvement is not native recovery evidence.
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
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.foot_pair_contact_step_v1 import foot_pair_velocity_step, foot_pair_geometry
from scripts.audit_joint_sequence_proposal_v1 import plain


def predict(job):
    try:
        model = _WORKER['controller'].qp.model
        interface = _WORKER['controller'].qp.interface
        before, after = state_from_row(job['before']), state_from_row(job['after'])
        result = foot_pair_velocity_step(model, interface, before, job['target'],
            _WORKER['spheres'], job['dt'], foot_friction=job['friction'])
        dq = pin.difference(model.model, after.q,
            pin.integrate(model.model, before.q, job['dt'] * result['velocity']))
        dv = result['velocity'] - after.v
        return dict(case=job['case'], physics_frame=job['frame'], completed=True,
            old_joint_velocity_error_rad_s=job['old_error'],
            joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
            joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
            root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
            root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
            pair_gaps_m=result['pair_gaps'], pair_impulse_ns=result['pair_impulse'],
            ground_impulse_ns=result['ground_impulse'], momentum_residual=result['momentum_residual'],
            external_pair_base_wrench_max=result['external_pair_base_wrench_max'],
            model_joint_velocity=result['velocity'][model.v_indices])
    except (ValueError, np.linalg.LinAlgError) as e:
        return dict(case=job['case'], physics_frame=job['frame'], completed=False,
            old_joint_velocity_error_rad_s=job['old_error'], error=repr(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run', 'baseline', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline = read(a.baseline)
    pins = dict(baseline['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed audited input ' + path)
    plan = read(a.plan)
    source = read(plan['source'])
    *_, context = source_context(source['configuration']['source'])
    initialize(*context)
    model = _WORKER['controller'].qp.model
    meta = read(a.run / 'foot_impact_observation.json')
    materials = read(a.run / 'native_admission.json')['material_admission']['feet']
    if len(materials) != 4 or any(m['material']['combine'] != 'average' for m in materials):
        raise ValueError('Require original four native material records')
    friction = materials[0]['material']['dynamic']
    if any(m['material']['dynamic'] != friction or m['material']['static'] != friction for m in materials):
        raise ValueError('This candidate requires identical foot coefficients')
    with np.load(a.run / 'foot_impact_observation.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    old = {(r['case'], r['physics_frame']): r for r in baseline['rows']}
    jobs = []
    geometric_error = 0.
    for lane, case in enumerate(meta['lanes']):
        states = []
        for i in range(len(data['physics_frame'])):
            sample = {k: v[i, lane] for k, v in data.items() if k != 'physics_frame'}
            s = state_from_sample(model, sample, meta['origins_world'][lane])
            states.append(dict(q=s.q, v=s.v, contacts=s.contacts, target=s.previous_command_target))
        # Independent finite-difference check of every sphere-pair normal
        # Jacobian at two actual configurations, including a collision frame.
        for index in [0, min(710, len(states) - 1)]:
            q = states[index]['q']
            pairs = foot_pair_geometry(model, q, _WORKER['spheres'])
            direction = np.linspace(-.7, .8, model.model.nv)
            epsilon = 1e-7
            plus = foot_pair_geometry(model, pin.integrate(model.model, q, epsilon * direction), _WORKER['spheres'])
            minus = foot_pair_geometry(model, pin.integrate(model.model, q, -epsilon * direction), _WORKER['spheres'])
            geometric_error = max(geometric_error, max(abs((b['gap'] - c['gap']) / (2 * epsilon)
                - d['jacobian'][2] @ direction) for b, c, d in zip(plus, minus, pairs)))
        for i, frame in enumerate(data['physics_frame'][:-1]):
            prior = old[case, int(frame)]
            if not prior['completed']:
                raise ValueError('Baseline incomplete, do not silently select successful intervals')
            jobs.append(dict(case=case, frame=int(frame), before=states[i], after=states[i + 1],
                target=states[i + 1]['target'], dt=meta['physics_dt_s'], friction=friction,
                old_error=prior['maximum_joint_velocity_error_rad_s']))
    if geometric_error > 1e-7:
        raise ValueError('Pair normal Jacobian failed geometric derivative check')
    for path in [a.plan, a.baseline, a.run / 'native_admission.json', Path(__file__).resolve(),
                 ROOT / 'scripts/foot_pair_contact_step_v1.py']:
        pins[str(path.resolve())] = sha(path)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=context) as pool:
        rows = list(pool.map(predict, jobs, chunksize=16))
    summaries = {}
    for case in meta['lanes']:
        selected = [r for r in rows if r['case'] == case]
        done = [r for r in selected if r['completed']]
        summaries[str(case)] = dict(intervals=len(selected), completed=len(done),
            maximum_error_rad_s=max((r['joint_velocity_error_rad_s'] for r in done), default=None),
            median_error_rad_s=float(np.median([r['joint_velocity_error_rad_s'] for r in done])) if done else None,
            improved_intervals=sum(r['joint_velocity_error_rad_s'] < r['old_joint_velocity_error_rad_s'] - 1e-6 for r in done),
            worsened_intervals=sum(r['joint_velocity_error_rad_s'] > r['old_joint_velocity_error_rad_s'] + 1e-6 for r in done),
            original_large_errors=[r for r in selected if r['old_joint_velocity_error_rad_s'] > 1.],
            failures=[r for r in selected if not r['completed']])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Input changed during paired evaluation ' + path)
    result = dict(scope=__doc__, summaries=summaries, rows=rows, source_sha256=pins,
        geometric_derivative_max_error=geometric_error, native_foot_friction=friction,
        seconds=time.monotonic() - start, native_controller_changed=False, valid_expert_labels=False,
        model_promoted=False)
    tmp = a.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
    tmp.replace(a.output)
    print(json.dumps({c: {k: v for k, v in s.items() if k not in ['original_large_errors', 'failures']}
        for c, s in summaries.items()}))


if __name__ == '__main__':
    main()
