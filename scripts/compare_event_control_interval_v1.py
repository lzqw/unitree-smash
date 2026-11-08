"""Paired 10 ms predictions using passive physical records and actual commands.

Compare the existing warm-start predictor, its cold-start equivalent, and
event-resolved integration. Never reset to measurements inside an interval.
"""
import argparse
import copy
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
from scripts.event_resolved_foot_step_v1 import event_resolved_foot_step
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres, surface_jacobians
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def cold_step(model, interface, state, target, spheres, dt):
    snapshot = model.snapshot(state.q, state.v)
    gap = sphere_gaps(model, state.q, spheres)
    selection = np.zeros((24, 18))
    selection[model.v_indices, np.arange(18)] = 1.
    result = speed_limited_velocity_step(
        snapshot['mass_matrix'], state.v, snapshot['nonlinear'], selection,
        interface.kp, interface.kd, target - state.q[model.q_indices],
        interface.effort, dt, surface_jacobians(model, state.q, spheres).reshape(12, 24),
        -gap / np.where(gap > 0, dt, .01), interface.friction, interface.velocity)
    following = copy.deepcopy(state)
    following.v = result['velocity']
    following.q = pin.integrate(model.model, state.q, dt * following.v)
    following.contacts = result['impulse'].reshape(4, 3)[:, 2] > 1e-8
    return following


def compare(job):
    w = _WORKER
    model = w['controller'].qp.model
    interface = w['controller'].qp.interface
    actual = state_from_row(job['after'])
    result = dict(case=job['case'], tick=job['tick'], modes={})
    for mode in ['original_warm', 'original_cold', 'event_resolved']:
        state = state_from_row(job['before'])
        events = []
        try:
            if mode == 'original_warm':
                if not np.array_equal(job['targets'], np.broadcast_to(job['targets'][0], (160, 18))):
                    raise ValueError('Original predictor requires a constant control target')
                state = predict_speed_limited_spheres(model, interface, state,
                    job['targets'][0], w['spheres'], substeps=160)['state']
            else:
                for micro, target in enumerate(job['targets']):
                    if mode == 'original_cold':
                        state = cold_step(model, interface, state, target, w['spheres'], job['dt'])
                    else:
                        step = event_resolved_foot_step(model, interface, state, target,
                            w['spheres'], job['dt'])
                        state = step['state']
                        events.extend(dict(e, micro=micro) for e in step['events'])
            delta = pin.difference(model.model, actual.q, state.q)
            dv = state.v - actual.v
            result['modes'][mode] = dict(completed=True, events=events, metrics=dict(
                root_position_error_m=float(np.linalg.norm(delta[:3])),
                root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
                root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
                root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
                maximum_joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
                maximum_joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices])))))
        except (ValueError, np.linalg.LinAlgError) as error:
            result['modes'][mode] = dict(completed=False, error=repr(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'run', 'audit', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence = read(args.audit)
    if not evidence['passed'] or not evidence['passive_original_replay']:
        raise ValueError('Audited passive native evidence required')
    pins = dict(evidence['source_sha256'])
    plan = read(args.plan)
    source = read(plan['source'])
    _, _, _, _, context = source_context(source['configuration']['source'])
    initialize(*context)
    model = _WORKER['controller'].qp.model
    meta = read(args.run / 'foot_impact_observation.json')
    with np.load(args.run / 'foot_impact_observation.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    frames = data['physics_frame']
    if not np.all(np.diff(frames) == 1) or abs(meta['physics_dt_s'] - .0000625) > 1e-12:
        raise ValueError('Continuous original physical-step records required')
    jobs = []
    for lane, case in enumerate(meta['lanes']):
        states = []
        for i in range(len(frames)):
            sample = {k: v[i, lane] for k, v in data.items() if k != 'physics_frame'}
            state = state_from_sample(model, sample, meta['origins_world'][lane])
            states.append(dict(q=state.q, v=state.v, contacts=state.contacts,
                target=state.previous_command_target))
        for start, frame in enumerate(frames):
            if frame % 160 or start + 160 >= len(frames):
                continue
            jobs.append(dict(case=case, tick=int(frame // 160), before=states[start],
                after=states[start + 160], dt=meta['physics_dt_s'],
                targets=np.asarray([s['target'] for s in states[start + 1:start + 161]])))
    for path in [args.plan, args.audit, Path(__file__).resolve()]:
        pins[str(path.resolve())] = sha(path)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve())
            digest = sha(path)
            if path in pins and pins[path] != digest:
                raise ValueError('Pinned source changed ' + path)
            pins[path] = digest
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source changed ' + path)
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=context) as pool:
        rows = list(pool.map(compare, jobs))
    summary = {}
    for case in meta['lanes']:
        chosen = [r for r in rows if r['case'] == case]
        summary[str(case)] = {mode: dict(
            completed=sum(r['modes'][mode]['completed'] for r in chosen),
            errors={k: dict(maximum=max(values), median=float(np.median(values)))
                for k in chosen[0]['modes'][mode].get('metrics', {})
                if (values := [r['modes'][mode]['metrics'][k] for r in chosen
                    if r['modes'][mode]['completed']])})
            for mode in ['original_warm', 'original_cold', 'event_resolved']}
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source changed during comparison ' + path)
    output = dict(scope=__doc__, rows=rows, summary=summary, source_sha256=pins,
        seconds=time.monotonic() - began, new_native_execution=False, model_promoted=False,
        valid_expert_labels=False, limitations=[
            'Diagnostic states selected before event-model construction; not held-out generalization.',
            'Each 10 ms interval begins from measurement; intermediate states are predictions.',
            'Nonfoot/self-contact, finite PhysX iterations and friction memory remain unmodeled.'])
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(output), indent=2, allow_nan=False) + '\n')
    tmp.replace(args.output)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
