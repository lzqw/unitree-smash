"""Evaluate endpoint velocity correction on every audited physical interval."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.endpoint_contact_velocity_probe_v1 import endpoint_velocity_probe
from scripts.audit_joint_sequence_proposal_v1 import plain


def predict(job):
    try:
        model, interface = _WORKER['controller'].qp.model, _WORKER['controller'].qp.interface
        before, after = state_from_row(job['before']), state_from_row(job['after'])
        r = endpoint_velocity_probe(model, interface, before, job['target'], _WORKER['spheres'],
            job['dt'], foot_friction=job['friction'])
        dv = r['velocity'] - after.v
        dq = pin.difference(model.model, after.q, r['q'])
        return dict(case=job['case'], physics_frame=job['frame'], completed=True,
            joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
            first_stage_error_rad_s=float(np.max(abs(r['first_velocity'][model.v_indices] - after.v[model.v_indices]))),
            joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
            root_position_error_m=float(np.linalg.norm(dq[:3])),
            root_linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
            root_angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:6])),
            ground_impulse_ns=r['ground_impulse'], pair_impulse_ns=r['pair_impulse'],
            actual_joint_velocity=after.v[model.v_indices], model_joint_velocity=r['velocity'][model.v_indices],
            momentum_residual=r['momentum_residual'])
    except (ValueError, np.linalg.LinAlgError) as e:
        return dict(case=job['case'], physics_frame=job['frame'], completed=False, error=repr(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run', 'baseline', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline = read(a.baseline); pins = dict(baseline['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed evidence ' + path)
    source = read(read(a.plan)['source']); *_, context = source_context(source['configuration']['source'])
    initialize(*context); model = _WORKER['controller'].qp.model
    meta = read(a.run / 'foot_impact_observation.json')
    materials = read(a.run / 'native_admission.json')['material_admission']['feet']
    friction = materials[0]['material']['dynamic']
    if any(m['material']['combine'] != 'average' or m['material']['dynamic'] != friction
            or m['material']['static'] != friction for m in materials):
        raise ValueError('Original uniform foot materials required')
    with np.load(a.run / 'foot_impact_observation.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    jobs = []
    for lane, case in enumerate(meta['lanes']):
        states = []
        for i in range(len(data['physics_frame'])):
            sample = {k: v[i, lane] for k, v in data.items() if k != 'physics_frame'}
            s = state_from_sample(model, sample, meta['origins_world'][lane])
            states.append(dict(q=s.q, v=s.v, contacts=s.contacts, target=s.previous_command_target))
        for i, frame in enumerate(data['physics_frame'][:-1]):
            jobs.append(dict(case=case, frame=int(frame), before=states[i], after=states[i + 1],
                target=states[i + 1]['target'], dt=meta['physics_dt_s'], friction=friction))
    for path in [a.plan, a.baseline, Path(__file__).resolve(), ROOT / 'scripts/endpoint_contact_velocity_probe_v1.py']:
        pins[str(path.resolve())] = sha(path)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=context) as pool:
        rows = list(pool.map(predict, jobs, chunksize=16))
    old = {(r['case'], r['physics_frame']): r for r in baseline['rows']}
    summaries = {}
    for case in meta['lanes']:
        selected = [r for r in rows if r['case'] == case]; done = [r for r in selected if r['completed']]
        for r in done:
            r['pair_only_error_rad_s'] = old[case, r['physics_frame']]['joint_velocity_error_rad_s']
            if abs(r['first_stage_error_rad_s'] - r['pair_only_error_rad_s']) > 1e-10:
                raise ValueError('First stage did not reproduce paired baseline')
        summaries[str(case)] = dict(intervals=len(selected), completed=len(done),
            max_error_rad_s=max((r['joint_velocity_error_rad_s'] for r in done), default=None),
            median_error_rad_s=float(np.median([r['joint_velocity_error_rad_s'] for r in done])) if done else None,
            improved=sum(r['joint_velocity_error_rad_s'] < r['pair_only_error_rad_s'] - 1e-6 for r in done),
            worsened=sum(r['joint_velocity_error_rad_s'] > r['pair_only_error_rad_s'] + 1e-6 for r in done),
            large_initial_errors=[r for r in done if r['pair_only_error_rad_s'] > 1.],
            failures=[r for r in selected if not r['completed']])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Input changed during comparison ' + path)
    result = dict(scope=__doc__, summaries=summaries, rows=rows, source_sha256=pins,
        native_controller_changed=False, model_promoted=False, valid_expert_labels=False)
    tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n'); tmp.replace(a.output)
    print(json.dumps({c: {k: v for k, v in s.items() if k not in ['large_initial_errors', 'failures']}
        for c, s in summaries.items()}))


if __name__ == '__main__':
    main()
