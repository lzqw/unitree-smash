"""Propagate competing contact models for whole native 10 ms control intervals.

Both models start at the same measured state once per control interval;
there are no measured-state resets inside it. Commands are the native tape.
"""
import argparse
import copy
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
from scripts.foot_pair_contact_step_v1 import foot_pair_velocity_step
from scripts.endpoint_contact_velocity_probe_v1 import endpoint_velocity_probe


def predict(job):
    model, interface = _WORKER['controller'].qp.model, _WORKER['controller'].qp.interface
    results = {}
    for mode in ['pair_step', 'endpoint_velocity']:
        state = state_from_row(job['states'][0]); worst = 0.; first_bad_gap = None
        try:
            for index, recorded in enumerate(job['states'][1:]):
                observed = state_from_row(recorded)
                target = recorded['target']
                function = foot_pair_velocity_step if mode == 'pair_step' else endpoint_velocity_probe
                r = function(model, interface, state, target, _WORKER['spheres'], job['dt'], foot_friction=job['friction'])
                next_q = r.get('q')
                if next_q is None:
                    next_q = pin.integrate(model.model, state.q, job['dt'] * r['velocity'])
                state.q = next_q; state.v = r['velocity']; state.previous_command_target = target.copy()
                error = float(np.max(abs(state.v[model.v_indices] - observed.v[model.v_indices])))
                worst = max(worst, error)
                gaps, _ = _WORKER['geometry'].evaluate(state.q)
                if first_bad_gap is None and gaps.min() < -1e-6:
                    body = int(np.argmin(gaps))
                    first_bad_gap = dict(micro=index + 1, body=_WORKER['geometry'].shapes[body]['body'], gap_m=float(gaps[body]))
            after = state_from_row(job['states'][-1])
            dq = pin.difference(model.model, after.q, state.q); dv = state.v - after.v
            results[mode] = dict(completed=True, root_position_error_m=float(np.linalg.norm(dq[:3])),
                root_rotation_error_rad=float(np.linalg.norm(dq[3:6])),
                joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
                joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))),
                largest_within_interval_velocity_error_rad_s=worst, first_predicted_nonfoot_ground=first_bad_gap)
        except (ValueError, np.linalg.LinAlgError) as e:
            results[mode] = dict(completed=False, completed_physics_steps=index, error=repr(e))
    return dict(case=job['case'], tick=job['tick'], models=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run', 'audit', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh evidence required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence = read(a.audit); pins = dict(evidence['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed evidence ' + path)
    source = read(read(a.plan)['source']); *_, context = source_context(source['configuration']['source'])
    initialize(*context); model = _WORKER['controller'].qp.model
    meta = read(a.run / 'foot_impact_observation.json')
    materials = read(a.run / 'native_admission.json')['material_admission']['feet']
    mu = materials[0]['material']['dynamic']
    if any(m['material']['combine'] != 'average' or m['material']['dynamic'] != mu for m in materials):
        raise ValueError('Wrong native friction combination')
    with np.load(a.run / 'foot_impact_observation.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    frames = data['physics_frame']
    if not np.array_equal(frames, np.arange(288 * 160, 300 * 160 + 1)):
        raise ValueError('Complete original physics window required')
    jobs = []
    for lane, case in enumerate(meta['lanes']):
        states = []
        for i in range(len(frames)):
            sample = {k: v[i, lane] for k, v in data.items() if k != 'physics_frame'}
            s = state_from_sample(model, sample, meta['origins_world'][lane])
            states.append(dict(q=s.q, v=s.v, contacts=s.contacts, target=s.previous_command_target))
        for tick in range(288, 300):
            start = (tick - 288) * 160
            jobs.append(dict(case=case, tick=tick, states=states[start:start + 161], dt=meta['physics_dt_s'], friction=mu))
    for path in [Path(__file__), a.plan, a.audit, ROOT / 'scripts/endpoint_contact_velocity_probe_v1.py',
                 ROOT / 'scripts/foot_pair_contact_step_v1.py']:
        pins[str(path.resolve())] = sha(path)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=context) as pool:
        rows = list(pool.map(predict, jobs))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Input changed during comparison ' + path)
    result = dict(scope=__doc__, rows=rows, source_sha256=pins, native_execution=False,
        controller_changed=False, model_promoted=False, valid_expert_labels=False)
    tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); tmp.replace(a.output)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
