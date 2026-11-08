"""Propagate actual recovery commands to episode end with foot-pair models.

Starts once from a measured native state; no intermediate state correction.
This is model diagnosis using a recorded command tape, not a new controller.
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.foot_pair_contact_step_v1 import foot_pair_velocity_step
from scripts.endpoint_contact_velocity_probe_v1 import endpoint_velocity_probe
from scripts.feedback_tail_rollout_v1 import terminal_measures, terminal_violations, kinematically_stable
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    w = _WORKER; controller = w['controller']; model, interface = controller.qp.model, controller.qp.interface
    trace = job['trace']; start = job['tick'] - 280
    state = state_from_row(dict(q=trace['q'][start], v=trace['v'][start],
        contacts=trace['contacts'][start], target=trace['previous_target_rad'][start]))
    geometry = HipMountPathConstraintV1(w['geometry'], job['geometry'], margin_rad=.02)
    function = foot_pair_velocity_step if job['mode'] == 'pair_step' else endpoint_velocity_probe
    rows = []; failure = None; longest = trailing = 0; terminal = None
    began = time.monotonic()
    for tick in range(job['tick'], 800):
        if time.monotonic() >= job['deadline']:
            failure = dict(tick=tick, reason='declared_compute_budget'); break
        target = trace['final_target_rad'][tick - 280]
        if (np.any(target < interface.limits[:, 0] - 1e-6) or np.any(target > interface.limits[:, 1] + 1e-6)
                or np.max(abs(target[12:] - state.previous_command_target[12:])) > .100001):
            failure = dict(tick=tick, reason='recorded_command_interface'); break
        try:
            for micro in range(160):
                r = function(model, interface, state, target, w['spheres'], .0000625, foot_friction=job['friction'])
                q = r.get('q')
                if q is None:
                    q = pin.integrate(model.model, state.q, .0000625 * r['velocity'])
                state.q = q; state.v = r['velocity']
                state.contacts = r['ground_impulse'][:, 2] > 1e-8
                gap, _ = geometry.evaluate(q)
                if gap.min() < -1e-6:
                    index = int(np.argmin(gap))
                    failure = dict(tick=tick, micro=micro, reason='model_nonfoot_ground',
                        body=geometry.shapes[index]['body'], gap_m=float(gap[index])); break
                if max(np.max(interface.limits[:, 0] - q[model.q_indices]),
                        np.max(q[model.q_indices] - interface.limits[:, 1]),
                        np.max(abs(state.v[model.v_indices]) - interface.velocity)) > 3e-4:
                    failure = dict(tick=tick, micro=micro, reason='model_joint_limits'); break
            if failure:
                break
        except (ValueError, np.linalg.LinAlgError) as e:
            failure = dict(tick=tick, micro=micro, reason='model_solver_or_geometry', error=repr(e)); break
        state.joint_position = state.q[model.q_indices].copy(); state.joint_velocity = state.v[model.v_indices].copy()
        state.root_position_world = state.q[:3].copy(); state.root_rotation_world = pin.Quaternion(state.q[3:7]).matrix()
        state.previous_command_target = target.copy()
        terminal = terminal_measures(controller, state)
        trailing = trailing + 1 if state.q[2] >= .12 and kinematically_stable(terminal, w['protocol']) else 0
        longest = max(longest, trailing)
        row = dict(tick=tick, q=state.q.copy(), v=state.v.copy(), contacts=state.contacts.copy())
        if tick < 799:
            dq = pin.difference(model.model, trace['q'][tick + 1 - 280], state.q)
            dv = state.v - trace['v'][tick + 1 - 280]
            row.update(root_position_error_m=float(np.linalg.norm(dq[:3])),
                joint_position_error_rad=float(np.max(abs(dq[model.v_indices]))),
                joint_velocity_error_rad_s=float(np.max(abs(dv[model.v_indices]))))
        rows.append(row)
    violations = terminal_violations(terminal, .01) if terminal else dict(no_endpoint=True)
    return dict(mode=job['mode'], start_tick=job['tick'], required_steps=800 - job['tick'],
        completed_steps=len(rows), failure=failure, terminal=terminal, terminal_violations=violations,
        trailing_stable_s=trailing * .01, longest_stable_s=longest * .01,
        model_suffix_passed=failure is None and not violations and trailing >= 80 and state.q[2] >= .12,
        seconds=time.monotonic() - began, trajectory=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'run', 'evidence', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output directory required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(read(a.plan)['source']); *_, geometry, context = source_context(source['configuration']['source'])
    evidence = read(a.evidence); pins = dict(evidence['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed evidence ' + path)
    with np.load(a.run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    ticks = [r['tick'] for r in read(a.run / 'sequence_native_result.json')['records']]
    if ticks != list(range(280, 800)):
        raise ValueError('Full actual command tape required')
    material = read(a.run / 'native_admission.json')['material_admission']['feet']
    friction = material[0]['material']['dynamic']
    if any(m['material']['dynamic'] != friction or m['material']['combine'] != 'average' for m in material):
        raise ValueError('Original native material combination required')
    for path in [a.plan, a.evidence, Path(__file__), a.run / 'sequence_native_trace.npz',
                 a.run / 'sequence_native_result.json', a.run / 'native_admission.json']:
        pins[str(path.resolve())] = sha(path)
    a.output.mkdir(parents=True)
    def write(name, obj):
        path = a.output / name; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(obj), indent=2, allow_nan=False) + '\n'); tmp.replace(path)
    write('inputs.json', dict(source_sha256=pins, max_seconds=600, workers=2))
    deadline = time.monotonic() + 600
    jobs = [dict(mode=mode, tick=tick, trace=trace, geometry=geometry, friction=friction, deadline=deadline)
        for tick in [280, 400] for mode in ['pair_step', 'endpoint_velocity']]
    write('status.json', dict(state='RUNNING'))
    try:
        rows = []
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=context) as pool:
            for row in pool.map(evaluate, jobs):
                rows.append(row); write(f"{row['mode']}_{row['start_tick']}.json", row)
                print(json.dumps({k: v for k, v in row.items() if k not in ['trajectory', 'terminal']}), flush=True)
        for path, digest in pins.items():
            if sha(path) != digest:
                raise ValueError('Input changed during full suffix prediction ' + path)
        write('result.json', dict(scope=__doc__, rows=rows, source_sha256=pins,
            native_controller_changed=False, model_promoted=False, valid_expert_labels=False))
        write('status.json', dict(state='COMPLETED'))
    except BaseException as e:
        write('status.json', dict(state='FAILED', error=repr(e))); raise


if __name__ == '__main__':
    main()
