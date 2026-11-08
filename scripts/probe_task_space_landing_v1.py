"""Construct landing commands in foot space, then check the full recovery.

Offline candidate search on four measured states. IK is only an initializer:
every candidate still faces the original coupled dynamics and path/terminal
constraints. No native state is replaced and no model result is an expert label.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
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
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.audit_joint_sequence_proposal_v1 import plain


def foot_knots(model, interface, state, original, front_down, rear_down, front_forward):
    """IK modifies only early leg targets, never the measured initial state."""
    knots = np.asarray(original).copy()
    if front_down == rear_down == front_forward == 0:
        return knots, [0., 0., 0.]
    yaw = np.arctan2(state.root_rotation_world[1, 0], state.root_rotation_world[0, 0])
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
    errors = []
    for index in range(3):
        q = state.q.copy()
        q[model.q_indices[:12]] = knots[index, :12]
        q[model.q_indices[12:]] = np.clip(state.joint_position[12:] + knots[index, 12:],
            interface.limits[12:, 0], interface.limits[12:, 1])
        snap = model.snapshot(q, np.zeros(24))
        feet_indices = [model.names.index(n) for n in model.feet]
        goal = snap['body_positions_world'][feet_indices].copy()
        goal[:2] += front_forward * forward
        goal[:2, 2] -= front_down
        goal[2:, 2] -= rear_down
        for _ in range(40):
            snap = model.snapshot(q, np.zeros(24))
            error = (goal - snap['body_positions_world'][feet_indices]).reshape(12)
            if np.max(np.abs(error)) < 1e-5:
                break
            jac = snap['foot_jacobian'][:, :, model.v_indices[:12]].reshape(12, 12)
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .003**2 * np.eye(12), error)
            q[model.q_indices[:12]] = np.clip(q[model.q_indices[:12]] + np.clip(delta, -.12, .12),
                interface.limits[:12, 0], interface.limits[:12, 1])
        snap = model.snapshot(q, np.zeros(24))
        errors.append(float(np.max(np.linalg.norm(goal - snap['body_positions_world'][feet_indices], axis=1))))
        knots[index, :12] = q[model.q_indices[:12]]
    return knots, errors


def evaluate(job):
    initialize(*job['context'])
    w = _WORKER; controller = w['controller']; state = state_from_row(job['query'])
    knobs = job['placement']
    knots, errors = foot_knots(controller.qp.model, controller.qp.interface, state,
        job['original_knots'], **knobs)
    row = dict(case=job['case'], placement=knobs, ik_max_foot_errors_m=errors,
        knots_rad=knots, native_execution=False, valid_expert_labels=False)
    # An unreachable foot request is not silently accepted as successful IK.
    if max(errors) > .005:
        return dict(row, ik_rejected=True, result=None, independently_verified=False)
    def rollout():
        base = w['factory'](controller.qp.model, controller.qp.interface, state, knots)
        policy = TerminalAttitudeFeedbackV1(base, proportional=.3, damping_s=.04, maximum_joint_delta_rad=.08)
        first, _ = policy.action(80, state, state.previous_command_target)
        if not np.array_equal(first, state.previous_command_target):
            raise ValueError('Changed first actual command at handoff')
        geometry = HipMountPathConstraintV1(w['geometry'], job['geometry'], margin_rad=.02)
        result = evaluate_suffix(controller, state, policy, w['spheres'], geometry, w['protocol'],
            control_tick=280, substeps=160, deadline=job['deadline'], save=True)
        for key, value in [('initial_q', state.q), ('initial_v', state.v),
                           ('initial_previous_command', state.previous_command_target),
                           ('initial_contacts', state.contacts)]:
            if not np.array_equal(result[key], value):
                raise ValueError('Initial state/history replaced')
        return result
    result = rollout(); verified = False
    if result['accepted']:
        replay = rollout()
        verified = replay['accepted'] and len(replay['trajectory']) == len(result['trajectory']) == 520 and all(
            np.array_equal(a[k], b[k]) for a, b in zip(replay['trajectory'], result['trajectory'])
            for k in ['q', 'v', 'target', 'contacts'])
    return dict(row, ik_rejected=False, result=result, independently_verified=bool(verified))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name, value):
        path = args.output/name; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n'); temp.replace(path)
    try:
        plan = read(args.plan); audit = read(Path(plan['bank_followup'])/'query_bank_audit.json')
        if not audit['passed']:
            raise ValueError('Verified passive native bank required')
        pins = dict(plan['source_sha256']); pins.update(audit['source_sha256'])
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError('Changed source ' + path)
        source = read(plan['source'])
        _, original, _, geometry, context = source_context(source['configuration']['source'])
        initialize(*context); model = _WORKER['controller'].qp.model
        meta = read(Path(plan['bank_run'])/'recovery_query_bank.json')
        with np.load(Path(plan['bank_run'])/'recovery_query_bank.npz', allow_pickle=False) as archive:
            z = {k: archive[k] for k in archive.files}
        placements = [dict(front_down=0., rear_down=0., front_forward=0.)]
        placements += [dict(front_down=f, rear_down=r, front_forward=x)
            for f, r, x in itertools.product([.025, .05, .08], [0., .04], [-.03, .03])]
        jobs = []; diagnostics = []
        for choice in read(plan['selection'])['selected']:
            lane = choice['case']; samples = []
            state = None
            for tick in [280, 295, 300, 305, 310]:
                index = int(np.flatnonzero(z['control_ticks'] == tick)[0])
                sample = {k: z[k][index, lane].copy() for k in z if k != 'control_ticks'}
                measured = state_from_sample(model, sample, meta['origins_world'][lane])
                snap = model.snapshot(measured.q, measured.v)
                samples.append(dict(tick=tick, root_height_m=measured.q[2],
                    forward_axis_vertical=measured.root_rotation_world[2, 0],
                    root_velocity_world=measured.root_rotation_world @ measured.v[:3],
                    omega_world=measured.root_rotation_world @ measured.v[3:6],
                    foot_positions_world=snap['body_positions_world'][[model.names.index(n) for n in model.feet]],
                    foot_force_world=sample['foot_force_world']))
                if tick == 280:
                    if not sample['unfinished_first_episode'] or sample['episode_index'] != 0 or sample['phase'][3] <= .5:
                        raise ValueError('Actual first-episode post-hit flight required')
                    state = measured
            diagnostics.append(dict(case=lane, actual_original_student_samples=samples))
            aligned, _ = transfer_reference(context, state)
            query = dict(q=state.q, v=state.v, contacts=state.contacts, target=state.previous_command_target)
            for placement in placements:
                jobs.append(dict(case=lane, context=aligned, query=query, original_knots=original,
                    geometry=geometry, placement=placement))
        for name, module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module, '__file__', None):
                path = str(Path(module.__file__).resolve()); pins[path] = sha(path)
        pins[str(Path(__file__).resolve())] = sha(__file__)
        write('measured_landing_geometry.json', dict(rows=diagnostics, source_sha256=pins))
        deadline = time.monotonic() + 600.
        for job in jobs: job['deadline'] = deadline
        write('status.json', dict(state='MODEL_SEARCH', candidates=len(jobs), workers=4,
            maximum_seconds=600, native_execution=False))
        rows = []
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
            futures = {pool.submit(evaluate, job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                row = future.result(); index = futures[future]; write(f'candidate_{index:03d}.json', row)
                result = row['result']
                compact = {k: v for k, v in row.items() if k not in ['result', 'knots_rad']}
                compact['result'] = {k: v for k, v in result.items() if k not in
                    ['trajectory', 'initial_q', 'initial_v', 'initial_contacts', 'initial_previous_command']} if result else None
                compact['file'] = f'candidate_{index:03d}.json'; rows.append(compact)
                print(json.dumps(plain(dict(case=row['case'], placement=row['placement'],
                    ik_rejected=row['ik_rejected'], completed=result['completed_model_steps'] if result else None))), flush=True)
        for path, expected in pins.items():
            if sha(path) != expected: raise ValueError('Source changed during experiment ' + path)
        write('result.json', dict(scope=__doc__, rows=rows, source_sha256=pins,
            formal_scoring_changed=False, native_execution=False, valid_expert_labels=False))
        write('status.json', dict(state='COMPLETED_MODEL_SEARCH', candidates=len(rows),
            ik_rejected=sum(r['ik_rejected'] for r in rows),
            verified=sum(r['independently_verified'] for r in rows), native_execution=False))
    except BaseException as error:
        write('status.json', dict(state='FAILED', error=repr(error))); raise


if __name__ == '__main__': main()
