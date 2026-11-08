"""Replay actual commands at release, first support and nonfoot-contact boundaries.

Every one-step model prediction begins at a recorded native state. Prediction
errors after a nonfoot contact include omitted collision response and cannot
be attributed solely to actuator tracking. No native state is teleported.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

WORK = {}


def initialize(configuration, nominal):
    WORK['controller'] = PureControlTrajectoryRecoveryV9(
        PositionPDInterface(**configuration['interface']), nominal, configuration['model_reference_library'])


def evaluate(query, deadline):
    result = {k: query[k] for k in ['case', 'event', 'tick', 'nonfoot_before_interval', 'nonfoot_during_interval']}
    if time.monotonic() >= deadline:
        return dict(result, predicted=False, rejection='compute_budget')
    c = WORK['controller']
    m = c.model
    q, v = np.asarray(query['q']), np.asarray(query['v'])
    state = PublicState(q, v, pin.Quaternion(q[3:7]).matrix(), q[:3].copy(),
        q[m.q_indices].copy(), v[m.v_indices].copy(), np.asarray(query['contacts'], bool), 0.)
    state.previous_command_target = np.asarray(query['previous'])
    try:
        prediction = predict_self_contact_spheres(m, c.interface, state, np.asarray(query['target']),
            c.geometry.spheres, geometry=c.path_geometry, substeps=160)
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(result, predicted=False, rejection=repr(error))
    actual_q, actual_v = np.asarray(query['next_q']), np.asarray(query['next_v'])
    actual_rotation = pin.Quaternion(actual_q[3:7]).matrix()
    predicted = prediction['state']
    difference = pin.difference(m.model, actual_q, predicted.q)
    joint_velocity_error = predicted.v[m.v_indices] - actual_v[m.v_indices]
    return dict(result, predicted=True,
        root_position_error_m=float(np.linalg.norm(predicted.q[:3] - actual_q[:3])),
        root_attitude_error_rad=float(np.linalg.norm(difference[3:6])),
        root_velocity_error_m_s=float(np.linalg.norm(predicted.root_rotation_world @ predicted.v[:3] - actual_rotation @ actual_v[:3])),
        root_angular_velocity_error_rad_s=float(np.linalg.norm(predicted.root_rotation_world @ predicted.v[3:6] - actual_rotation @ actual_v[3:6])),
        joint_position_error_max_rad=float(np.max(abs(difference[m.v_indices]))),
        joint_velocity_error_max_rad_s=float(np.max(abs(joint_velocity_error))),
        worst_velocity_joint=m.joint_order[int(np.argmax(abs(joint_velocity_error)))],
        predicted_nonfoot_violation_m=prediction['diagnostics']['nonfoot_ground_violation_m'],
        first_predicted_nonfoot_violation=prediction['first_ground_violation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'chronology', 'configuration', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    if args.output.exists() or not 60 <= args.max_seconds <= 1200:
        raise ValueError('Fresh bounded output required')
    if read(args.run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native source required')
    cfg = read(args.configuration)
    nominal = read(args.run / 'native_command_inputs.json')['default_joint_position_rad']
    events = {r['case']: r for r in read(args.chronology)['cases']}
    with np.load(args.run / 'pure_control_commands.npz') as archive:
        data = {k: archive[k].copy() for k in archive.files}
    diagnostics = [json.loads(line) for line in (args.run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(data['tick']):
        raise ValueError('Diagnostic/action row mismatch')
    queries = []
    for lane in sorted(events):
        ids = np.flatnonzero(data['lane'] == lane)
        if len(ids) < 2:
            continue
        by_tick = {int(data['tick'][i]): (int(i), int(j)) for i, j in zip(ids[:-1], ids[1:])
                   if data['tick'][j] == data['tick'][i] + 1}
        event = events[lane]
        nonfoot_times = [event[k]['time_s'] for k in ['first_head_ground', 'first_other_ground'] if event[k] is not None]
        first_nonfoot = min(nonfoot_times, default=float('inf'))
        requested = [('release', int(data['tick'][ids[0]]))]
        first_support = [int(data['tick'][i]) for i in ids if any(diagnostics[i]['phase']['ground_contacts'])]
        if first_support:
            requested.append(('first_observed_support', first_support[0]))
        if np.isfinite(first_nonfoot):
            collision_tick = int(np.floor(first_nonfoot / .01 + 1e-8))
            requested.extend([('before_nonfoot', collision_tick - 1), ('nonfoot_interval', collision_tick)])
        for name, tick in requested:
            if tick not in by_tick:
                continue
            i, j = by_tick[tick]
            if not np.allclose(data['previous'][j], data['final'][i], rtol=0., atol=1e-6):
                raise ValueError('Actual command history mismatch')
            queries.append(dict(case=lane, event=name, tick=tick,
                nonfoot_before_interval=first_nonfoot < tick * .01,
                nonfoot_during_interval=tick * .01 <= first_nonfoot < (tick + 1) * .01,
                q=data['q'][i], v=data['v'][i], previous=data['previous'][i], target=data['final'][i],
                next_q=data['q'][j], next_v=data['v'][j], contacts=diagnostics[i]['phase']['ground_contacts']))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize,
                             initargs=(cfg, nominal)) as pool:
        rows = [future.result() for future in as_completed([pool.submit(evaluate, q, started + args.max_seconds) for q in queries])]
    rows.sort(key=lambda r: (r['case'], r['tick'], r['event']))
    keys = ['root_position_error_m', 'root_attitude_error_rad', 'root_velocity_error_m_s',
            'root_angular_velocity_error_rad_s', 'joint_position_error_max_rad', 'joint_velocity_error_max_rad_s']
    groups = []
    for event in sorted({r['event'] for r in rows}):
        subset = [r for r in rows if r['event'] == event]
        accepted = [r for r in subset if r['predicted']]
        groups.append(dict(event=event, requested=len(subset), predicted=len(accepted),
            previous_nonfoot_contacts=sum(r['nonfoot_before_interval'] for r in subset),
            errors={k: dict(zip(['median', 'p90', 'maximum'], np.quantile([r[k] for r in accepted], [.5, .9, 1.])))
                    for k in keys} if accepted else {}))
    pins = [Path(__file__), args.configuration, args.chronology, args.run / 'pure_control_commands.npz',
            args.run / 'pure_control_diagnostics.jsonl', args.run / 'job.json']
    output = dict(groups=groups, rows=rows, wall_seconds=time.monotonic() - started,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins},
        model_replay_only=True, native_rerun=False, valid_expert_labels=False,
        limitation='Model includes foot friction, joint stops and selected hip-mount self contacts; other self contacts and nonfoot impact response remain unmodeled.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(output), indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(plain(dict(groups=groups, wall_seconds=output['wall_seconds']))), flush=True)


if __name__ == '__main__':
    main()
