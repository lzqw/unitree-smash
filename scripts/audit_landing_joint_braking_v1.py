"""Replay a recorded model limit failure and inspect bounded braking requests.

One control interval is diagnostic only. It does not establish recoverability,
native execution, a safe fallback, or expert labels. No physical gains change.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_landing_reference_tracking_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.audit_joint_sequence_proposal_v1 import plain
import scripts.predict_frictional_spheres_v1 as predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(args.source)
    row = next(r for r in source['rows'] if r['probe']['name'] == 'orientation_world_1_+1')
    failure = row['result']['failure']; step = failure['step']
    if failure['reason'] != 'path_constraints' or 'joint_position_violation_rad' not in failure['violations']:
        raise ValueError('Expected independently recorded joint limit failure')
    fine_paths = [Path(p) for p in source['source_sha256'] if Path(p).name == 'refine_support_tail_fine160_v1.json']
    if len(fine_paths) != 1:
        raise ValueError('Unique original nominal source required')
    fine = read(fine_paths[0]); tail = read(fine['configuration']['source'])
    parent = read(tail['configuration']['source_search']); cfg = parent['configuration']
    query = read(cfg['query']); manifest = read(Path(cfg['bundle']) / 'manifest.json')
    nominal = fine['independent_replay']['trajectory']
    rq = np.asarray([query['initial_q']] + [r['q'] for r in nominal[:-1]])
    rv = np.asarray([query['initial_v']] + [r['v'] for r in nominal[:-1]])
    initialize(manifest['configuration'], query, fine['fixed_early_knots_rad'], cfg['sphere_audit'],
               rq, rv, source['configuration'])
    w = _WORKER; model = w['controller'].qp.model; interface = w['controller'].qp.interface
    initial = perturb_state(w['initial'], row['probe'])
    base = w['factory'](model, interface, initial, fine['best_knots_rad'])
    policy = TerminalAttitudeFeedbackV1(base, proportional=.3, damping_s=.04, maximum_joint_delta_rad=.08)
    saved = row['result']['trajectory'][-1]
    if len(row['result']['trajectory']) != step:
        raise ValueError('Saved states must end immediately before failure')
    state = copy.deepcopy(initial)
    state.q = np.asarray(saved['q']); state.v = np.asarray(saved['v'])
    state.root_rotation_world = pin.Quaternion(state.q[3:7]).matrix()
    state.root_position_world = state.q[:3].copy()
    state.joint_position = state.q[model.q_indices].copy()
    state.joint_velocity = state.v[model.v_indices].copy()
    state.contacts = np.asarray(saved['contacts'], bool)
    state.previous_command_target = np.asarray(saved['target'])
    target, _ = policy.action(80 + step, state, state.previous_command_target)
    j = model.joint_order.index('FL_calf_joint')
    hashes = dict(source['source_sha256'])
    for p in [args.source, Path(__file__)]:
        hashes[str(p.resolve())] = sha(p)
    for p, digest in hashes.items():
        if sha(p) != digest: raise ValueError('Changed source ' + p)
    def predict(command):
        return predictor.predict_frictional_spheres(model, interface, state, command,
                 w['spheres'], geometry=w['geometry'], substeps=160)
    uncaptured = predict(target)
    error = abs(uncaptured['diagnostics']['joint_position_violation_rad'] - failure['violations']['joint_position_violation_rad'])
    if error > 1e-12: raise ValueError('Original failing interval did not reproduce')
    original_step = predictor.frictional_velocity_step
    rows = []
    commands = [('original', target.copy())]
    for delta in [.02, .05, .10]:
        command = target.copy(); command[j] = max(interface.limits[j, 0], command[j] - delta)
        commands.append(('lower_calf_target_' + str(delta), command))
    command = target.copy(); command[j] = interface.limits[j, 0]
    commands.append(('calf_target_at_lower_limit_diagnostic', command))
    for label, command in commands:
        traces = []
        def traced_step(*a, **kw):
            result = original_step(*a, **kw)
            traces.append(dict(torque=result['torque'].copy(), velocity=result['velocity'].copy(),
                               impulse=result['impulse'].copy()))
            return result
        predictor.frictional_velocity_step = traced_step
        try:
            prediction = predict(command)
        finally:
            predictor.frictional_velocity_step = original_step
        if label == 'original':
            for k in ['q', 'v']:
                if not np.array_equal(getattr(prediction['state'], k), getattr(uncaptured['state'], k)):
                    raise ValueError('Diagnostic recording changed the model result')
        torque = np.asarray([r['torque'] for r in traces])
        velocities = np.asarray([r['velocity'] for r in traces])[:, model.v_indices]
        normal_force = np.asarray([r['impulse'][2::3] for r in traces]) / (.01 / 160)
        rows.append(dict(label=label, calf_target_rad=float(command[j]),
            explicit_initial_pd_estimate_nm=float(np.clip(interface.kp[j]*(command[j]-state.joint_position[j])
                - interface.kd[j]*state.joint_velocity[j], -interface.effort[j], interface.effort[j])),
            calf_implicit_torque_min_nm=float(torque[:, j].min()),
            calf_implicit_torque_max_nm=float(torque[:, j].max()),
            calf_velocity_first_rad_s=float(velocities[0, j]),
            calf_velocity_last_rad_s=float(velocities[-1, j]),
            calf_position_last_rad=float(prediction['state'].joint_position[j]),
            mean_normal_force_per_foot_n=normal_force.mean(0),
            diagnostics=prediction['diagnostics'],first_ground_violation=prediction['first_ground_violation']))
    for p, digest in hashes.items():
        if sha(p) != digest: raise ValueError('Source changed during diagnostic ' + p)
    out = dict(scope=__doc__, source_sha256=hashes, probe=row['probe'], failed_model_control_step=step,
        original_failure=failure, failure_reproduction_error_rad=error, recording_changes_model_result=False,
        joint_name=model.joint_order[j], joint_position_rad=float(state.joint_position[j]),
        joint_velocity_rad_s=float(state.joint_velocity[j]), joint_limits_rad=interface.limits[j],
        original_target_rad=float(target[j]), previous_target_rad=float(state.previous_command_target[j]),
        original_feedback=dict(tracking=base.last_diagnostic,attitude=policy.last_diagnostic),
        interval_s=.01, substeps=160, rows=rows, native_execution=False,valid_expert_labels=False,
        limitations=['Force and implicit torque are model outputs, not native telemetry.',
                     'Avoiding one interval violation is not stable recovery.',
                     'Lower-limit request is a diagnostic bound, never deployed as a fallback.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(out), indent=2, allow_nan=False)+'\n')
    print(json.dumps(plain(dict(joint=out['joint_name'],position=out['joint_position_rad'],
        velocity=out['joint_velocity_rad_s'], original_target=out['original_target_rad'], rows=rows))), flush=True)


if __name__ == '__main__':
    main()
