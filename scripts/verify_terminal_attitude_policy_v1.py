"""Export a complete attitude-feedback model candidate from its native query.

Uses 160 substeps and the original hardware/geometry checks. This is not an
executed native continuation, a general recovery certificate, or expert data.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_arm_margin_v1 import load_verified_policy
from scripts.audit_native_terminal_support_v1 import state_at
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'native-run', 'warm-plan', 'base-validation', 'sphere-audit', 'feedback-config', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve earlier validation')
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = read(a.bundle/'manifest.json')
    native = read(a.native_run/'sequence_native_result.json')
    if native['case'] != 3 or native['start_tick'] != 460 or native['teacher_model'] != 'planned_prefix_terminal_arm_margin_v1':
        raise ValueError('Require the measured arm-margin query and unchanged first episode')
    if sha(a.sphere_audit) != native['sphere_audit_sha256'] or sha(a.base_validation) != native['validation_sha256']:
        raise ValueError('Different source model or base policy')
    with np.load(a.native_run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    with np.load(a.bundle/manifest['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    c = build_controller(manifest['configuration'])
    base, base_record = load_verified_policy(c.qp.model, c, query, a.warm_plan, a.base_validation)
    parameters = read(a.feedback_config)
    policy = TerminalAttitudeFeedbackV1(base, **parameters)
    initial = state_at(c.qp.model, trace, 0)
    initial.previous_command_target = trace['previous_target_rad'][0].copy()
    if (np.max(abs(pin.difference(c.qp.model.model, np.asarray(base_record['initial_q']), initial.q))) > 2e-5
            or np.max(abs(np.asarray(base_record['initial_v'])-initial.v)) > 2e-4):
        raise ValueError('Initial native query differs from the base model query')
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry = AuthoredGroundClearance(geometry_path, c.qp.model, margin_m=0.)
    spheres = read(a.sphere_audit)['spheres']
    protocol = read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']
    files = [Path(__file__), a.bundle/'manifest.json', a.native_run/'sequence_native_trace.npz',
        a.native_run/'sequence_native_result.json', a.warm_plan, a.base_validation, a.sphere_audit,
        a.feedback_config, geometry_path, ROOT/'configs/evaluation_protocol_v2.json', c.qp.model.path]
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    hashes = {str(p.resolve()): sha(p) for p in files}
    state = copy.deepcopy(initial)
    states, commands, records, failure = [], [], [], None
    began = time.monotonic()
    for offset in range(340):
        command, mode = policy.action(offset, state, state.previous_command_target)
        prediction = predict_frictional_spheres(c.qp.model, c.qp.interface, state, command,
                                               spheres, geometry=geometry, substeps=160)
        d = prediction['diagnostics']
        bad = {k: v for k, v in d.items() if (k == 'nonfoot_ground_violation_m' and v > 1e-6)
               or (k in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm'] and v > 3e-4)}
        records.append(dict(offset=offset, mode=mode, diagnostics=d, attitude=copy.deepcopy(policy.last_diagnostic)))
        if bad:
            failure = dict(offset=offset, violations=bad, first_ground_violation=prediction['first_ground_violation'])
            break
        state = prediction['state']
        state.previous_command_target = command.copy()
        states.append(copy.deepcopy(state)); commands.append(command.copy())
    window = kinematic_window(c.qp.model, states, protocol, .01)
    final_rpy = np.degrees(pin.rpy.matrixToRpy(state.root_rotation_world))
    terminal = dict(roll_pitch_deg=final_rpy[:2].tolist(),
        attitude_margin_deg=protocol['maximum_abs_roll_pitch_deg']-float(np.max(abs(final_rpy[:2]))),
        horizontal_speed_m_s=float(np.linalg.norm((state.root_rotation_world @ state.v[:3])[:2])),
        angular_speed_rad_s=float(np.linalg.norm(state.v[3:6])),
        joint_speed_rad_s=float(np.max(abs(state.joint_velocity))), support_count=int(state.contacts.sum()))
    terminal_ok = (terminal['attitude_margin_deg'] >= 0 and terminal['support_count'] > 0
        and terminal['horizontal_speed_m_s'] <= protocol['maximum_horizontal_root_speed_m_s']
        and terminal['angular_speed_rad_s'] <= protocol['maximum_root_angular_speed_rad_s']
        and terminal['joint_speed_rad_s'] <= protocol['maximum_abs_joint_velocity_rad_s'])
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Input changed during validation')
    output = dict(scope=__doc__, case=3, start_tick=460, needed_steps=340,
        executed_model_steps=len(states), model_candidate_accepted=failure is None and len(states) == 340
        and window['longest_model_kinematic_window_s'] >= .8 and terminal_ok,
        failure=failure, model_window=window, terminal=terminal,
        base_validation_path=str(a.base_validation.resolve()), base_validation_sha256=sha(a.base_validation),
        feedback_config_path=str(a.feedback_config.resolve()), feedback_config_sha256=sha(a.feedback_config),
        feedback_parameters=parameters, integration_substeps=160,
        initial_q=initial.q, initial_v=initial.v, initial_contacts=initial.contacts,
        actual_previous_command=initial.previous_command_target,
        selected_tail=base_record['selected_tail'],
        states=[dict(q=s.q, v=s.v, contacts=s.contacts) for s in states], commands=commands, records=records,
        source_sha256=hashes, wall_seconds=time.monotonic()-began,
        measured_native_initial_state=True, native_continuation_executed=False, valid_expert_labels=False)
    with a.output.open('x') as stream:
        stream.write(json.dumps(plain(output), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: output[k] for k in ['model_candidate_accepted', 'executed_model_steps', 'failure', 'model_window', 'terminal']}))


if __name__ == '__main__':
    main()
