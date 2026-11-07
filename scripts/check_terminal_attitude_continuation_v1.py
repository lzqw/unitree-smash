"""Compare terminal attitude feedback from measured native late-recovery states.

Model branches start at the same recorded state and actual command history.
They do not reset or continue the native simulator and are not expert labels.
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
from scripts.native_terminal_support_v1 import load_verified_policy
from scripts.audit_native_terminal_support_v1 import state_at
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'bundle', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ticks', type=int, nargs='+', default=[620, 700])
    parser.add_argument('--substeps', type=int, choices=[16, 160], default=16)
    a = parser.parse_args()
    if a.output.exists() or any(t < 540 or t >= 800 for t in a.ticks):
        raise ValueError('Require new output and explicit late-recovery queries')
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result, inputs = (read(a.run/n) for n in ['sequence_native_result.json', 'INPUTS.json'])
    manifest = read(a.bundle/'manifest.json')
    sphere = read(a.sphere_audit)
    if result['case'] != 3 or result['start_tick'] != 460 or sha(a.sphere_audit) != result['sphere_audit_sha256']:
        raise ValueError('Unexpected recorded case or contact geometry')
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    with np.load(a.bundle/manifest['bundle_name'], allow_pickle=False) as z:
        query = z['case_003_query_public'].copy()
    c = build_controller(manifest['configuration'])
    base, _ = load_verified_policy(c.qp.model, c, query, Path(inputs['frictional_warm_plan']), Path(result['validation_path']))
    policy = TerminalAttitudeFeedbackV1(base)
    geometry = AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json', c.qp.model, margin_m=0.)
    protocol = read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']
    files = [Path(__file__), a.run/'sequence_native_trace.npz', a.run/'sequence_native_result.json',
        a.run/'INPUTS.json', a.bundle/'manifest.json', a.sphere_audit,
        ROOT/'configs/evaluation_protocol_v2.json', ROOT/'configs/go2_piper_authored_ground_geometry_v1.json']
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    hashes = {str(p.resolve()): sha(p) for p in files}
    records = []
    for tick in a.ticks:
        initial = state_at(c.qp.model, trace, tick-460)
        initial.previous_command_target = trace['previous_target_rad'][tick-460].copy()
        for label, active in [('original_terminal_feedback', base), ('support_kinematic_attitude_feedback', policy)]:
            state = copy.deepcopy(initial)
            states, diagnostics, failure = [], [], None
            began = time.monotonic()
            for current in range(tick, 800):
                command, _ = active.action(current-460, state, state.previous_command_target)
                predicted = predict_frictional_spheres(c.qp.model, c.qp.interface, state, command,
                    sphere['spheres'], geometry=geometry, substeps=a.substeps)
                d = predicted['diagnostics']
                bad = {k: v for k, v in d.items() if (k == 'nonfoot_ground_violation_m' and v > 1e-6)
                    or (k in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm'] and v > 3e-4)}
                diagnostics.append(dict(tick=current, **d))
                if bad:
                    failure = dict(tick=current, violations=bad, first_ground_violation=predicted['first_ground_violation'])
                    break
                state = predicted['state']
                state.previous_command_target = command.copy()
                states.append(copy.deepcopy(state))
            rpy = np.degrees(pin.rpy.matrixToRpy(state.root_rotation_world))
            window = kinematic_window(c.qp.model, states, protocol, .01)
            record = dict(query_tick=tick, policy=label, completed_model_steps=len(states), needed_steps=800-tick,
                failure=failure, model_window=window, last_roll_pitch_deg=rpy[:2].tolist(),
                last_joint_speed_rad_s=float(np.max(abs(state.joint_velocity))),
                last_kinematic_feedback=policy.last_diagnostic if active is policy else None,
                wall_seconds=time.monotonic()-began, diagnostics=diagnostics)
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ['diagnostics', 'last_kinematic_feedback']}), flush=True)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Source changed during prediction')
    report = dict(scope=__doc__, measured_case=3, integration_substeps=a.substeps,
        feedback_parameters=dict(proportional=policy.proportional, damping_s=policy.damping_s,
            maximum_joint_delta_rad=policy.maximum_joint_delta_rad,
            regularization_m=policy.regularization_m, ramp_s=policy.ramp_s),
        rows=records, source_sha256=hashes, native_continuation_executed=False,
        original_native_isolation_audit_remains_failed=True, valid_expert_labels=False)
    with a.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
