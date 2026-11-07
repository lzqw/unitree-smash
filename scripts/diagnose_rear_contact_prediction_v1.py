"""Offline contact-set sensitivity on actual dispatched commands.

Remove RL-foot contact only in a counterfactual predictor. This is not a new
physical model, an admissible teacher, or an online contact selection rule.
Contact-label agreement is deliberately not scored: removing a contact makes
that comparison tautological. Inspect changes in measured velocity errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts import predict_frictional_spheres_v1 as prediction


def exclude_contact_solver(original, excluded):
    """Re-solve full coupled dynamics with a smaller contact Jacobian."""
    def solve(mass, velocity, bias, selection, kp, kd, error, effort,
              dt, jacobian, normal_target, mu, *, warm=None, **kwargs):
        count = len(normal_target)
        keep = np.asarray([i for i in range(count) if i != excluded])
        rows = (3 * keep[:, None] + np.arange(3)).ravel()
        result = original(mass, velocity, bias, selection, kp, kd, error,
                          effort, dt, jacobian[rows], normal_target[keep], mu,
                          warm=None if warm is None else warm[rows], **kwargs)
        expanded = np.zeros(3 * count)
        expanded[rows] = result['impulse']
        return dict(result, impulse=expanded)
    return solve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['run', 'bundle', 'sphere-audit', 'one-step-audit', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    audit = json.loads(a.one_step_audit.read_text())
    for name, expected in audit['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Changed baseline input: ' + name)
    manifest = json.loads((a.bundle / 'manifest.json').read_text())
    c = build_controller(manifest['configuration'])
    model = c.qp.model
    geometry = AuthoredGroundClearance(
        ROOT / 'configs/go2_piper_authored_ground_geometry_v1.json', model, margin_m=0.)
    spheres = json.loads(a.sphere_audit.read_text())['spheres']
    if not model.feet[2].startswith('RL'):
        raise ValueError('Unexpected foot order')
    with np.load(a.run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {name: z[name].copy() for name in z.files}
    result = json.loads((a.run / 'sequence_native_result.json').read_text())
    reference = next(g for g in audit['groups'] if g['substeps'] == 160)
    files = [Path(__file__), a.one_step_audit, a.bundle / 'manifest.json', a.sphere_audit]
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    hashes = dict(audit['source_sha256'])
    hashes.update({str(f.resolve()): digest(f) for f in files})
    original = prediction.frictional_velocity_step
    rows = []
    for row in reference['rows']:
        index = row['tick'] - result['start_tick']
        state = state_at(model, trace, index)
        actual = state_at(model, trace, index + 1)
        prediction.frictional_velocity_step = exclude_contact_solver(original, 2)
        try:
            predicted = prediction.predict_frictional_spheres(
                model, c.qp.interface, state, trace['final_target_rad'][index],
                spheres, substeps=160, geometry=geometry)
        finally:
            prediction.frictional_velocity_step = original
        following = predicted['state']
        rot = pin.Quaternion(following.q[3:7]).matrix()
        native_rot = pin.Quaternion(actual.q[3:7]).matrix()
        rows.append(dict(tick=row['tick'],
            baseline_joint_velocity_error_rad_s=row['joint_velocity_error_rad_s'],
            no_rl_contact_joint_velocity_error_rad_s=(following.v[model.v_indices] - actual.v[model.v_indices]).tolist(),
            root_linear_velocity_world_error_m_s=(rot @ following.v[:3] - native_rot @ actual.v[:3]).tolist(),
            root_angular_velocity_world_error_rad_s=(rot @ following.v[3:6] - native_rot @ actual.v[3:6]).tolist(),
            native_next_rl_normal_force_n=row['native_next_force_world_n'][2][2],
            baseline_next_rl_normal_force_n=row['predicted_last_normal_force_n'][2],
            counterfactual_path_diagnostics=predicted['diagnostics']))
    summaries = {}
    for mode, key in [('baseline', 'baseline_joint_velocity_error_rad_s'),
                      ('no_rl_contact', 'no_rl_contact_joint_velocity_error_rad_s')]:
        errors = np.asarray([r[key] for r in rows])
        summaries[mode] = dict(
            joint_velocity_rmse_rad_s=dict(zip(model.joint_order, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
            maximum_joint_velocity_error_rad_s=float(np.max(abs(errors))))
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError('Input changed during diagnostic: ' + name)
    output = dict(scope=__doc__,case=result['case'],rows=rows,summary=summaries,
                  source_sha256=hashes,substeps=160,native_execution=False,
                  controller_changed=False,valid_teacher=False,valid_expert_labels=False)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
