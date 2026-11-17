"""Explicit offline adapter: impulse-model diagnostics to original force gates.

Not PhysX equivalence or a native reference certificate. Contact-law residuals
retain their original solver units; only impulse balance/cone quantities are
converted to force units. Authored geometry is checked at every microstep.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.diagnostic_nonfoot_spheres_v1 import nonfoot_spheres
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1

ROOT = Path(__file__).resolve().parents[1]


def admission_diagnostics(result, dt):
    if not np.isfinite(dt) or dt <= 0 or result.get('ground_geometry_checked') is not True:
        raise ValueError('Explicit microstep and checked ground geometry required')
    d = dict(result['diagnostics'])
    required = ('momentum_residual', 'normal_impulse_violation_ns', 'friction_cone',
        'normal_complementarity', 'tangent_fixed_point', 'speed_constraint_residual',
        'joint_velocity_violation_rad_s', 'joint_position_violation_rad',
        'motor_torque_violation_nm', 'nonfoot_ground_violation_m', 'nonfoot_contact_substeps')
    for key in required:
        if key not in d or not np.isfinite(d[key]) or d[key] < 0:
            raise ValueError('Missing or invalid physical diagnostic: ' + key)
    for key in ('normal_complementarity', 'tangent_fixed_point', 'speed_constraint_residual'):
        if d[key] > 2e-7:
            raise ValueError('Original coupled solver contract failed: ' + key)
    d['dynamics_residual_max'] = d['momentum_residual'] / dt
    d['normal_force_violation_n'] = d['normal_impulse_violation_ns'] / dt
    d['friction_violation_n'] = d['friction_cone'] / dt
    return d


class ExperimentalSphereAdmissionV1:
    def __init__(self, model, geometry, sphere_manifest, *, iterations=2000):
        record = json.loads(Path(sphere_manifest).read_text())
        protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
        asset = ROOT/protocol['asset']['path']
        if hashlib.sha256(asset.read_bytes()).hexdigest() != record['asset_sha256'] or record['asset_sha256'] != protocol['asset']['sha256']:
            raise ValueError('Sphere asset drift')
        for path, digest in record['dependency_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('Sphere dependency drift')
        if type(iterations) is not int or not 1 <= iterations <= 4000:
            raise ValueError('Bounded iterations required')
        self.spheres = record['spheres']
        if [s['body'] for s in self.spheres] != model.feet:
            raise ValueError('Foot order mismatch')
        self.geometry = geometry
        self.extras = nonfoot_spheres(geometry, model.feet)
        self.hip = HipMountCollisionResponseV1(model)
        self.iterations = iterations

    def __call__(self, model, interface, state, target, *, duration_s, substeps, enforce_joint_stops):
        if duration_s != .01 or enforce_joint_stops is not True:
            raise ValueError('Original control period and joint stops required')
        result = predict_speed_limited_spheres(model, interface, state, target, self.spheres,
            substeps=substeps, geometry=self.geometry, enforce_joint_stops=True,
            diagnostic_nonfoot_spheres=self.extras, diagnostic_hip_mount_response=self.hip,
            contact_max_iterations=self.iterations)
        result['diagnostics'] = admission_diagnostics(result, duration_s/substeps)
        return result
