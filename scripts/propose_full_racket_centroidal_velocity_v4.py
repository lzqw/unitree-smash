#!/usr/bin/env python3
"""Offline velocity proposals with floating-base counterreaction at fixed pose.

This conserves instantaneous centroidal momentum when changing arm velocity.
It is not a trajectory, actuator tracking result, or executed policy score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4

BODY_NAMES = ['base', 'FL_hip', 'FR_hip', 'Head_upper', 'RL_hip', 'RR_hip', 'MountPlate',
              'FL_thigh', 'FR_thigh', 'Head_lower', 'RL_thigh', 'RR_thigh', 'base_link',
              'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'link1', 'FL_foot', 'FR_foot',
              'RL_foot', 'RR_foot', 'link2', 'link3', 'link4', 'link5', 'link6', 'Racket']


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def momentum_maps(mass, position, inertia, linear, angular):
    """Return centroid and base response to relative joint-rate changes.

    `linear`/`angular` contain body-COM Jacobians with the base held fixed.
    Angular response B satisfies delta_base_omega = B @ delta_joint_rate.
    Linear COM momentum and angular momentum about the centroid are fixed.
    """
    mass, position, inertia, linear, angular = [np.asarray(x) for x in
                                               [mass, position, inertia, linear, angular]]
    if np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError('Positive total mass and nonnegative body masses required')
    center = (mass[:, None] * position).sum(0) / mass.sum()
    relative = position - center
    center_jacobian = (mass[:, None, None] * linear).sum(0) / mass.sum()
    locked = np.zeros((3, 3))
    joint_momentum = np.zeros((3, linear.shape[-1]))
    for m, r, tensor, jp, jw in zip(mass, relative, inertia, linear, angular):
        locked += tensor + m * ((r @ r) * np.eye(3) - np.outer(r, r))
        joint_momentum += tensor @ jw + m * skew(r) @ jp
    if np.linalg.eigvalsh(locked).min() <= 0:
        raise ValueError('Locked inertia must be positive definite')
    base_response = -np.linalg.solve(locked, joint_momentum)
    return center, center_jacobian, base_response, locked


def event_geometry(event, inertials, model):
    poses = np.asarray(event['body_link_poses_local'])
    assert poses.shape == (28, 7)
    root_rotation = rotation(poses[0, 3:])
    root_position = poses[0, :3]
    q = np.asarray(event['joint_position_rad'][-6:])
    p = root_position + root_rotation @ model.arm_origin
    r = root_rotation.copy()
    origins, axes = [], []
    for i, (offset, turn) in enumerate(model.joints):
        p = p + r @ offset
        r = r @ turn
        origins.append(p.copy())
        axes.append(r[:, 2].copy())
        r = r @ Rotation.from_rotvec([0., 0., q[i]]).as_matrix()
    reconstruction_error = float(np.linalg.norm(p - event['flange_position_m']))
    # Native world coordinates were float32 before subtracting env origins.
    # This50um offline consistency guard does not change a contact threshold.
    assert reconstruction_error < 5e-5, reconstruction_error
    origins, axes = np.asarray(origins), np.asarray(axes)
    mass, centers, tensors, linear, angular = [], [], [], [], []
    for index, name in enumerate(BODY_NAMES):
        body = inertials[name]
        rb = rotation(poses[index, 3:])
        center = poses[index, :3] + rb @ np.asarray(body['com_body_m'])
        principal = rb @ rotation(body['principal_axes_wxyz'])
        tensor = principal @ np.diag(body['principal_inertia_kg_m2']) @ principal.T
        count = 6 if name == 'Racket' else int(name[-1]) if name.startswith('link') and name[-1].isdigit() else 0
        jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
        if count:
            jp[:, :count] = np.cross(axes[:count], center - origins[:count]).T
            jw[:, :count] = axes[:count].T
        mass.append(body['mass_kg'])
        centers.append(center)
        tensors.append(tensor)
        linear.append(jp)
        angular.append(jw)
    hit, normal = np.asarray(event['point_m']), np.asarray(event['face_normal'])
    point_jacobian = np.cross(axes, hit - origins).T
    normal_jacobian = np.cross(axes, np.broadcast_to(normal, (6, 3))).T
    return (*[np.asarray(x) for x in [mass, centers, tensors, linear, angular]],
            point_jacobian, normal_jacobian, reconstruction_error)


def propose(event, inertials, model, goal=1.2):
    mass, positions, tensors, linear, angular, jp, jn, reconstruction_error = event_geometry(event, inertials, model)
    center, jc, base, locked = momentum_maps(mass, positions, tensors, linear, angular)
    hit, normal = np.asarray(event['point_m']), np.asarray(event['face_normal'])
    coupled_point = jp - jc - skew(hit - center) @ base
    coupled_normal = jn - skew(normal) @ base
    q, qd = [np.asarray(event[k][-6:]) for k in ['joint_position_rad', 'joint_velocity_rad_s']]
    velocity = np.asarray(event['point_velocity_mps'])
    normal_rate = np.cross(event['flange_angular_velocity_rad_s'], normal)
    lower = np.maximum(-5., (model.bounds[:, 0] - q) / .08)
    upper = np.minimum(5., (model.bounds[:, 1] - q) / .08)
    assert np.all(upper > lower)
    matrix = np.r_[3 * (normal @ coupled_point)[None], .3 * coupled_normal, .03 * np.eye(6)]
    point_offset = velocity - coupled_point @ qd
    rate_offset = normal_rate - coupled_normal @ qd
    rhs = np.r_[3 * (goal - normal @ point_offset), -.3 * rate_offset, .03 * qd]
    fit = lsq_linear(matrix, rhs, bounds=(lower, upper), tol=1e-12, max_iter=1000)
    assert fit.success
    change = fit.x - qd
    base_change = base @ change
    body_velocities = -jc @ change + np.cross(base_change, positions - center) + linear @ change
    body_omegas = base_change + angular @ change
    delta_linear = (mass[:, None] * body_velocities).sum(0)
    delta_angular = (tensors @ body_omegas[..., None]).squeeze(-1).sum(0)
    delta_angular += np.cross(positions - center, mass[:, None] * body_velocities).sum(0)
    assert np.linalg.norm(delta_linear) < 1e-10
    assert np.linalg.norm(delta_angular) < 1e-10
    vn = normal @ coupled_point
    max_delta = np.where(vn >= 0, upper, lower) - qd
    return dict(env_id=event['env_id'],episode_index=event['episode_index'],time_s=event['time_s'],
                actual_contact_point_m=event['point_m'],actual_point_velocity_mps=velocity.tolist(),
                actual_face_normal=normal.tolist(),actual_normal_speed_mps=float(normal @ velocity),
                actual_joint_velocity_rad_s=qd.tolist(),proposed_joint_velocity_rad_s=fit.x.tolist(),
                coupled_proposed_point_velocity_mps=(velocity + coupled_point @ change).tolist(),
                coupled_proposed_normal_speed_mps=float(normal @ (velocity + coupled_point @ change)),
                coupled_proposed_normal_rate_rad_s=(normal_rate + coupled_normal @ change).tolist(),
                proposed_base_omega_change_rad_s=base_change.tolist(),
                proposed_base_omega_rad_s=(np.asarray(event['root_angular_velocity_rad_s']) + base_change).tolist(),
                coupled_box_max_normal_speed_mps=float(normal @ (velocity + coupled_point @ max_delta)),
                fixed_base_proposal_normal_speed_mps=float(normal @ (velocity + jp @ change)),
                momentum_linear_residual_kg_mps=delta_linear.tolist(),
                momentum_angular_residual_kg_m2ps=delta_angular.tolist(),
                locked_inertia_eigenvalues_kg_m2=np.linalg.eigvalsh(locked).tolist(),
                source_kinematics_reconstruction_error_m=reconstruction_error,
                offline_reconstruction_consistency_guard_m=5e-5,
                total_mass_kg=float(mass.sum()),native_execution_of_proposal=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contacts', type=Path, required=True)
    parser.add_argument('--inertials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output')
    raw = args.contacts.read_bytes()
    inertials_raw = args.inertials.read_bytes()
    info = json.loads(inertials_raw)
    assert hashlib.sha256((ROOT / info['asset_path']).read_bytes()).hexdigest() == info['asset_sha256']
    bodies = {b['name']: b for b in info['bodies']}
    assert set(bodies) == set(BODY_NAMES)
    events = [x for x in json.loads(raw) if x['episode_index'] == 0 and x['first_racket_contact'] and x['string_contact']]
    model = FullRacketGeometryV4()
    rows = [propose(e, bodies, model) for e in events]
    args.output.mkdir(parents=True)
    (args.output / 'SOURCE_CONTACTS.json').write_bytes(raw)
    (args.output / 'SOURCE_INERTIALS.json').write_bytes(inertials_raw)
    (args.output / 'generator_source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='Offline instantaneous fixed-configuration centroidal momentum-conserving arm velocity proposals.',
                  source_contacts=str(args.contacts),source_contacts_sha256=hashlib.sha256(raw).hexdigest(),
                  source_inertials_sha256=hashlib.sha256(inertials_raw).hexdigest(),rows=rows,
                  limitations=['Only arm joint-rate variations are modeled; leg rates and current configuration remain fixed.',
                               'Centroidal momentum is held fixed over an infinitesimal velocity variation, not a finite swing.',
                               'No actuator dynamics, torque tracking, earlier interception, collision or recovery is verified.',
                               'Authored USD inertias are used; all source physical parameters remain unchanged.',
                               'Any proposed controller must be executed natively from ground before performance claims.'],
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_acceptance=False)
    (args.output / 'PROPOSALS.json').write_text(json.dumps(report,indent=2) + '\n')
    print(json.dumps([dict(env_id=r['env_id'],actual=r['actual_normal_speed_mps'],
                           proposed=r['coupled_proposed_normal_speed_mps'],maximum=r['coupled_box_max_normal_speed_mps'],
                           qdot=r['proposed_joint_velocity_rad_s']) for r in rows]),flush=True)


if __name__ == '__main__':
    main()
