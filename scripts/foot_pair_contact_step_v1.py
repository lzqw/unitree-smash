"""Experimental foot-ground plus foot-foot contact prediction.

All pair impulses act at a common point on both bodies, so they cannot add
external base force/torque. Per-pair friction uses a coordinate change in the
existing coupled solver. No recorded future force/state enters the solve.
Other self contacts and PhysX finite-iteration history remain unmodeled.
"""
import itertools

import numpy as np
import pinocchio as pin

from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step


def foot_pair_geometry(model, q, spheres):
    pin.computeJointJacobians(model.model, model.data, q)
    pin.updateFramePlacements(model.model, model.data)
    centers, poses, jacobians = [], [], []
    for sphere in spheres:
        frame = model.body_frames[sphere['body']]
        pose = model.data.oMf[frame]
        poses.append(pose)
        centers.append(pose.act(np.asarray(sphere['center_body_m'])))
        jacobians.append(pin.getFrameJacobian(model.model, model.data, frame,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy())
    rows = []
    for a, b in itertools.combinations(range(len(spheres)), 2):
        difference = centers[a] - centers[b]
        distance = float(np.linalg.norm(difference))
        if distance < 1e-10:
            raise ValueError('Coincident sphere centers have no contact normal')
        normal = difference / distance
        gap = distance - spheres[a]['radius_m'] - spheres[b]['radius_m']
        point_a = centers[a] - spheres[a]['radius_m'] * normal
        point_b = centers[b] + spheres[b]['radius_m'] * normal
        point = .5 * (point_a + point_b)
        axis = np.eye(3)[int(np.argmin(abs(normal)))]
        tangent = np.cross(axis, normal)
        tangent /= np.linalg.norm(tangent)
        basis = np.column_stack([tangent, np.cross(normal, tangent), normal])
        ja = jacobians[a][:3] - pin.skew(point - poses[a].translation) @ jacobians[a][3:]
        jb = jacobians[b][:3] - pin.skew(point - poses[b].translation) @ jacobians[b][3:]
        relative = basis.T @ (ja - jb)
        if np.max(abs(relative[:, :6])) > 1e-9:
            raise ValueError('Internal pair produced an external base wrench')
        rows.append(dict(pair=(a, b), gap=gap, jacobian=relative, basis=basis))
    return rows


def foot_pair_velocity_step(model, interface, state, target, spheres, dt, *, foot_friction):
    if (not np.isfinite(foot_friction) or foot_friction <= 0
            or not np.isfinite(interface.friction) or interface.friction <= 0):
        raise ValueError('Positive native ground and foot material coefficients required')
    q, v = state.q, state.v
    snap = model.snapshot(q, v)
    ground = surface_jacobians(model, q, spheres)
    pairs = foot_pair_geometry(model, q, spheres)
    ground_gaps = sphere_gaps(model, q, spheres)
    gaps = np.r_[ground_gaps, [p['gap'] for p in pairs]]
    jacobian = np.concatenate([ground, np.array([p['jacobian'] for p in pairs])])
    # J'=D J, lambda_physical=D lambda'. The original circular cone with
    # coefficient mu then represents radius foot_friction*lambda_normal.
    scale = np.ones((len(gaps), 3))
    scale[4:, :2] = foot_friction / interface.friction
    transformed = (scale[:, :, None] * jacobian).reshape(-1, model.model.nv)
    selection = np.zeros((model.model.nv, len(model.v_indices)))
    selection[model.v_indices, np.arange(len(model.v_indices))] = 1.
    result = position_limited_velocity_step(snap['mass_matrix'], v, snap['nonlinear'],
        selection, interface.kp, interface.kd, target - q[model.q_indices],
        interface.effort, dt, transformed, -gaps / np.where(gaps > 0, dt, .01),
        interface.friction, interface.velocity, q[model.q_indices], interface.limits)
    impulse = result['impulse'].reshape(-1, 3) * scale
    return dict(result, physical_contact_impulse=impulse,
        ground_impulse=impulse[:4], pair_impulse=impulse[4:], pair_gaps=gaps[4:],
        pair_indices=[p['pair'] for p in pairs], pair_basis=[p['basis'] for p in pairs],
        external_pair_base_wrench_max=float(max(np.max(abs(p['jacobian'][:, :6])) for p in pairs)))
