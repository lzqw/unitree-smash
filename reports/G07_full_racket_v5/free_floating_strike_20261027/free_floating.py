"""Instantaneous racket Jacobian at fixed total linear/angular momentum.

This is a known-model calculation, not an actuator or collision simulator.
All vectors use the instantaneous root frame. The root translation is the
velocity of the root link origin, not its rigid body's center of mass.
"""
import numpy as np


def hit_state(model, state, local_point, local_normal):
    tool = model.indices['Racket']
    offset = np.einsum('nij,nj->ni', state['rotation'][:, tool], local_point)
    normal = np.einsum('nij,nj->ni', state['rotation'][:, tool], local_normal)
    point = state['position'][:, tool] + offset
    velocity = state['linear'][:, tool] + np.cross(state['angular'][:, tool], offset)
    normal_rate = np.cross(state['angular'][:, tool], normal)
    return point, velocity, normal_rate


def linearization(model, q, local_point, local_normal, root_linear, root_angular, qd):
    """Eliminate root velocities while holding COM velocity and total H.

    H = I*w + A*qd, V_COM = v_root + Cw*w + Cq*qd.
    v_hit = v_root + Tw*w + Tq*qd.
    Thus w = I^-1*H - I^-1*A*qd and
    v_hit = V_COM + (Tw-Cw)*I^-1*H
            + [Tq-Cq-(Tw-Cw)*I^-1*A]*qd.
    """
    q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    count = len(q)
    state = model.forward(q, qd, root_linear, root_angular)
    actual_point, actual_velocity, actual_normal_rate = hit_state(model, state, local_point, local_normal)
    momentum_columns, com_columns, hit_columns, normal_columns = [], [], [], []
    for column in range(21):
        w, dq = np.zeros((count, 3)), np.zeros_like(q)
        if column < 3:
            w[:, column] = 1.
        else:
            dq[:, column-3] = 1.
        basis = model.forward(q, dq, root_angular=w)
        _, v, dn = hit_state(model, basis, local_point, local_normal)
        momentum_columns.append(basis['angular_momentum'])
        com_columns.append(basis['center_velocity'])
        hit_columns.append(v)
        normal_columns.append(dn)
    h, c, t, n = [np.stack(columns, axis=-1) for columns in
                  (momentum_columns, com_columns, hit_columns, normal_columns)]
    locked, joint_momentum = h[:, :, :3], h[:, :, 3:]
    w0 = np.linalg.solve(locked, state['angular_momentum'][..., None])[..., 0]
    wj = -np.linalg.solve(locked, joint_momentum)
    root_v0 = state['center_velocity'] - np.einsum('nij,nj->ni', c[:, :, :3], w0)
    root_vj = -c[:, :, 3:] - c[:, :, :3] @ wj
    difference = t[:, :, :3]-c[:, :, :3]
    hit0 = state['center_velocity'] + np.einsum('nij,nj->ni', difference, w0)
    hitj = t[:, :, 3:]-c[:, :, 3:]+difference @ wj
    normal0 = np.einsum('nij,nj->ni', n[:, :, :3], w0)
    normalj = n[:, :, 3:]+n[:, :, :3] @ wj
    return dict(state=state, point=actual_point, velocity=actual_velocity, normal_rate=actual_normal_rate,
        angular_momentum_matrix=h, com_velocity_matrix=c, fixed_root_hit_matrix=t,
        fixed_root_normal_rate_matrix=n, root_angular_offset=w0, root_angular_jacobian=wj,
        root_linear_offset=root_v0, root_linear_jacobian=root_vj,
        hit_offset=hit0, hit_jacobian=hitj, normal_rate_offset=normal0, normal_rate_jacobian=normalj)
