"""Batch the existing interval inequalities without changing row semantics.

Offline acceleration candidate; not installed in the native controller.
All substeps, effort/velocity/position limits and friction facets are retained.
"""
import numpy as np


def interval_constraints(stages, selection, *, effort, velocity_limit,
                         joint_limits, friction, friction_facets=8):
    if not stages or friction <= 0 or friction_facets < 4:
        raise ValueError('Physical support bounds required')
    count, na = len(stages), selection.shape[1]
    stack = lambda key: np.stack([s[key] for s in stages])
    torque, torque_bias = stack('torque_map'), stack('torque_bias')
    velocity = selection.T @ stack('velocity_map')
    velocity_bias = (selection.T @ stack('velocity_bias')[..., None])[..., 0]
    position, position_bias = stack('joint_position_map'), stack('joint_position_bias')
    matrices = [torque, velocity, position]
    lows = [-effort-torque_bias, -velocity_limit-velocity_bias,
            joint_limits[:, 0]-position_bias]
    highs = [effort-torque_bias, velocity_limit-velocity_bias,
             joint_limits[:, 1]-position_bias]
    feet = len(stages[0]['force_bias']) // 3
    if any(len(s['force_bias']) != 3*feet for s in stages):
        raise ValueError('Contact dimension must remain fixed inside an interval')
    if feet:
        force = stack('force_map').reshape(count, feet, 3, na)
        force_bias = stack('force_bias').reshape(count, feet, 3)
        angles = np.arange(friction_facets)*2.*np.pi/friction_facets
        cone = np.c_[np.cos(angles), np.sin(angles),
                     np.full(friction_facets, -friction*np.cos(np.pi/friction_facets))]
        fm = np.concatenate((force[:, :, 2:3], cone @ force), axis=2)
        fb = np.concatenate((force_bias[:, :, 2:3],
                             (cone @ force_bias[..., None])[..., 0]), axis=2)
        lower = np.full_like(fb, -np.inf); lower[:, :, 0] = 0.
        upper = np.zeros_like(fb); upper[:, :, 0] = np.inf
        matrices.append(fm.reshape(count, -1, na))
        lows.append((lower-fb).reshape(count, -1))
        highs.append((upper-fb).reshape(count, -1))
    labels = []
    for i in range(count):
        for kind in ('motor_torque', 'joint_velocity', 'joint_position'):
            labels.extend((i, kind, j) for j in range(na))
        for foot in range(feet):
            labels.append((i, f'normal_force_{foot}', 0))
            labels.extend((i, f'friction_{foot}', j) for j in range(friction_facets))
    return dict(matrix=np.concatenate(matrices, axis=1).reshape(-1, na),
                low=np.concatenate(lows, axis=1).reshape(-1),
                high=np.concatenate(highs, axis=1).reshape(-1), labels=labels)
