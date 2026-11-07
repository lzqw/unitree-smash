"""Static reference using the same foot-sphere surface Jacobian as prediction.

This is a future preload proposal, not a dynamically feasible transition.
The caller must enforce command history and verify a full dynamic rollout.
"""
import copy
import numpy as np
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.predict_frictional_spheres_v1 import surface_jacobians


def sphere_static_reference(controller, state, spheres):
    model = controller.qp.model
    interface = controller.qp.interface
    feet = np.flatnonzero(state.contacts)
    if not len(feet):
        return dict(solved=False, reason='No predicted loaded support')
    snapshot = copy.deepcopy(model.snapshot(state.q, np.zeros(model.model.nv)))
    jacobian = surface_jacobians(model, state.q, spheres)[feet].reshape(-1, model.model.nv)
    selection = np.zeros((model.model.nv, 18))
    selection[model.v_indices, np.arange(18)] = 1.
    dynamics = np.c_[selection, jacobian.T]
    nf = len(feet)
    n = 18+3*nf
    low_torque = np.maximum(-interface.effort, interface.kp*(interface.limits[:, 0]-state.joint_position))
    high_torque = np.minimum(interface.effort, interface.kp*(interface.limits[:, 1]-state.joint_position))
    angles = np.arange(16)*2*np.pi/16
    # Inscribed polygon is conservative relative to the predictor's circle.
    cone = np.c_[np.cos(angles), np.sin(angles), np.full(16, -interface.friction*np.cos(np.pi/16))]
    rows = [dynamics, np.c_[np.eye(18), np.zeros((18, 3*nf))]]
    lower = [snapshot['nonlinear'], low_torque]
    upper = [snapshot['nonlinear'], high_torque]
    for i in range(nf):
        force = np.zeros((3, n))
        force[:, 18+3*i:18+3*i+3] = np.eye(3)
        rows.extend([force[2:3], cone@force])
        lower.extend([np.array([0.]), np.full(16, -np.inf)])
        upper.extend([np.array([np.inf]), np.zeros(16)])
    P = np.diag(np.r_[1/interface.kp**2, np.full(3*nf, 1e-6)])
    linear = np.r_[(state.joint_position-state.previous_command_target)/interface.kp, np.zeros(3*nf)]
    solver = DenseActiveSetQP()
    solver.setup(P=P, q=linear, A=np.vstack(rows), l=np.concatenate(lower), u=np.concatenate(upper),
        eps_abs=1e-7, eps_rel=1e-7)
    solved = solver.solve()
    if solved.info.status_val != 1:
        return dict(solved=False, reason=solved.info.status)
    torque = solved.x[:18]
    forces = solved.x[18:].reshape(nf, 3)
    target = state.joint_position+torque/interface.kp
    residual = float(np.max(abs(dynamics@solved.x-snapshot['nonlinear'])))
    cone_error = float(np.max(np.linalg.norm(forces[:, :2], axis=1)-interface.friction*forces[:, 2]))
    if residual > 1e-6 or cone_error > 1e-6 or np.min(forces[:, 2]) < -1e-6:
        raise ValueError('Independent surface-force balance failed')
    return dict(solved=True, target=target.tolist(), torque=torque.tolist(), forces=forces.tolist(),
        support_indices=feet.tolist(), dynamics_residual=residual, future_reference_only=True,
        rate_limited_dynamic_validation_required=True, native_execution=False)
