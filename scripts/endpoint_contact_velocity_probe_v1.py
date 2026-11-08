"""Two-stage position/velocity contact hypothesis, not an exact TGS solver.

Stage one predicts position with the existing coupled step. Stage two solves
the same interval's momentum/drive problem using that predicted contact
geometry, without integrating position again. No measured endpoint is used.
No external impulse or motor torque is charged a second time.
"""
import numpy as np
import pinocchio as pin

from scripts.foot_pair_contact_step_v1 import foot_pair_velocity_step, foot_pair_geometry
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step


def endpoint_velocity_probe(model, interface, state, target, spheres, dt, *, foot_friction):
    first = foot_pair_velocity_step(model, interface, state, target, spheres, dt,
        foot_friction=foot_friction)
    predicted_q = pin.integrate(model.model, state.q, dt * first['velocity'])
    ground = surface_jacobians(model, predicted_q, spheres)
    pairs = foot_pair_geometry(model, predicted_q, spheres)
    gaps = np.r_[sphere_gaps(model, predicted_q, spheres), [p['gap'] for p in pairs]]
    jac = np.concatenate([ground, np.array([p['jacobian'] for p in pairs])])
    scale = np.ones((len(gaps), 3)); scale[4:, :2] = foot_friction / interface.friction
    transformed = (scale[:, :, None] * jac).reshape(-1, model.model.nv)
    select = np.zeros((model.model.nv, len(model.v_indices)))
    select[model.v_indices, np.arange(len(model.v_indices))] = 1.
    # This replaces the interval impulse solve; it is not a second full dt
    # applied to the already advanced velocity. Positive residual separation
    # still permits closing, while penetration is not given an energy bias.
    snap = model.snapshot(state.q, state.v)
    second = position_limited_velocity_step(snap['mass_matrix'], state.v, snap['nonlinear'],
        select, interface.kp, interface.kd, target - state.q[model.q_indices], interface.effort,
        dt, transformed, -np.maximum(gaps, 0.) / dt, interface.friction, interface.velocity,
        state.q[model.q_indices], interface.limits)
    impulse = second['impulse'].reshape(-1, 3) * scale
    return dict(q=predicted_q, velocity=second['velocity'], ground_impulse=impulse[:4],
        pair_impulse=impulse[4:], momentum_residual=second['momentum_residual'],
        first_velocity=first['velocity'], first_momentum_residual=first['momentum_residual'],
        native_equivalent=False)
