"""Rotating-frame acceleration regressions for causal contact prediction."""
import torch
from quadsmash.control.full_racket_rally_controller_v4 import (
    public_world_basis_in_body, transported_body_acceleration,
)


def test_constant_world_twist_does_not_invent_acceleration_when_body_rotates():
    # Exact finite rotation, not an infinitesimal omega-cross-v approximation.
    angle = torch.tensor(.37, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    basis = torch.tensor([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=torch.float64)[None]
    old_basis = torch.eye(3, dtype=torch.float64)[None]
    public = torch.zeros(1, 794, dtype=torch.float64)
    public[:, 763:766] = basis[:, :, 0]
    public[:, 6:9] = -basis[:, :, 2]
    torch.testing.assert_close(public_world_basis_in_body(public), basis)
    for world_vector in ([1., 2., .4], [.3, -.2, 2.]):
        previous = torch.tensor([world_vector], dtype=torch.float64)
        current = torch.bmm(basis, previous[:, :, None]).squeeze(-1)
        result = transported_body_acceleration(current, previous, basis, old_basis,
                                               torch.tensor([.01]), torch.tensor([True]))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-12, rtol=0.)
        assert torch.linalg.norm((current-previous)/.01) > 1.


def test_inertial_acceleration_is_expressed_in_current_body_and_lanes_are_isolated():
    basis = torch.tensor([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
                          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=torch.float64)
    old_basis = torch.eye(3, dtype=torch.float64)[None].repeat(2, 1, 1)
    previous = torch.tensor([[1., 0., 0.], [9., 8., 7.]], dtype=torch.float64)
    acceleration_world = torch.tensor([[2., -3., -9.81], [0., 0., 0.]], dtype=torch.float64)
    current = torch.bmm(basis, (previous+.01*acceleration_world)[:, :, None]).squeeze(-1)
    result = transported_body_acceleration(current, previous, basis, old_basis,
        torch.tensor([.01, float('nan')], dtype=torch.float64), torch.tensor([True, False]))
    expected = torch.bmm(basis, acceleration_world[:, :, None]).squeeze(-1)
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()


def test_world_pose_translation_pitch_and_twist_round_trip():
    import numpy as np
    from quadsmash.control.full_racket_rally_controller_v4 import planning_world_pose
    from quadsmash.control.contact_first_trajectory_search_v1 import rotation_current_from_future_from_rotvec
    rotation=rotation_current_from_future_from_rotvec(np.array([.2,-.4,.3]))
    origin=np.array([.7,-.6,.9]);command=np.array([1.1,0.,1.8])
    public=np.zeros(794);public[6:9]=rotation.T@np.array([0.,0.,-1.])
    public[763:766]=rotation.T@np.array([1.,0.,0.])
    public[760:763]=rotation.T@(command-origin)
    basis,root=planning_world_pose(public,command)
    np.testing.assert_allclose(basis,rotation,atol=1e-12)
    np.testing.assert_allclose(root,origin,atol=1e-12)
    point=np.array([.3,.1,.4]);linear=np.array([.2,-.1,.6]);angular=np.array([.3,.5,-.2])
    world_velocity=rotation@(linear+np.cross(angular,point))
    np.testing.assert_allclose(basis.T@world_velocity,linear+np.cross(angular,point),atol=1e-12)


def test_causal_shuttle_target_has_no_duplicate_nominal_intercept_offset():
    """Reproduce the 100.23 mm failure in the production callback math."""
    import numpy as np
    from quadsmash.control.full_racket_rally_controller_v4 import (
        causal_shuttle_target_in_planning_frame, planning_world_pose,
    )
    from quadsmash.control.contact_first_trajectory_search_v1 import (
        rotation_current_from_future_from_rotvec,
    )
    from quadsmash.control.initial_intercept import propagate_ball_state

    physics = {'wind_m_s': [0., 0., 0.],
               'quadratic_drag_k_per_m': 0.025442307692307694}
    rotation = rotation_current_from_future_from_rotvec(np.array([.2, -.3, .1]))
    origin = np.array([.68, -.04, .25])
    p0 = np.array([2.45, .16, 1.8])
    v0 = np.array([-7.8, -.5, -3.6])
    duration = .79
    reference_incoming = np.array([-8.1, -.4, -3.9])
    desired_racket_velocity = np.array([2.2, -.8, -.5])
    command_world = np.array([1.1, 0., 1.8])
    public = np.zeros(794)
    public[6:9] = rotation.T @ [0., 0., -1.]
    public[763:766] = rotation.T @ [1., 0., 0.]
    public[760:763] = rotation.T @ (command_world - origin)
    basis, root = planning_world_pose(public, command_world)
    target = causal_shuttle_target_in_planning_frame(
        physics, p0, v0, reference_incoming, basis, root,
        desired_racket_velocity, duration)
    ball, incoming = propagate_ball_state(p0, v0, duration, physics)
    np.testing.assert_allclose(
        root + basis @ target['position_world_m'], ball, atol=2e-12)
    np.testing.assert_allclose(
        basis @ target['incoming_ball_velocity_world_m_s'], incoming, atol=2e-12)
    np.testing.assert_allclose(
        target['velocity_world_m_s'],
        desired_racket_velocity + basis.T @ (incoming-reference_incoming),
        atol=2e-12)
    # The frozen probe's legacy nominal-target displacement was ~100.23 mm.
    # Adding it again must reproduce the error, never the production target.
    legacy_offset = np.array([.10022794451986757, 0., 0.])
    double_compensated = target['position_world_m'] + basis.T @ legacy_offset
    assert np.linalg.norm(root + basis @ double_compensated - ball) > .100
    later = causal_shuttle_target_in_planning_frame(
        physics, p0, v0, reference_incoming, basis, root,
        desired_racket_velocity, duration+.02)
    later_ball, _ = propagate_ball_state(p0, v0, duration+.02, physics)
    np.testing.assert_allclose(root + basis @ later['position_world_m'],
                               later_ball, atol=2e-12)


def test_causal_shuttle_target_rejects_incorrect_frame_shapes():
    import numpy as np
    import pytest
    from quadsmash.control.full_racket_rally_controller_v4 import (
        causal_shuttle_target_in_planning_frame,
    )
    physics = {'wind_m_s':[0.,0.,0.], 'quadratic_drag_k_per_m':0.}
    with pytest.raises(ValueError, match='planning-frame inputs'):
        causal_shuttle_target_in_planning_frame(
            physics, np.zeros(3), np.zeros(3), np.zeros(3),
            np.ones((2, 3)), np.zeros(3), np.zeros(3), .2)
