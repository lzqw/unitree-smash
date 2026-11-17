import numpy as np
import pytest

from quadsmash.control.contact_first_trajectory_search_v1 import (
    ContactFirstSearchConfig,
    ContactFirstTrajectorySearchV1,
    solve_ballistic_plane_contact,
)


def _inputs():
    jp = np.c_[np.eye(3), np.zeros((3, 3))]
    jn = np.c_[np.zeros((3, 3)), np.eye(3)]
    return dict(
        q_rad=np.zeros(6),
        qd_rad_s=np.zeros(6),
        position_jacobian=jp,
        normal_jacobian=jn,
        racket_position_world_m=np.zeros(3),
        racket_normal_world=np.array([0., 0., 1.]),
        target_position_world_m=np.array([.1, 0., 0.]),
        target_normal_world=np.array([0., 0., 1.]),
        target_velocity_world_m_s=np.array([1., 0., 0.]),
        contact_time_s=.20,
        lower_rad=np.full(6, -.5),
        upper_rad=np.full(6, .5),
        velocity_limits_rad_s=np.full(6, 5.),
    )


def test_ballistic_contact_returns_first_descending_plane_crossing():
    event = solve_ballistic_plane_contact(
        [1., 0., 2.], [0., 0., -1.], [1., 0., 1.5], [0., 0., 1.],
        max_time_s=2.)
    assert event is not None
    # Gravity makes the first descending crossing earlier than the constant
    # velocity estimate of 0.5 s.
    assert event.time_s == pytest.approx(0.23321676, abs=1e-6)
    assert event.position_world_m[2] == pytest.approx(1.5, abs=1e-6)
    assert event.velocity_world_m_s[2] < 0.


def test_contact_first_search_is_deterministic_and_hits_terminal_state():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    first = planner.plan(**_inputs())
    second = planner.plan(**_inputs())
    np.testing.assert_array_equal(first.q_rad, second.q_rad)
    np.testing.assert_array_equal(first.qd_rad_s, second.qd_rad_s)
    assert first.contact_index == 20
    assert first.q_rad.shape == (32, 6)
    assert first.qd_rad_s.shape == (32, 6)
    np.testing.assert_allclose(first.q_rad[first.contact_index, :3], [.1, 0., 0.], atol=3e-4)
    np.testing.assert_allclose(first.q_rad[first.contact_index, 3:], 0.)


def test_base_velocity_is_removed_once_from_joint_velocity_target():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['base_linear_velocity_world_m_s'] = np.array([.4, 0., 0.])
    result = planner.plan(**inputs)
    # The first three arm joints map directly to racket position in this
    # synthetic Jacobian, so the target joint speed is 1.0 - 0.4.
    assert result.qd_rad_s[result.contact_index, 0] == pytest.approx(.6, abs=2e-3)


def test_full_body_evaluator_can_reject_candidates_without_partial_acceptance():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(-.02, 0., .02), velocity_scales=(.8, 1., 1.2)))
    calls = []

    def evaluator(q, qd, contact_index, metadata):
        calls.append((q.copy(), qd.copy(), contact_index, dict(metadata)))
        return {"accepted": metadata["contact_time_s"] >= .20}

    result = planner.plan(**_inputs(), evaluator=evaluator)
    assert calls
    assert result.contact_time_s >= .20
    assert result.metadata['evaluator']['accepted'] is True
    assert all(q.shape == (32, 6) and qd.shape == (32, 6)
               for q, qd, _, _ in calls)


def test_invalid_contact_time_and_singular_normal_are_rejected():
    planner = ContactFirstTrajectorySearchV1()
    with pytest.raises(ValueError, match='Positive contact time'):
        planner.plan(**dict(_inputs(), contact_time_s=.0))
    with pytest.raises(ValueError, match='Nonzero contact-plane normal'):
        solve_ballistic_plane_contact([0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0])

