import numpy as np
import pytest

from quadsmash.control.contact_first_trajectory_search_v1 import (
    ContactFirstSearchConfig,
    ContactFirstTrajectorySearchV1,
    rotation_current_from_future_from_rotvec,
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


def test_base_angular_velocity_is_removed_at_racket_offset():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['base_angular_velocity_world_rad_s'] = np.array([0., 1., 0.])
    inputs['racket_offset_world_m'] = np.array([0., 0., 1.])
    result = planner.plan(**inputs)
    # omega x r = [1, 0, 0], exactly consuming the requested racket velocity.
    assert result.metadata['base_point_velocity_world_m_s'][0] == pytest.approx(1.)
    assert result.qd_rad_s[result.contact_index, 0] == pytest.approx(0., abs=2e-3)
    assert 0.0 <= result.metadata['peak_joint_speed_ratio'] <= 1.0
    assert 0.0 <= result.metadata['peak_joint_step_ratio'] <= 1.0


def test_causal_base_acceleration_is_extrapolated_to_contact():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['base_linear_acceleration_world_m_s2'] = np.array([1., 0., 0.])
    result = planner.plan(**inputs)
    # Over 0.20 s, the causal base-velocity estimate gains 0.20 m/s.
    assert result.metadata['base_point_velocity_world_m_s'][0] == pytest.approx(.2)
    assert result.metadata['relative_target_velocity_world_m_s'][0] == pytest.approx(.8)


def test_causal_base_translation_is_removed_from_arm_contact_target():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['target_position_world_m'] = np.array([.1, 0., 0.])
    inputs['base_linear_acceleration_world_m_s2'] = np.array([1., 0., 0.])
    result = planner.plan(**inputs)
    # The base translates by 0.5*a*t^2 before contact, so the arm terminal
    # target is .08 m in the synthetic linear model rather than .10 m.
    np.testing.assert_allclose(
        result.q_rad[result.contact_index, :3], [.08, 0., 0.], atol=3e-4)
    assert result.metadata['base_displacement_world_m'][0] == pytest.approx(.02)
    assert result.metadata['relative_target_position_world_m'][0] == pytest.approx(.08)


def test_causal_base_angular_acceleration_is_extrapolated_to_contact():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['base_angular_acceleration_world_rad_s2'] = np.array([0., 1., 0.])
    inputs['racket_offset_world_m'] = np.array([0., 0., 1.])
    result = planner.plan(**inputs)
    # alpha x r = [1, 0, 0] at t=.20 s, contributing .20 m/s.
    assert result.metadata['base_point_velocity_world_m_s'][0] == pytest.approx(.2)
    assert result.metadata['relative_target_velocity_world_m_s'][0] == pytest.approx(.8)


def test_base_acceleration_prediction_is_bounded_and_audited():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        base_angular_acceleration_limit_rad_s2=2.0))
    inputs = _inputs()
    inputs['base_angular_acceleration_world_rad_s2'] = np.array([0., 9., 0.])
    result = planner.plan(**inputs)
    assert result.metadata['raw_base_angular_acceleration_world_rad_s2'][1] == pytest.approx(9.)
    assert result.metadata['used_base_angular_acceleration_world_rad_s2'][1] == pytest.approx(2.)


def test_unreachable_velocity_uses_explicit_reachable_projection():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        velocity_tolerance_m_s=.1, minimum_contact_speed_m_s=.1))
    inputs = _inputs()
    inputs['target_velocity_world_m_s'] = np.array([10., 0., 0.])
    result = planner.plan(**inputs)
    assert result.metadata['velocity_projection'] is True
    assert result.metadata['achieved_contact_speed_m_s'] >= .1
    np.testing.assert_allclose(
        result.metadata['effective_target_velocity_future_base_frame_m_s'],
        result.metadata['requested_target_velocity_world_m_s'] / 2., atol=1e-6)


def test_zero_contact_speed_is_rejected_when_required():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        minimum_contact_speed_m_s=.1))
    inputs = _inputs()
    inputs['target_velocity_world_m_s'] = np.zeros(3)
    with pytest.raises(RuntimeError, match='No contact-first candidate'):
        planner.plan(**inputs)
    assert planner.last_diagnostics['rejection_counts']['zero_contact_speed'] == 1


def test_time_varying_target_is_queried_for_each_contact_offset():
    calls = []
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(-.02, 0., .02),
        velocity_scales=(1.,)))

    def target_state(duration_s):
        calls.append(float(duration_s))
        return {
            "position_world_m": np.array([.1 + .5 * (duration_s - .2), 0., 0.]),
            "velocity_world_m_s": np.array([.5, 0., 0.]),
            "normal_world": np.array([0., 0., 1.]),
        }

    result = planner.plan(**_inputs(), target_state_fn=target_state)
    # The nominal event is resolved once for the warm start, then each
    # distinct candidate offset is queried once from the causal cache.
    assert calls == pytest.approx([.20, .18, .22], abs=1e-9)
    expected = .1 + .5 * (result.contact_time_s - .2)
    assert result.metadata['target_position_world_m'][0] == pytest.approx(expected)


def test_future_base_rotation_transforms_target_into_future_body_frame():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        max_iteration_step_rad=1.0, minimum_contact_speed_m_s=0.0))
    inputs = _inputs()
    inputs.update(
        target_position_world_m=np.array([0., 1., 0.]),
        target_velocity_world_m_s=np.zeros(3),
        upper_rad=np.full(6, 2.),
        velocity_limits_rad_s=np.full(6, 20.),
    )
    quarter_turn = rotation_current_from_future_from_rotvec(
        np.array([0., 0., np.pi / 2.]))

    def base_state(duration_s):
        return {
            "rotation_current_from_future": quarter_turn,
            "translation_current_m": np.zeros(3),
            "linear_velocity_current_m_s": np.zeros(3),
            "angular_velocity_current_rad_s": np.zeros(3),
        }

    result = planner.plan(**inputs, base_state_fn=base_state)
    # The future body y axis is current-frame -x for this convention, so a
    # current-frame target at +y appears at +x in the future body frame.
    np.testing.assert_allclose(
        result.q_rad[result.contact_index, :3], [1., 0., 0.], atol=3e-3)
    np.testing.assert_allclose(
        result.metadata["base_rotation_current_from_future"], quarter_turn)
    assert result.metadata["base_pose_source"] == "causal_callback"


def test_forward_kinematics_callback_relinearizes_before_acceptance():
    calls = []

    def forward_kinematics(q):
        calls.append(q.copy())
        position = np.array([np.sin(q[0]), q[1], q[2]])
        jacobian = np.zeros((3, 6))
        jacobian[0, 0] = np.cos(q[0])
        jacobian[1, 1] = 1.0
        jacobian[2, 2] = 1.0
        normal = np.array([0., 0., 1.])
        normal_jacobian = np.zeros((3, 6))
        return position, normal, jacobian, normal_jacobian

    inputs = _inputs()
    inputs.update(
        racket_position_world_m=np.zeros(3),
        target_position_world_m=np.array([np.sin(.2), 0., 0.]),
        target_velocity_world_m_s=np.array([.3, 0., 0.]),
        position_jacobian=np.c_[np.eye(3), np.zeros((3, 3))],
        normal_jacobian=np.zeros((3, 6)),
    )
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        kinematic_iterations=4, max_iteration_step_rad=.25))
    result = planner.plan(**inputs, kinematics_fn=forward_kinematics)
    # Four full task iterations, the original position-priority calls, and
    # four bounded position-only candidate refinements.
    assert len(calls) == 16
    assert result.position_error_m < 1e-3
    assert result.metadata['kinematic_iterations'] == 4


def test_regular_unit_normal_task_explores_one_joint_nullspace():
    calls = []

    def forward_kinematics(q):
        calls.append(q.copy())
        position = q[:3].copy()
        normal = np.array([0., 0., 1.])
        jp = np.c_[np.eye(3), np.zeros((3, 3))]
        jn = np.zeros((3, 6))
        jn[0, 3] = 1.
        jn[1, 4] = 1.
        return position, normal, jp, jn

    inputs = _inputs()
    inputs.update(
        position_jacobian=np.c_[np.eye(3), np.zeros((3, 3))],
        normal_jacobian=np.array([
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0.],
        ]),
    )
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        kinematic_iterations=1))
    result = planner.plan(**inputs, kinematics_fn=forward_kinematics)
    # Initial full-task call, four nullspace branches, two position-priority
    # setup calls, and three bounded candidate refinements.
    assert len(calls) == 1 + 4 + 2 + 1
    assert result.metadata['search_diagnostics']['candidates_considered'] == 6
    assert result.metadata['nullspace_offset_rad'] in (-.12, -.06, .06, .12, 0.)


def test_hermite_speed_overshoot_is_rejected_before_scoring():
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,)))
    inputs = _inputs()
    inputs['velocity_limits_rad_s'] = np.full(6, .2)
    with pytest.raises(RuntimeError, match='No contact-first candidate'):
        planner.plan(**inputs)
    assert planner.last_diagnostics['rejection_counts'] == {
        'contact_time_below_joint_speed_bound': 1}
    assert planner.last_diagnostics['minimum_joint_speed_duration_s'] > .2


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



def test_physical_evaluator_receives_position_normal_velocity_in_one_frame():
    rotation = rotation_current_from_future_from_rotvec(np.array([0., .1, 0.]))
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, contact_time_offsets_s=(0.,), velocity_scales=(1.,),
        normal_tangent_offsets_rad=((0.,0.),), nullspace_offsets_rad=(),
        adaptive_normal_target_max_deviation_rad=0.,
        allow_reachable_velocity_projection=False))
    seen = []
    inputs = _inputs()
    def evaluate(q, qd, contact, metadata):
        seen.append(metadata)
        np.testing.assert_allclose(metadata['target_position_world_m'],
                                   inputs['target_position_world_m'], atol=1e-12)
        np.testing.assert_allclose(metadata['target_normal_world'],
                                   inputs['target_normal_world'], atol=1e-12)
        np.testing.assert_allclose(metadata['target_velocity_world_m_s'],
                                   inputs['target_velocity_world_m_s'], atol=1e-12)
        assert metadata['terminal_target_frame'] == 'current_planning_frame'
        return True
    planner.plan(**inputs, evaluator=evaluate, base_state_fn=lambda _: dict(
        rotation_current_from_future=rotation, translation_current_m=np.zeros(3),
        linear_velocity_current_m_s=np.zeros(3), angular_velocity_current_rad_s=np.zeros(3)))
    assert seen


def test_two_element_causal_target_keeps_original_normal_after_base_rotation():
    rotation = rotation_current_from_future_from_rotvec(np.array([0., .1, 0.]))
    inputs = _inputs()
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(
        horizon_steps=32, normal_tangent_offsets_rad=((0.,0.),),
        adaptive_normal_target_max_deviation_rad=0., nullspace_offsets_rad=()))
    result = planner.plan(**inputs,
        target_state_fn=lambda _: (inputs['target_position_world_m'], inputs['target_velocity_world_m_s']),
        base_state_fn=lambda _: dict(rotation_current_from_future=rotation,
            translation_current_m=np.zeros(3), linear_velocity_current_m_s=np.zeros(3),
            angular_velocity_current_rad_s=np.zeros(3)))
    np.testing.assert_allclose(result.metadata['requested_target_normal_world'],
                               inputs['target_normal_world'], atol=1e-12)


def test_fast_racket_moving_with_ball_is_not_an_approaching_impact():
    inputs = _inputs()
    cfg = ContactFirstSearchConfig(horizon_steps=32, normal_tangent_offsets_rad=((0.,0.),),
        adaptive_normal_target_max_deviation_rad=0., nullspace_offsets_rad=())
    planner = ContactFirstTrajectorySearchV1(cfg)
    # Racket speed is high along x, but the selected face points +z and the
    # ball is separating along +z. No normal impact can occur.
    def target(_):
        return dict(position_world_m=inputs['target_position_world_m'],
            velocity_world_m_s=inputs['target_velocity_world_m_s'],
            incoming_ball_velocity_world_m_s=np.array([1.,0.,1.]))
    with pytest.raises(RuntimeError, match='nonapproaching_contact'):
        planner.plan(**inputs, target_state_fn=target)
    assert planner.last_diagnostics['rejection_counts']['nonapproaching_contact'] > 0


def test_approaching_ball_velocity_is_causal_and_audited():
    inputs = _inputs()
    planner = ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(horizon_steps=32))
    result = planner.plan(**inputs, target_state_fn=lambda t: dict(
        position_world_m=inputs['target_position_world_m'],
        velocity_world_m_s=inputs['target_velocity_world_m_s'],
        incoming_ball_velocity_world_m_s=np.array([0.,0.,-2.-t])))
    assert result.metadata['approaching_contact_checked']
    assert result.metadata['relative_normal_closing_speed_m_s'] > 0.
    np.testing.assert_allclose(result.metadata['incoming_ball_velocity_current_frame_m_s'],
                               [0.,0.,-2.-result.contact_time_s])


def test_invalid_outgoing_outcome_cannot_win_on_low_ik_error():
    planner=ContactFirstTrajectorySearchV1(ContactFirstSearchConfig(horizon_steps=32))
    with pytest.raises(RuntimeError,match='impact_predicted_net_intersection'):
        planner.plan(**_inputs(),impact_evaluator=lambda _:dict(valid_return=False,
            rejection_reason='predicted_net_intersection'))
    result=planner.plan(**_inputs(),impact_evaluator=lambda _:dict(valid_return=True,
        ranking_cost=2.,native_execution=False))
    assert result.metadata['impact_outcome']['ranking_cost']==2.
    assert result.score>=2.
