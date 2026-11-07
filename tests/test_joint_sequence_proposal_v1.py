import numpy as np
import pytest
from scripts.landing_joint_sequence_proposal_v1 import affine_sequence, knot_basis


def test_sequence_matches_independent_scalar_implicit_equations():
    basis = knot_basis(5, 3)
    commands = np.array([.3, -.1, .2])
    q, v, mass, kp, kd = .1, .2, 1.7, 3.2, .4
    stages = list(affine_sequence(np.array([[mass]]), np.array([.25]), np.ones((1, 1)),
        np.array([kp]), np.array([kd]), np.empty((0, 1)), np.empty(0),
        np.array([v]), np.array([q]), basis, substeps=4))
    displacement = 0.; h = .01/4
    for s in stages:
        u = float(basis[s['tick']]@commands)
        # Backward Euler PD: m(v_next-v)/h = kp(u-q-h*v_next)-kd*v_next-b.
        v = (mass*v+h*(kp*(u-q)-.25))/(mass+h*kd+h*h*kp)
        q += h*v; displacement += h*v
        np.testing.assert_allclose(s['velocity_map']@commands+s['velocity_bias'], [v], atol=1e-13)
        np.testing.assert_allclose(s['joint_position_map']@commands+s['joint_position_bias'], [q], atol=1e-13)
        np.testing.assert_allclose(s['torque_map']@commands+s['torque_bias'], [kp*(u-q)-kd*v], atol=1e-13)
        np.testing.assert_allclose(s['displacement_map']@commands+s['displacement_bias'], [displacement], atol=1e-13)
    # The first command still affects the endpoint; future knots cannot alter past states.
    assert abs(stages[-1]['velocity_map'][0, 0]) > 1e-4
    assert stages[0]['velocity_map'][0, -1] == 0.


def test_contact_force_satisfies_dynamics_without_fictitious_velocity():
    basis = knot_basis(3, 2); command = np.array([.1, .2])
    stages = affine_sequence(np.array([[2.]]), np.array([9.]), np.ones((1, 1)),
        np.array([4.]), np.array([.3]), np.ones((1, 1)), np.zeros(1),
        np.zeros(1), np.zeros(1), basis, substeps=2)
    for s in stages:
        v = s['velocity_map']@command+s['velocity_bias']
        f = s['force_map']@command+s['force_bias']
        torque = s['torque_map']@command+s['torque_bias']
        np.testing.assert_allclose(v, 0., atol=1e-13)
        np.testing.assert_allclose(torque+f, 9., atol=1e-12)


def test_knot_basis_is_convex_and_bounds_are_explicit():
    b = knot_basis(40, 9)
    np.testing.assert_allclose(b.sum(1), 1.)
    assert np.all(b >= 0.)
    np.testing.assert_array_equal(b[0], np.eye(9)[0])
    np.testing.assert_array_equal(b[-1], np.eye(9)[-1])
    for args in [(1, 1), (40, 41), (81, 9), (40, 1)]:
        with pytest.raises(ValueError): knot_basis(*args)


def test_nonlinear_verifier_rejects_unadmitted_commands_and_restores_history():
    from test_admitted_arm_support_qp_v1 import fixture
    from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    from scripts.landing_joint_sequence_proposal_v1 import verify_sequence
    _, interface, state = fixture()
    state.previous_command_target = state.joint_position.copy()
    controller = LandingFullCommandIntervalMPCV1(interface,
        settings=LandingMPCSettingsV2(horizon_steps=40), arm_target_rate_rad_s=10.)
    prior = state.previous_command_target.copy()
    controller.qp.interface.bind_previous_command(prior)
    bad = np.tile(prior, (40, 1)); bad[0, 12] += .3
    outcome = verify_sequence(controller, state, bad)
    assert not outcome['accepted'] and outcome['failed_stage'] == 0
    assert outcome['rejection'] == 'Actual command admission differs'
    np.testing.assert_array_equal(controller.qp.interface.previous_command, prior)
    np.testing.assert_array_equal(state.previous_command_target, prior)
