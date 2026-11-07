"""Check the new derivative against independent interval propagation."""
import copy
import numpy as np
from test_admitted_arm_support_qp_v1 import fixture
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.landing_joint_sequence_proposal_v1 import knot_basis
from scripts.landing_sequence_linearization_v1 import make_nominal_profile
from scripts.landing_differentiated_sequence_v1 import (
    affine_differentiated_sequence, interval_derivatives,
)
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive


def setup_profile():
    _, interface, state = fixture()
    state.previous_command_target = state.joint_position.copy()
    controller = LandingFullCommandIntervalMPCV1(interface,
        settings=LandingMPCSettingsV2(horizon_steps=4), arm_target_rate_rad_s=10.)
    plan = controller.solve(state, state.joint_position, force=True)
    profile = make_nominal_profile(controller, state, plan)
    assert profile is not None
    return controller, state, profile


def test_command_derivative_matches_independent_fixed_interval_condensation():
    c, state, profile = setup_profile()
    node = profile[0]; snap = node['snapshot']; model = c.qp.model
    nominal, jac, _ = interval_derivatives(c, state, node)
    feet = np.flatnonzero(state.contacts)
    selection = np.zeros((24, 18)); selection[model.v_indices, np.arange(18)] = 1.
    independent = condense_implicit_drive(snap['mass_matrix'], snap['nonlinear'],
        selection, c.qp.interface.kp, c.qp.interface.kd,
        snap['foot_jacobian'][feet].reshape(-1, 24),
        (snap['foot_jacobian_dot'][feet] @ state.v).reshape(-1),
        state.v, state.joint_position, substeps=16,
        contact_velocity_decay_s=c.qp.settings.contact_velocity_decay_s)
    for k, reference in enumerate(independent):
        for name, section in [('velocity', slice(0, 24)),
                              ('joint_position', slice(24, 42)),
                              ('force', slice(66, 78)), ('torque', slice(78, 96))]:
            np.testing.assert_allclose(jac[k, section, 48:], reference[name+'_map'],
                                       rtol=2e-5, atol=1e-5)
            np.testing.assert_allclose(nominal[k, section],
                reference[name+'_map'] @ node['command'] + reference[name+'_bias'], atol=1e-8)


def test_multistep_directional_derivative_matches_refreshed_dynamics():
    c, state, profile = setup_profile(); saved = copy.deepcopy(state)
    basis = knot_basis(4, 2)
    direction = np.random.default_rng(21).normal(0, .01, (2, 18))
    rows = list(affine_differentiated_sequence(c, state, basis, profile))
    responses = [];configuration_responses=[]
    epsilon = 1e-3
    for sign in [-1., 1.]:
        following = copy.deepcopy(state); velocities = [];configurations=[]
        for tick, node in enumerate(profile):
            following.contacts = node['state'].contacts.copy()
            command = node['command'] + sign * epsilon * (basis[tick] @ direction)
            outcome = predict_implicit_support(c.qp.model, c.qp.interface, following,
                command, substeps=16,
                contact_velocity_decay_s=c.qp.settings.contact_velocity_decay_s)
            assert outcome['diagnostics']['maximum_active_set_iterations'] == 1
            following = outcome['state']; velocities.append(following.v.copy())
            import pinocchio as pin
            configurations.append(pin.difference(c.qp.model.model,node['following'].q,following.q))
        responses.append(np.asarray(velocities))
        configuration_responses.append(np.asarray(configurations))
    actual = (responses[1] - responses[0]) / (2 * epsilon)
    for tick in range(4):
        predicted = rows[(tick+1)*16-1]['velocity_map'] @ direction.ravel()
        # Central-difference and manifold sensitivities have finite precision;
        # bound the error relative to the independently propagated response.
        assert np.max(abs(predicted-actual[tick])) < 2e-3 * max(np.max(abs(actual[tick])), 1e-3)
        actual_q=(configuration_responses[1][tick]-configuration_responses[0][tick])/(2*epsilon)
        predicted_q=rows[(tick+1)*16-1]['endpoint_configuration_map']@direction.ravel()
        assert np.max(abs(predicted_q-actual_q))<2e-3*max(np.max(abs(actual_q)),1e-3)
    np.testing.assert_array_equal(state.q, saved.q)
    np.testing.assert_array_equal(state.v, saved.v)
    np.testing.assert_array_equal(state.contacts, saved.contacts)
