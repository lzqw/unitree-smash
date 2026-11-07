import numpy as np
from scripts.landing_arm_feasibility_mpc_v1 import ArmFeasibilityIntervalQP
from scripts.landing_active_set_interval_mpc_v1 import ActiveSetIntervalCommandSupportQP
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from test_admitted_arm_support_qp_v1 import fixture


def test_feasible_preserved_command_is_not_replaced():
    _, interface, state = fixture()
    state.previous_command_target = state.joint_position.copy()
    settings = SupportSettings()
    original = ActiveSetIntervalCommandSupportQP(interface, settings, arm_target_rate_rad_s=10.)
    candidate = ArmFeasibilityIntervalQP(interface, settings, arm_target_rate_rad_s=10.)
    old = original.solve(state, state.joint_position)
    new = candidate.solve(state, state.joint_position)
    assert old['accepted'] and new['accepted']
    assert not new['arm_feasibility_fallback_attempted']
    np.testing.assert_array_equal(new['joint_position_target'], old['joint_position_target'])


def test_infeasible_arm_preservation_can_recover_with_original_limits_and_one_reference_tick():
    _, interface, state = fixture()
    reference = state.joint_position.copy()
    reference[17] += .4
    state.previous_command_target = state.joint_position.copy()
    state.previous_command_target[17] += .3
    settings = SupportSettings()
    original = ActiveSetIntervalCommandSupportQP(interface, settings, arm_target_rate_rad_s=10.)
    candidate = ArmFeasibilityIntervalQP(interface, settings, arm_target_rate_rad_s=10.)
    fixed = original.solve(state, reference)
    assert not fixed['accepted'] and fixed['solver_status'] == 'primal infeasible'
    result = candidate.solve(state, reference)
    assert result['accepted']
    assert result['arm_feasibility_fallback']['free_accepted']
    assert candidate.preserve_arm_command is True
    # Retrying a QP is not an extra control interval.
    for key in ('reference_position', 'reference_velocity', 'support_anchors', 'previous_contacts'):
        np.testing.assert_array_equal(candidate.reference_state()[key], original.reference_state()[key])
    command = result['joint_position_target']
    assert np.max(abs(command[12:] - state.previous_command_target[12:])) <= .1 + 1e-10
    prediction = predict_implicit_support(candidate.model, candidate.interface, state, command, substeps=16)
    assert max(v for k,v in prediction['diagnostics'].items() if k != 'maximum_active_set_iterations') < 3e-4
    assert max(result['residuals'].values()) < 3e-4


def test_irrecoverable_model_constraints_remain_rejected():
    _, interface, state = fixture()
    state.q[2] = .11
    state.root_position_world[2] = .11
    state.previous_command_target = state.joint_position.copy()
    candidate = ArmFeasibilityIntervalQP(interface, SupportSettings(), arm_target_rate_rad_s=10.)
    result = candidate.solve(state, state.joint_position)
    assert not result['accepted']
    assert candidate.preserve_arm_command is True
