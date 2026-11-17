import json
from types import SimpleNamespace

import numpy as np

from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _fixture(vertical_velocity=5.0):
    interface = json.loads(
        (ROOT / "reports/codex_teacher_joint_eval_20261116/controller.json").read_text()
    )["interface"]
    authored = json.loads(
        (ROOT / "reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json").read_text()
    )["defaults_before_neutral_arm_override"]
    joint_order = json.loads(
        (ROOT / "configs/go2_piper_racket_landing_model_v1.json").read_text()
    )["joint_order"]
    default = np.asarray([authored[name] for name in joint_order], dtype=np.float64)
    default[12:] = json.loads(
        (ROOT / "configs/full_racket_reset_v4.json").read_text()
    )["arm_joint_positions_rad"]
    # The public racket point is [.4, 0, 1.2] in the identity body frame,
    # placing the reconstructed root at [.7, 0, .6].  A positive vertical
    # speed makes this an airborne prefix; no support force is fabricated.
    public = np.zeros(794, dtype=np.float64)
    public[6:9] = [0.0, 0.0, -1.0]
    public[760:763] = [0.4, 0.0, 1.2]
    public[763:766] = [1.0, 0.0, 0.0]
    public[:3] = [0.0, 0.0, vertical_velocity]
    return public, default, interface


def test_fullbody_evaluator_accepts_airborne_prefix_with_measured_contacts_only():
    public, default, interface = _fixture()
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface, substeps=1)
    q = np.repeat(default[None, 12:], 31, axis=0)
    qd = np.zeros_like(q)
    result = evaluator(q, qd, 30, {"source": "cpu_test"})
    assert result["accepted"] is True
    assert result["nodes_checked"] == 31
    assert result["airborne_ground_force_policy"] == "measured_contacts_only"
    assert result["diagnostics"]["normal_force_violation_n"] == 0.0


def test_fullbody_evaluator_accepts_explicit_loaded_state_without_public_reconstruction():
    public, default, interface = _fixture()
    model = LoadedPinocchioModelV1()
    state = model.state_from_public(public, default)
    evaluator = ContactFirstFullBodyEvaluatorV1(
        interface_config=interface, initial_state=state, model=model, substeps=1)
    q = np.repeat(default[None, 12:], 4, axis=0)
    qd = np.zeros_like(q)
    result = evaluator(q, qd, 3, {"source": "explicit_loaded_state"})
    assert result["accepted"] is True
    assert result["nodes_checked"] == 4


def test_fullbody_evaluator_from_native_reads_live_tensor_contract():
    _, default, interface = _fixture()
    import torch

    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.7, 0.0, 0.6]], dtype=torch.float64),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        root_com_ang_vel_w=torch.zeros((1, 3), dtype=torch.float64),
        root_com_lin_vel_w=torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64),
        body_com_pos_b=torch.zeros((1, 1, 3), dtype=torch.float64),
        joint_pos=torch.as_tensor(default[None], dtype=torch.float64),
        joint_vel=torch.zeros((1, 18), dtype=torch.float64),
    )
    env = SimpleNamespace(
        robot=SimpleNamespace(data=data),
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3), dtype=torch.float64)),
        joint_ids=np.arange(18),
        observed_foot_contact=torch.zeros((1, 4), dtype=torch.bool),
    )
    evaluator = ContactFirstFullBodyEvaluatorV1.from_native(
        env, 0, interface, substeps=1)
    q = np.repeat(default[None, 12:], 4, axis=0)
    result = evaluator(q, np.zeros_like(q), 3, {"source": "fake_native_tensors"})
    assert result["accepted"] is True
    assert result["airborne_ground_force_policy"] == "measured_contacts_only"


def test_fullbody_evaluator_rejects_prefix_that_enters_authored_ground():
    public, default, interface = _fixture(vertical_velocity=0.0)
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface, substeps=1)
    q = np.repeat(default[None, 12:], 31, axis=0)
    qd = np.zeros_like(q)
    result = evaluator(q, qd, 30, {"source": "cpu_test"})
    assert result["accepted"] is False
    assert result["rejection"] == "full_body_prefix_violation"
    assert "nonfoot_ground_violation_m" in result["violations"]


def test_fullbody_evaluator_rejects_arm_velocity_above_original_limit():
    public, default, interface = _fixture()
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface, substeps=1)
    q = np.repeat(default[None, 12:], 3, axis=0)
    qd = np.zeros_like(q)
    qd[1, 0] = interface["velocity"][12] + 1.0
    result = evaluator(q, qd, 2, {"source": "cpu_test"})
    assert result["accepted"] is False
    assert result["rejection"] == "physical_arm_velocity_limit"


def test_fullbody_evaluator_can_check_actual_racket_terminal_state():
    public, default, interface = _fixture()
    evaluator = ContactFirstFullBodyEvaluatorV1(
        public, default, interface, substeps=1,
        racket_measure_fn=lambda state: (
            np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.zeros(3)))
    q = np.repeat(default[None, 12:], 4, axis=0)
    metadata = {
        "target_position_world_m": np.array([0., 0., 0.]),
        "target_normal_world": np.array([0., 0., 1.]),
        "target_velocity_world_m_s": np.zeros(3),
    }
    result = evaluator(q, np.zeros_like(q), 3, metadata)
    assert result["accepted"] is True
    assert result["actual_racket_position_error_m"] == 0.

    rejected = ContactFirstFullBodyEvaluatorV1(
        public, default, interface, substeps=1,
        racket_measure_fn=lambda state: (
            np.array([.2, 0., 0.]), np.array([0., 0., 1.]), np.zeros(3)))
    result = rejected(q, np.zeros_like(q), 3, metadata)
    assert result["accepted"] is False
    assert result["rejection"] == "actual_racket_terminal_residual"


def test_contact_prefix_does_not_apply_extra_one_tick_target_slew_by_default():
    public, default, interface = _fixture()
    q = np.repeat(default[None, 12:], 4, axis=0)
    q[1:, 0] += 0.10
    qd = np.zeros_like(q)
    # This target step is outside the historical software envelope, but the
    # replay remains within the original physical position/velocity/torque
    # checks.  The contact-first evaluator must not reject it by default.
    unrestricted = ContactFirstFullBodyEvaluatorV1(
        public, default, interface, substeps=1)
    result = unrestricted(q, qd, 3, {"source": "contact_prefix"})
    assert result["accepted"] is True
    restricted = ContactFirstFullBodyEvaluatorV1(
        public, default, interface, substeps=1,
        enforce_target_admission_bounds=True)
    restricted_result = restricted(q, qd, 3, {"source": "legacy_envelope"})
    assert restricted_result["accepted"] is False
    assert restricted_result["rejection"] == "native_admitted_target_bounds"


def test_explicit_leg_sequence_changes_fullbody_prediction_without_aliasing():
    public, default, interface = _fixture()
    arm = np.repeat(default[None, 12:], 6, axis=0)
    legs = np.repeat(default[None, :12], 6, axis=0)
    legs[:, 0] += np.linspace(0., .08, 6)
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface, substeps=1,
        leg_target_sequence_rad=legs, record_state_trace=True)
    frozen_legs = legs.copy()
    # Caller mutation after construction cannot alter a candidate.
    legs[:] = 999.
    result = evaluator(arm, np.zeros_like(arm), 5, {})
    repeated = evaluator(arm, np.zeros_like(arm), 5, {})
    held = ContactFirstFullBodyEvaluatorV1(public, default, interface, substeps=1,
        record_state_trace=True)(arm, np.zeros_like(arm), 5, {})
    assert result['accepted'] and held['accepted']
    assert result == repeated
    assert result['leg_reference'] == 'explicit_sequence'
    np.testing.assert_allclose(np.array([r['final_target_rad'][:12]
        for r in result['state_trace']]), frozen_legs[:5])
    # Full inertia couples leg motion back into the floating base.
    assert np.linalg.norm(np.array(result['state_trace'][-1]['v'])[:6] -
                          np.array(held['state_trace'][-1]['v'])[:6]) > 1e-6
    assert result['replayed_command_sha256'] != held['replayed_command_sha256']
    assert all(not any(r['contacts']) for r in result['state_trace'])


def test_explicit_leg_sequence_rejects_missing_nodes_and_original_limit_violation():
    import pytest
    public, default, interface = _fixture()
    legs = np.repeat(default[None, :12], 2, axis=0)
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface,
        leg_target_sequence_rad=legs)
    arm = np.repeat(default[None, 12:], 4, axis=0)
    assert evaluator(arm, np.zeros_like(arm), 3, {})['rejection'] == 'incomplete_leg_target_sequence'
    legs[1, 0] = interface['limits'][0][1] + .01
    with pytest.raises(ValueError, match='original position limits'):
        ContactFirstFullBodyEvaluatorV1(public, default, interface,
            leg_target_sequence_rad=legs)


def test_contact_terminal_uses_exact_number_of_control_intervals():
    public, default, interface = _fixture()
    evaluator = ContactFirstFullBodyEvaluatorV1(public, default, interface,
        substeps=1, record_state_trace=True)
    arm = np.repeat(default[None, 12:], 4, axis=0)
    result = evaluator(arm, np.zeros_like(arm), 3, {})
    assert result['accepted']
    assert result['prediction_intervals'] == 3
    assert result['terminal_elapsed_s'] == .03
    assert [row['elapsed_s'] for row in result['state_trace']] == [.01, .02, .03]
    # Independent replay: at t=.03 only targets at t=0,.01,.02 were applied.
    from scripts.implicit_support_prediction_v1 import predict_implicit_support
    from copy import deepcopy
    state = deepcopy(evaluator.initial_state)
    for _ in range(3):
        state = predict_implicit_support(evaluator.model, evaluator.interface,
            state, default, duration_s=.01, substeps=1)['state']
    np.testing.assert_allclose(result['state_trace'][-1]['q'], state.q, atol=1e-12)


def test_unilateral_takeoff_releases_measured_feet_without_fictitious_force():
    public, default, interface = _fixture(vertical_velocity=5.)
    model = LoadedPinocchioModelV1()
    state = model.state_from_public(public, default)
    # Measured support at the beginning of an upward separating interval.
    state.contacts[:] = [True, False, False, False]
    evaluator = ContactFirstFullBodyEvaluatorV1(interface_config=interface,
        initial_state=state, model=model, substeps=1,
        contact_prediction='unilateral_release', record_state_trace=True)
    arm = np.repeat(default[None,12:], 4, axis=0)
    result = evaluator(arm, np.zeros_like(arm), 3, {})
    assert result['accepted']
    assert result['contact_modes_changed']
    assert not result['new_contact_candidates_added']
    assert result['diagnostics']['motor_torque_violation_nm'] == 0.
    for row in result['state_trace']:
        assert not any(row['contacts'])
        np.testing.assert_allclose(row['foot_forces_world_n'], np.zeros((4,3)), atol=1e-12)
    assert evaluator.initial_state.contacts.tolist() == [True, False, False, False]


def test_four_foot_release_warm_start_fixes_original_active_set_failure():
    public, default, interface = _fixture(vertical_velocity=5.)
    model = LoadedPinocchioModelV1()
    state = model.state_from_public(public, default)
    state.contacts[:] = True
    evaluator = ContactFirstFullBodyEvaluatorV1(interface_config=interface,
        initial_state=state, model=model, substeps=1,
        contact_prediction='unilateral_release')
    arm = np.repeat(default[None,12:], 4, axis=0)
    result = evaluator(arm, np.zeros_like(arm), 3, {})
    assert result['accepted']
    assert result['contact_modes_changed']
    assert result['diagnostics']['dynamics_residual_max'] < 2e-5
    from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
    import pytest
    trace=[]
    with pytest.raises(ValueError,match='did not converge'):
        predict_unilateral_support(model,evaluator.interface,state,default,substeps=1,
            separating_warm_start=False,iteration_trace=trace)
    assert len(trace)==80
    assert all(np.isfinite(r['linear_residual']) for r in trace)


def test_released_feet_remain_force_free_across_internal_substeps():
    from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
    from copy import deepcopy
    public, default, interface = _fixture(vertical_velocity=5.)
    model=LoadedPinocchioModelV1();state=model.state_from_public(public,default)
    state.contacts[:]=True
    free=deepcopy(state);free.contacts[:]=False
    evaluator=ContactFirstFullBodyEvaluatorV1(interface_config=interface,initial_state=state,model=model)
    trace=[]
    released=predict_unilateral_support(model,evaluator.interface,state,default,substeps=4,iteration_trace=trace)
    ballistic=predict_unilateral_support(model,evaluator.interface,free,default,substeps=4)
    np.testing.assert_allclose(released['state'].q,ballistic['state'].q,atol=1e-12)
    np.testing.assert_allclose(released['state'].v,ballistic['state'].v,atol=1e-12)
    assert all(not any(t['modes']) for t in trace)
    assert all(np.max(np.abs(t['forces_n'])) == 0. for t in trace)


def test_physical_measurement_uses_selected_authored_contact_metadata():
    public,default,interface=_fixture()
    seen=[]
    def measure(state,meta):
        seen.append(meta['contact_point_local'])
        return np.array([.1,0.,0.]),np.array([1.,0.,0.]),np.array([1.,0.,0.])
    evaluator=ContactFirstFullBodyEvaluatorV1(public,default,interface,substeps=1,
        racket_measure_fn=measure,measurement_uses_metadata=True)
    meta=dict(contact_point_local=[.0006,0.,.07],target_position_world_m=[.1,0.,0.],
              target_normal_world=[1.,0.,0.],target_velocity_world_m_s=[1.,0.,0.])
    q=np.repeat(default[None,12:],3,axis=0)
    result=evaluator(q,np.zeros_like(q),2,meta)
    assert seen==[meta['contact_point_local']]
    assert result['accepted'],result


def test_native_timestep_profile_is_explicit_and_not_physx_equivalence():
    public,default,interface=_fixture()
    evaluator=ContactFirstFullBodyEvaluatorV1(public,default,interface,substeps=160,
        contact_prediction='unilateral_release',enforce_joint_stops=True)
    q=np.repeat(default[None,12:],2,axis=0)
    result=evaluator(q,np.zeros_like(q),1,{})
    assert result['accepted']
    contract=result['prediction_contract']
    assert contract['integration_substeps']==160
    assert contract['integration_dt_s']==.0000625
    assert contract['joint_position_stops']
    assert not contract['physx_equivalent']
    assert contract['velocity_limit_response']=='postsolve_rejection_only'
