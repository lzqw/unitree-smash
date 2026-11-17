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
