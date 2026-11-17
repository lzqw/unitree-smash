import json

import numpy as np

from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1


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
