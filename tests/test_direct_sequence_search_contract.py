import numpy as np

from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import (
    PureControlStateDrivenLongMPCV1QPContinuityRepair,
)


def test_direct_search_contract_is_frozen_and_full18():
    cls = PureControlStateDrivenLongMPCV1QPContinuityRepair
    assert cls.direct_search_nodes == (0, 8, 16, 24, 31)
    assert cls.horizon_steps == 32
    assert cls.direct_search_samples == 8
    assert cls.direct_search_iterations == 1
    assert cls.direct_search_seed == 20261111
    obj = cls.__new__(cls)
    directions_a = obj._direct_search_directions()
    directions_b = obj._direct_search_directions()
    assert directions_a.shape == (8, 5, 18)
    np.testing.assert_array_equal(directions_a, directions_b)
    np.testing.assert_allclose(np.linalg.norm(directions_a, axis=2), 1., atol=1e-12)


def test_direct_search_interpolation_has_32_nodes_and_keeps_bounds():
    cls = PureControlStateDrivenLongMPCV1QPContinuityRepair
    controls = np.arange(5 * 18, dtype=float).reshape(5, 18) / 100.
    seq = cls._sequence_interpolate(controls, cls.direct_search_nodes, 32)
    assert seq.shape == (32, 18)
    np.testing.assert_array_equal(seq[0], controls[0])
    np.testing.assert_array_equal(seq[8], controls[1])
    np.testing.assert_array_equal(seq[16], controls[2])
    np.testing.assert_array_equal(seq[24], controls[3])
    np.testing.assert_array_equal(seq[31], controls[4])
    assert np.all(seq >= controls.min(axis=0) - 1e-12)
    assert np.all(seq <= controls.max(axis=0) + 1e-12)


def test_direct_search_score_keeps_head_and_nonfoot_clearance_separate():
    cls = PureControlStateDrivenLongMPCV1QPContinuityRepair
    obj = cls.__new__(cls)

    class Predicted:
        root_rotation_world = np.eye(3)
        v = np.zeros(6)
        contacts = np.ones(4, dtype=bool)

    predicted = Predicted()
    sequence = np.zeros((32, 18))
    previous = np.zeros(18)
    head_risk = obj._direct_search_score(predicted, sequence, previous, .08,
                                         min_head_gap=.0, min_nonfoot_gap=.08)
    nonfoot_risk = obj._direct_search_score(predicted, sequence, previous, .08,
                                            min_head_gap=.08, min_nonfoot_gap=.0)
    assert head_risk == nonfoot_risk
    safe = obj._direct_search_score(predicted, sequence, previous, .08,
                                    min_head_gap=.08, min_nonfoot_gap=.08)
    assert safe < head_risk
