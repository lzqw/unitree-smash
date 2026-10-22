import numpy as np
import pytest

from quadsmash.physics import ShuttlecockModel, ShuttlecockParams, ShuttlecockState


def test_gravity_matches_analytic_solution_without_drag():
    dt = 0.002
    model = ShuttlecockModel(ShuttlecockParams(drag_coefficient=0.0))
    initial = ShuttlecockState(np.array([0.0, 0.0, 1.0]), np.array([2.0, 0.0, 3.0]))
    final = model.rollout(initial, 0.4, dt)[-1]
    t = final.time - initial.time
    expected = initial.position + initial.velocity * t + 0.5 * np.array([0.0, 0.0, -9.81]) * t * t
    assert np.linalg.norm(final.position - expected) < 2e-4


def test_drag_reduces_speed_and_rollout_is_repeatable():
    model = ShuttlecockModel()
    initial = ShuttlecockState(np.array([0.0, 0.0, 1.0]), np.array([4.0, 0.0, 3.0]))
    a = model.rollout(initial, 0.4, 0.002)[-1]
    b = model.rollout(initial, 0.4, 0.002)[-1]
    assert np.linalg.norm(a.velocity) < np.linalg.norm(initial.velocity)
    assert np.array_equal(a.position, b.position)


def test_invalid_inputs_raise():
    assert np.array_equal(ShuttlecockState(np.zeros(3), np.zeros(3)).copy().velocity, np.zeros(3))
    with pytest.raises(ValueError):
        ShuttlecockState(np.zeros(3), np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError):
        ShuttlecockModel().step(ShuttlecockState(np.zeros(3), np.zeros(3)), 0.0)

