"""Rotating-frame acceleration regressions for causal contact prediction."""
import torch
from quadsmash.control.full_racket_rally_controller_v4 import (
    public_world_basis_in_body, transported_body_acceleration,
)


def test_constant_world_twist_does_not_invent_acceleration_when_body_rotates():
    # Exact finite rotation, not an infinitesimal omega-cross-v approximation.
    angle = torch.tensor(.37, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    basis = torch.tensor([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=torch.float64)[None]
    old_basis = torch.eye(3, dtype=torch.float64)[None]
    public = torch.zeros(1, 794, dtype=torch.float64)
    public[:, 763:766] = basis[:, :, 0]
    public[:, 6:9] = -basis[:, :, 2]
    torch.testing.assert_close(public_world_basis_in_body(public), basis)
    for world_vector in ([1., 2., .4], [.3, -.2, 2.]):
        previous = torch.tensor([world_vector], dtype=torch.float64)
        current = torch.bmm(basis, previous[:, :, None]).squeeze(-1)
        result = transported_body_acceleration(current, previous, basis, old_basis,
                                               torch.tensor([.01]), torch.tensor([True]))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-12, rtol=0.)
        assert torch.linalg.norm((current-previous)/.01) > 1.


def test_inertial_acceleration_is_expressed_in_current_body_and_lanes_are_isolated():
    basis = torch.tensor([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
                          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=torch.float64)
    old_basis = torch.eye(3, dtype=torch.float64)[None].repeat(2, 1, 1)
    previous = torch.tensor([[1., 0., 0.], [9., 8., 7.]], dtype=torch.float64)
    acceleration_world = torch.tensor([[2., -3., -9.81], [0., 0., 0.]], dtype=torch.float64)
    current = torch.bmm(basis, (previous+.01*acceleration_world)[:, :, None]).squeeze(-1)
    result = transported_body_acceleration(current, previous, basis, old_basis,
        torch.tensor([.01, float('nan')], dtype=torch.float64), torch.tensor([True, False]))
    expected = torch.bmm(basis, acceleration_world[:, :, None]).squeeze(-1)
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()
