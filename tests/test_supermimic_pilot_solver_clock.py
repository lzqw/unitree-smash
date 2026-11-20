import torch

from scripts.calibrate_supermimic_pilot_oracle import solver_first
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def test_solver_uses_step_start_and_indexes_the_impulse_at_step_end():
    prior = BallImpulsePriorV1()
    # Same flat patch, independently imposed gaps to isolate the solver clock.
    gap = torch.tensor([[.0001, .0006, .0001], [-.0001, .0001, .0001]])
    local = torch.zeros(2, 3, 3)
    local[..., 0] = .0256
    quat = torch.zeros(2, 3, 4)
    quat[..., 0] = 1.
    velocity = torch.zeros_like(local)
    velocity[..., 0] = torch.tensor([-4., -4., 4.])
    trial = dict(gap=gap, local=local, q=quat, v=velocity,
                 linear=torch.zeros_like(local), angular=torch.zeros_like(local))
    assert solver_first(trial, prior, .0002).tolist() == [1, 2, 2]


def test_sliding_friction_requires_separating_overlap_and_slip():
    from scripts.calibrate_supermimic_pilot_oracle import sliding_speed_change
    gap = torch.tensor([-.001, .001, -.001, -.001])
    vn = torch.tensor([.5, .5, -.5, .5])
    tangent = torch.tensor([[0., 10., 0.]]).repeat(4, 1)
    tangent[-1] = 0.
    result = sliding_speed_change(gap, vn, tangent, .0000625, .16)
    assert 0 < result[0] <= 2/7 * 10.
    torch.testing.assert_close(result[1:], torch.zeros(3))
