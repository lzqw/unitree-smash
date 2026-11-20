import pytest
import torch

from quadsmash.rl.supermimic_ball_residual_v2 import (
    BallResidualV2, RacketSweepSupportedBallV2, load_ball_residual,
)


def test_sweep_only_transfers_racket_mass_to_free():
    p = torch.tensor([[.3, .4, .1, .1, .1]]).repeat(3, 1)
    gap = torch.tensor([-.002, 0., .002], requires_grad=True)
    q = RacketSweepSupportedBallV2.probability_with_support(p, gap, 0., .002)
    torch.testing.assert_close(q[:, 1], torch.tensor([.4, .2, 0.]))
    torch.testing.assert_close(q[:, :2].sum(1), p[:, :2].sum(1))
    torch.testing.assert_close(q[:, 2:], p[:, 2:])
    q[:, 1].sum().backward()
    assert gap.grad[1] < 0 and torch.isfinite(gap.grad).all()


def test_checkpoint_reproduces_sweep_and_rejects_changed_geometry():
    model = RacketSweepSupportedBallV2()
    spec = dict(geometry_hashes=model.preview.geometry_hashes, gap_bias_m=float(model.gap_bias_m),
                width_m=float(model.width_m))
    ck = dict(schema=model.schema, model=model.state_dict(), event_calibration=dict(swept_racket_support=spec))
    restored = load_ball_residual(ck)
    f = torch.zeros(2, 25)
    f[:, 6] = torch.tensor([.05, 1.])
    f[:, 9] = -10.
    assert restored.min_racket_gap(f)[0] < 0 < restored.min_racket_gap(f)[1]
    pred = restored(f, torch.tensor([True, False]))[0]
    torch.testing.assert_close(pred, model(f, torch.tensor([True, False]))[0])
    assert (pred[1] == 0).all()
    spec['geometry_hashes'] = {}
    with pytest.raises(ValueError, match='geometry'):
        load_ball_residual(ck)
    old = BallResidualV2()
    assert type(load_ball_residual(dict(schema=old.schema, model=old.state_dict()))) is BallResidualV2
