import pytest
import torch

from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused FP32 CUDA kernel')
def test_fused_research_flight_matches_reference_and_preserves_derivatives():
    from quadsmash.rl.supermimic_research_flight_cuda import enable_research_flight
    from torch import nn

    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(7726)
    source = BallImpulsePriorV1(gap_bias_m=.002, normal_impulse_scale=.8).to(device).eval()
    source.wind.copy_(torch.tensor([.2, -.3, 1.], device=device))
    model = nn.Module()
    model.impulse_prior = source
    state_keys = tuple(model.state_dict())
    backend = enable_research_flight(model)
    fast = model.impulse_prior
    assert tuple(model.state_dict()) == state_keys
    assert source.calibration_manifest() == fast.calibration_manifest()
    assert len(backend['source_sha256']) == 64

    for size in (1, 4, 16, 128):
        full = torch.randn((size, 55), generator=generator, device=device) * 12
        p, v = full[:, 49:52], full[:, 52:]
        duration = full[:, 1].abs() * .0008
        duration[::3] = 0.
        duration[1::4] = .003724
        assert fast._can_fuse(p, v, duration)
        for fn, args in [('advance', (p, v, duration)), ('advance_trajectory', (p, v))]:
            for actual, expected in zip(getattr(fast, fn)(*args), getattr(source, fn)(*args)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    p, v, duration = (t[:4].detach().requires_grad_() for t in (p, v, duration))
    assert not fast._can_fuse(p, v, duration)
    def derivatives(prior):
        pos, vel = prior.advance(p, v, duration)
        return torch.autograd.grad(pos.sum() + vel.sum(), (p, v, duration))
    for actual, expected in zip(derivatives(fast), derivatives(source)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.impulse_prior = source
    with pytest.raises(ValueError, match='differs'):
        enable_research_flight(model, dict(backend, source_sha256='changed'))
