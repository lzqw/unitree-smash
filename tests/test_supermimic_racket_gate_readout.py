import torch

from scripts.calibrate_supermimic_ball_event_logits import fit_racket_readout


def test_fitted_racket_row_reproduces_report_through_full_five_mode_softmax():
    generator = torch.Generator().manual_seed(177)
    weights = torch.tensor([1., -.5, .7])
    groups, hidden = {}, {}
    slope, offset = 2., -.4
    for name in ('pre', 'edge', 'ordinary', 'post'):
        x = torch.randn(12, 3, generator=generator)
        logits = torch.randn(12, 5, generator=generator)
        response = torch.randn(12, 4, 3, generator=generator)
        scaled = slope*logits
        scaled[:, 1] += offset + x @ weights
        target = (scaled.softmax(-1)[:,1:,None]*response).sum(1)
        groups[name] = (logits, response, target)
        hidden[name] = x
    with torch.no_grad():
        row, stats = fit_racket_readout(groups, hidden, slope, offset)
    assert row.norm() > 0
    for name, (logits, response, target) in groups.items():
        scaled = slope*logits
        scaled[:, 1] += offset + hidden[name] @ row
        error = ((scaled.softmax(-1)[:,1:,None]*response).sum(1)-target).square().mean().sqrt()
        torch.testing.assert_close(error, torch.tensor(stats['after_rmse_mps'][name]))
