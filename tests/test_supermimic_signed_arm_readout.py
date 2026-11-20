import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_supermimic_signed_arm_readout import Moments, arm_rows, solve
from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1


def test_arm_errors_match_the_actual_world_transition():
    torch.manual_seed(427)
    world = PhysicalWorldV1(hidden=(10,)).eval()
    state = torch.randn(2, 3, 55)
    state[:, :, 3:7] = torch.nn.functional.normalize(state[:, :, 3:7], dim=-1)
    packed = dict(state=state, targets=torch.randn(2, 3, 18), previous_action=torch.randn(2, 3, 18),
                  feet=torch.zeros(2, 3, 4), present=torch.ones(2, 3, dtype=torch.bool),
                  elapsed=torch.zeros(2, 3), next_state=torch.randn(2, 3, 55))
    t = torch.tensor([0, 1])
    c = torch.tensor([2, 0])
    launch = torch.randn(2, 6)
    _, err, _ = arm_rows(world, packed, t, c, launch)
    predicted = world(*(packed[k][t,c] for k in ('state','targets','previous_action','feet','present','elapsed')),launch)[0]
    dims = list(range(25,31)) + list(range(43,49))
    torch.testing.assert_close(err * world.delta_scale[dims], (packed['next_state'][t,c] - predicted)[:,dims])


def test_paired_fit_reduces_signed_error_and_moments_match_direct_residual():
    gen = torch.Generator().manual_seed(83)
    x = torch.randn(32, 4, generator=gen, dtype=torch.double)
    delta = torch.randn(16, 4, generator=gen, dtype=torch.double) * .02
    truth = torch.randn(4, 3, generator=gen, dtype=torch.double)
    positive, paired = Moments(4,3), Moments(4,3)
    for stats, features in ((positive, x), (paired, delta)):
        y = features @ truth
        stats.add(features, y, y)
    step = solve(positive, paired, .1)
    torch.testing.assert_close(paired.rmse(step), ((delta @ (truth-step)).square().mean(0)).sqrt())
    assert (paired.rmse(step) < paired.rmse(torch.zeros_like(step))).all()
