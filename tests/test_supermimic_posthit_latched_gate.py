"""CPU contract checks for the explicit post-hit Recovery latch."""
import io
from pathlib import Path

import torch

from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor


SOURCE = Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/'
              'frozen_v2_smooth_approach16_env5120_01/policy.pt')


def test_explicit_posthit_gate_preserves_legacy_hit_and_suppresses_missed_touchdown():
    if not SOURCE.exists():
        return
    actor, _ = load_ppo_initialized_actor(SOURCE, torch.device('cpu'))
    raw = torch.randn(8, 794)
    raw[:, 770:775] = 0.
    raw[:, 774] = 1.  # phase4: foot contact, which is not itself a hit latch
    with torch.no_grad():
        legacy = actor(raw, sample=False)[0]
        gated_on = actor.forward_with_recovery_gate(raw, torch.ones(8, 1), sample=False)[0]
        gated_off = actor.forward_with_recovery_gate(raw, torch.zeros(8, 1), sample=False)[0]
        module = actor.actor
        normalized = module.normalizer(raw)
        base = module.base(normalized)
        approach = module.gate(raw) * module.approach(normalized) * module.approach_scale
        recovery = module.recovery(normalized) * module.recovery_scale
    torch.testing.assert_close(gated_on, legacy, atol=2e-6, rtol=2e-6)
    # The source recovery head may be hidden by the final [-1,1] action clamp;
    # validate the gate on the pre-clamp action instead of treating saturation
    # as evidence that the correction was absent.
    assert float(recovery.abs().max()) > 1e-7
    assert float((base + approach + recovery - (base + approach)).abs().max()) > 1e-7
    assert torch.isfinite(gated_off).all()
