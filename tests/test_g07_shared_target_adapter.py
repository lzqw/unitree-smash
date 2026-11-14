from pathlib import Path

import pytest
import torch

from quadsmash.control.g07_shared_target_residual_v1 import (
    SCHEMA,
    make_g07_residual_controller,
)
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor


CHECKPOINT = Path(
    "/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/"
    "frozen_v2_smooth_approach16_env5120_01/policy.pt"
)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="measured G07 checkpoint is not mounted")
def test_g07_checkpoint_is_strictly_loadable_on_cpu():
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert source["schema"] == SCHEMA
    actor, metadata = load_ppo_initialized_actor(CHECKPOINT, torch.device("cpu"))
    assert metadata["native_schema"] == SCHEMA
    assert metadata["obs_dim"] == 794
    assert metadata["action_dim"] == 18
    assert actor(torch.zeros(2, 794))[0].shape == (2, 18)

    controller = make_g07_residual_controller(source, 1, torch.device("cpu"))
    obs = {"policy": torch.zeros(1, 794)}
    default = torch.zeros(18)
    limits = torch.stack((-torch.ones(18), torch.ones(18)), dim=-1)
    action = controller.action(obs, default, limits)
    assert action.shape == (1, 18)
    assert torch.isfinite(action).all()
    assert torch.all(action.abs() <= 1.0 + 1e-6)

