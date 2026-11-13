"""Pure Torch action-contract helpers with no Isaac/PhysX imports."""
from __future__ import annotations

import torch


def decode_full_racket_normalized_action(action: torch.Tensor, joint_limits: torch.Tensor) -> torch.Tensor:
    """Decode a full-racket normalized command using FullRacketEnvV4's map."""
    if action.ndim != 2 or action.shape[-1] != 18:
        raise ValueError(f'expected [N,18] normalized action, got {tuple(action.shape)}')
    if joint_limits.ndim != 3 or joint_limits.shape[-2:] != (18, 2):
        raise ValueError(f'expected [N,18,2] limits, got {tuple(joint_limits.shape)}')
    return joint_limits[..., 0] + 0.5 * (action + 1.0) * (joint_limits[..., 1] - joint_limits[..., 0])
