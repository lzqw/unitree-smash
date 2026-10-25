"""Optional walking heading feedback from the current public task reference."""
from math import isfinite

import torch

from .initial_intercept import observed_world_heading


def validate_walk_heading_gain(value=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Walk heading gain must be a finite number in[0,4]')
    value = float(value)
    if not isfinite(value) or not 0. <= value <= 4.:
        raise ValueError('Walk heading gain must be a finite number in[0,4]')
    return value


def walk_heading_arguments(public, gain):
    """Supply the existing walker yaw channel; zero preserves its legacy call.

    The full-racket task's public command direction is world +X. Its body
    projection and gravity reconstruct heading without a simulator pose.
    Go2WalkTransfer applies its existing yaw command cap and readiness gate.
    """
    if gain == 0.:
        return {}
    heading = observed_world_heading(public, [1., 0., 0.])
    return dict(heading_error=-heading, heading_kp=torch.full_like(heading, gain))
