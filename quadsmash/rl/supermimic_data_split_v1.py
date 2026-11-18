"""Explicit whole-lane split contract shared by offline model fits and references."""
from numbers import Integral
import torch


def explicit_lane_split(payload, lane_count):
    groups = []
    for key in ('train_lanes', 'validation_lanes'):
        values = payload.get(key)
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError('Missing nonempty split: ' + key)
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
            raise ValueError('Lane identities must be integers')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate lane identity in ' + key)
        if min(values) < 0 or max(values) >= lane_count:
            raise ValueError('Lane identity out of range')
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError('Training/validation overlap')
    if groups[0] | groups[1] != set(range(lane_count)):
        raise ValueError('Explicit split must account for every lane')
    return tuple(torch.tensor(sorted(group), dtype=torch.long) for group in groups)
