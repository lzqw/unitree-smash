"""TRAIN-only shaping for an actual airborne attack's first landing.

This term changes neither rally state nor acceptance. Slow target hits earn no
bonus; near misses after legal attacks receive bounded, smooth feedback. The
existing complete-success bonus still requires the whole chain at timeout.
"""
import math

import torch


REWARD_VERSION = 'native-full-chain-residual-v2-joint-placement'


def placement_settings(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'weight', 'distance_scale_m'}:
        raise ValueError('Declare exactly weight and distance_scale_m for joint placement')
    for key, low, high in [('weight', 0., 150.), ('distance_scale_m', 1e-4, 10.)]:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            raise ValueError('Finite numeric joint placement settings required')
        if not (low < number <= high):
            raise ValueError('Joint placement setting outside its declared TRAIN range: ' + key)
    return {key: float(value[key]) for key in ['weight', 'distance_scale_m']}


class MeasuredJointPlacementRewardV4:
    def __init__(self, count, device, court, settings):
        settings = placement_settings(settings)
        if settings is None:
            raise ValueError('Explicit settings required to enable joint placement')
        self.weight = settings['weight']
        self.scale = settings['distance_scale_m']
        self.x_low, self.x_high = map(float, court['target_x_limits_m'])
        self.y_half = float(court['target_y_half_width_m'])
        if (not all(math.isfinite(x) for x in [self.x_low, self.x_high, self.y_half])
                or self.x_low >= self.x_high or self.y_half <= 0):
            raise ValueError('Use finite original target geometry')
        self.seen_ground = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        if ids is None:
            self.seen_ground.zero_()
        else:
            self.seen_ground[ids] = False

    def step(self, *, ground_seen, legal_air, attack, crossed, faulted, first_landing):
        for flag in [ground_seen, legal_air, attack, crossed, faulted]:
            if flag.shape != self.seen_ground.shape or flag.dtype != torch.bool:
                raise ValueError('Require one measured boolean per lane')
        if first_landing.shape != (len(self.seen_ground), 3):
            raise ValueError('Require measured first-landing xyz per lane')
        finite = torch.isfinite(first_landing).all(1)
        safe = torch.where(finite[:, None], first_landing, torch.zeros_like(first_landing))
        x, y = safe[:, 0], safe[:, 1]
        dx = (self.x_low - x).clamp(min=0) + (x - self.x_high).clamp(min=0)
        dy = (y.abs() - self.y_half).clamp(min=0)
        proximity = torch.exp(-(dx.square() + dy.square()) / self.scale**2)
        eligible = (ground_seen & (~self.seen_ground) & finite & legal_air
                    & attack & crossed & (~faulted))
        # Observe the first ground event even if it was ineligible. Later
        # changes to flags or a post-rally bounce cannot create a bonus.
        self.seen_ground |= ground_seen
        return torch.where(eligible, self.weight * proximity, torch.zeros_like(proximity))
