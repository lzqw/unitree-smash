"""Measured touchdown/contact hysteresis for landing support controllers."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TouchdownContactSettingsV1:
    engage_force_n: float = 5.0
    release_force_n: float = 1.0
    release_ticks: int = 3

    def __post_init__(self):
        if (isinstance(self.engage_force_n, bool) or
                not np.isfinite(self.engage_force_n) or self.engage_force_n < 0.):
            raise ValueError('engage_force_n must be finite and nonnegative')
        if (isinstance(self.release_force_n, bool) or
                not np.isfinite(self.release_force_n) or self.release_force_n < 0. or
                self.release_force_n > self.engage_force_n):
            raise ValueError('release_force_n must be between zero and engage_force_n')
        if (isinstance(self.release_ticks, bool) or not isinstance(self.release_ticks, int) or
                self.release_ticks < 1):
            raise ValueError('release_ticks must be a positive integer')


class TouchdownContactStateV1:
    def __init__(self, settings=TouchdownContactSettingsV1()):
        if not isinstance(settings, TouchdownContactSettingsV1):
            raise ValueError('settings must be TouchdownContactSettingsV1')
        self.settings = settings
        self.latched = np.zeros(4, dtype=bool)
        self.release_count = np.zeros(4, dtype=np.int32)

    def reset(self):
        self.latched[:] = False
        self.release_count[:] = 0

    def update(self, geometric_contacts, force_world=None):
        geometric = np.asarray(geometric_contacts, dtype=bool)
        if geometric.shape != (4,):
            raise ValueError('geometric_contacts must have shape (4,)')
        force = None if force_world is None else np.asarray(force_world, dtype=np.float64)
        if force is not None and (force.shape != (4, 3) or not np.isfinite(force).all()):
            raise ValueError('force_world must have shape (4,3) and be finite')
        for i in range(4):
            if not geometric[i]:
                self.latched[i] = False
                self.release_count[i] = 0
                continue
            if force is None or self.settings.engage_force_n <= 0.:
                self.latched[i] = True
                self.release_count[i] = 0
                continue
            upward = float(force[i, 2])
            if upward >= self.settings.engage_force_n:
                self.latched[i] = True
                self.release_count[i] = 0
            elif self.latched[i] and upward > self.settings.release_force_n:
                self.release_count[i] = 0
            elif self.latched[i]:
                self.release_count[i] += 1
                if self.release_count[i] >= self.settings.release_ticks:
                    self.latched[i] = False
                    self.release_count[i] = 0
            else:
                self.release_count[i] = 0
        return self.latched.copy()


__all__ = ['TouchdownContactSettingsV1', 'TouchdownContactStateV1']
