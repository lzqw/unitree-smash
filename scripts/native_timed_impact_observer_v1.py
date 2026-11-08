"""Passive high-rate window spanning the successful timed recovery touchdown."""
import numpy as np
from scripts.native_foot_impact_observer_v1 import NativeFootImpactObserverV1


class NativeTimedImpactObserverV1(NativeFootImpactObserverV1):
    def __init__(self, env, output):
        super().__init__(env, output)
        # Keep both case85 and an untouched comparison lane; no new sensors.
        self.first, self.last = 288*160, 300*160
        self.capacity = self.last-self.first+1
        self.frames = np.empty(self.capacity, dtype=np.int64)
