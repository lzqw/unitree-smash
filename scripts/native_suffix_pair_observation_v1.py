"""Observe TRAIN3/125 contact partners at every physics frame near divergence."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.native_body_pair_observation_v1 import NativeBodyPairObservationV1


class NativeSuffixPairObservationV1(NativeBodyPairObservationV1):
    def __init__(self, env, output):
        super().__init__(env, output)
        # The completed paired microtrace first differed at frame 90570.
        # Capture every physical frame on both sides, not every fourth frame.
        self.expected_frames = np.arange(89600, 92001)
        self.allowed = set(self.expected_frames.tolist())
        self.frames = np.empty(len(self.expected_frames), dtype=np.int64)

    def save(self):
        result = super().save()
        result.update(sampling_interval_s=self.env.physics_dt,
            expected_first_frame=int(self.expected_frames[0]),
            expected_last_frame=int(self.expected_frames[-1]),
            scope='Every-physics-frame contact partner observation near TRAIN125 divergence.',
            derived_observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        path = self.output / 'body_pair_observation.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(path)
        return result
