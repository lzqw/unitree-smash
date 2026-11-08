"""Use physical control ticks rather than float32 floor for reference indexing.

The environment runs at 100 Hz. Its float32 episode clock can lie slightly
below an exact control boundary; flooring that value repeats the prior frame.
Rounding to the nearest control boundary preserves elapsed-time progression,
including skipped observations, without replaying a stale reference.
"""
import math
from scripts.pure_control_long_trajectory_recovery_v11 import PureControlLongTrajectoryRecoveryV11


class PureControlReferenceClockV12(PureControlLongTrajectoryRecoveryV11):
    def _reference_time(self):
        if self.chosen is None:
            return None, 0.
        elapsed = float(self.elapsed)
        if not math.isfinite(elapsed) or elapsed < -1e-6:
            raise ValueError('Finite nonnegative physical elapsed time required')
        tick = int(round(elapsed/.01))
        if abs(elapsed-tick*.01) > .0025:
            raise ValueError('Reference caller is not on the declared 100 Hz control grid')
        duration_ticks = len(self.chosen['trajectory'])
        if tick >= duration_ticks:
            return None, 0.
        return tick, min(1., (duration_ticks-tick)/10.)

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        index, weight = self._reference_time()
        diagnostic.update(controller='pure_control_reference_clock_v12',
            reference_control_tick=index, reference_weight_from_ticks=weight,
            reference_clock='nearest physical 100 Hz control boundary')
        return target, diagnostic
