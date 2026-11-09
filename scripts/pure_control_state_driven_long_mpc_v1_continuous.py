"""Fixed 0.32 s long MPC with a short replan handoff transition.

This ablation keeps the best v1 horizon and 16-tick replanning cadence.  When
an updated plan replaces a cached plan, its first few commands are blended
with the still-pending old commands so that a replan cannot introduce a large
target discontinuity.  The blend is only a command handoff aid; all physical
limits and native execution remain unchanged.
"""
import numpy as np

from scripts.pure_control_state_driven_long_mpc_v1 import PureControlStateDrivenLongMPCV1


class PureControlStateDrivenLongMPCV1Continuous(PureControlStateDrivenLongMPCV1):
    replan_stride = 16
    handoff_steps = 4

    def _handoff(self, old_remaining, new_sequence):
        if not old_remaining or not new_sequence or self.handoff_steps <= 0:
            return new_sequence
        count = min(self.handoff_steps, len(old_remaining), len(new_sequence))
        merged = [np.asarray(x, dtype=float).copy() for x in new_sequence]
        for index in range(count):
            alpha = float(index + 1) / float(count + 1)
            merged[index] = ((1.0 - alpha) * np.asarray(old_remaining[index], dtype=float)
                             + alpha * merged[index])
        return merged

    def command(self, state, phase, time_s):
        # Capture commands still pending from the prior plan before command()
        # resets the cursor or replaces the plan.
        old_remaining = []
        phase_changed = phase != self._last_phase
        need_replan = (not self._plan or self._plan_cursor >= len(self._plan) or
                       self._tick - self._last_plan_tick >= self.replan_stride or phase_changed)
        if need_replan and self._plan and not phase_changed:
            old_remaining = [x.copy() for x in self._plan[self._plan_cursor:]]

        target, diagnostics = super().command(state, phase, time_s)
        if diagnostics.get('replanned') and diagnostics.get('solver_accepted') and old_remaining:
            # The parent has already selected the first new command.  Replace
            # the cached plan with a blended handoff while preserving the
            # command selected for this tick as the first output.
            new_plan = [target.copy()]
            if self._plan_cursor < len(self._plan):
                new_plan.extend(x.copy() for x in self._plan[self._plan_cursor:])
            # Parent cursor is one past the emitted target.
            merged = self._handoff(old_remaining, new_plan)
            self._plan = merged
            self._plan_cursor = 1
            target = merged[0].copy()
        diagnostics = dict(diagnostics)
        diagnostics.update(controller='pure_control_state_driven_long_mpc_v1_continuous',
                           handoff_steps=self.handoff_steps,
                           reference_library_required=False,
                           train_case_required=False)
        return target, diagnostics


__all__ = ['PureControlStateDrivenLongMPCV1Continuous']
