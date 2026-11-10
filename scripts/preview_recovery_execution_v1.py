"""Side-effect-free preview of the unchanged native recovery arbiter."""
import numpy as np

from scripts.event_recovery_manager_v1 import RecoveryCommandArbiterV1


def preview_recovery_execution(limits, requested_target, previous_final_target):
    """Call the exact executor transform with branch-local copies.

    This is a preview only. The caller must still send the original request
    through the original native execution path, exactly once.
    """
    requested = np.asarray(requested_target, dtype=float).copy()
    previous = np.asarray(previous_final_target, dtype=float).copy()
    final, mapping = RecoveryCommandArbiterV1(np.asarray(limits).copy()).select(
        None, requested, previous, owns_recovery=True, dt_s=.01)
    return final.copy(), mapping
