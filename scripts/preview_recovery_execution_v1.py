"""Side-effect-free preview of the unchanged native recovery arbiter."""
import numpy as np

from scripts.event_recovery_manager_v1 import RecoveryCommandArbiterV1
from scripts.full_authority_position_interface_v1 import FullAuthorityCommandArbiterV1


def preview_recovery_execution(limits, requested_target, previous_final_target,
                               *, software_arm_slew=True):
    """Call the exact executor transform with branch-local copies.

    This is a preview only. The caller must still send the original request
    through the original native execution path, exactly once.
    """
    requested = np.asarray(requested_target, dtype=float).copy()
    previous = np.asarray(previous_final_target, dtype=float).copy()
    arbiter = (RecoveryCommandArbiterV1(np.asarray(limits).copy())
               if software_arm_slew else
               FullAuthorityCommandArbiterV1(np.asarray(limits).copy()))
    final, mapping = arbiter.select(None, requested, previous,
                                    owns_recovery=True, dt_s=.01)
    return final.copy(), mapping
