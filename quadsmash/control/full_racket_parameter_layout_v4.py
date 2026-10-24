"""Canonical parameter order shared by native search and CPU bank preflight."""
from .full_racket_teacher_v4 import PARAMETER_NAMES
from .full_racket_arm_search_v4 import VELOCITY_PARAMETER_NAMES
from .full_racket_recovery_search_v4 import RECOVERY_PARAMETER_NAMES
from .full_racket_contact_normal_search_v4 import NORMAL_PARAMETER_NAMES


def controller_parameter_names(*, search_arm_velocity=False, search_recovery_parameters=False,
                               search_contact_normal=False):
    if search_recovery_parameters and not search_arm_velocity:
        raise ValueError('Recovery37layout requires arm-velocity32coordinates')
    if search_contact_normal and not search_recovery_parameters:
        raise ValueError('Contact-normal38layout requires recovery37coordinates')
    return (PARAMETER_NAMES + ['intercept_plane_x', 'strike_timing_residual']
            + (VELOCITY_PARAMETER_NAMES if search_arm_velocity else [])
            + (RECOVERY_PARAMETER_NAMES if search_recovery_parameters else [])
            + (NORMAL_PARAMETER_NAMES if search_contact_normal else []))
