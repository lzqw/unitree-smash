"""Historical constant-angular-velocity base extrapolator, shared with audit."""
import numpy as np
from .contact_first_trajectory_search_v1 import rotation_current_from_future_from_rotvec


def extrapolate_base(duration,linear,angular,linear_acceleration,angular_acceleration):
    t=float(duration)
    return dict(rotation_current_from_future=rotation_current_from_future_from_rotvec(np.asarray(angular)*t),
        translation_current_m=np.asarray(linear)*t+.5*np.asarray(linear_acceleration)*t*t,
        linear_velocity_current_m_s=np.asarray(linear)+np.asarray(linear_acceleration)*t,
        angular_velocity_current_rad_s=np.asarray(angular)+np.asarray(angular_acceleration)*t)
