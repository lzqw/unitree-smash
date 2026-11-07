"""Expose each signed terminal joint constraint without changing its feasible set."""
import numpy as np
from scripts.terminal_settling_constraints_v1 import TARGETS, constraints, rank
from scripts.terminal_settling_constraints_v1 import assess as assess_v1


def component_constraints(gap, joint_velocity, omega, horizontal, vertical):
    gap = np.asarray(gap, float)
    velocity = np.asarray(joint_velocity, float)
    if gap.shape != (18,) or velocity.shape != (18,):
        raise ValueError('All 18 actual command and velocity components required')
    if not np.isfinite(np.r_[gap, velocity, omega, horizontal, vertical]).all():
        raise ValueError('Finite endpoint required')
    # ± rows preserve exactly |x| <= limit, and retain gradients for joints
    # that are not currently the largest absolute velocity.
    return np.r_[gap/TARGETS['command_preload_gap_rad']-1,
        -gap/TARGETS['command_preload_gap_rad']-1,
        velocity/TARGETS['joint_speed_rad_s']-1,
        -velocity/TARGETS['joint_speed_rad_s']-1,
        omega/TARGETS['angular_speed_rad_s']-1,
        horizontal/TARGETS['horizontal_speed_m_s']-1,
        vertical/TARGETS['vertical_speed_m_s']-1]


def assess(controller, template, result, spheres):
    assessment = assess_v1(controller, template, result, spheres)
    if not assessment['solved']:
        return assessment
    velocity = np.asarray(result['states'][-1]['v'])[controller.qp.model.v_indices]
    terminal = result['terminal']
    if not np.isclose(np.max(abs(velocity)), terminal['joint_speed'], atol=1e-10, rtol=0):
        raise ValueError('Terminal summary differs from actual predicted velocity')
    values = component_constraints(assessment['command_gap_rad'], velocity,
        terminal['omega'], terminal['horizontal'], terminal['vertical_speed'])
    maximum = max(0., float(values.max()))
    if not np.isclose(maximum, assessment['maximum_scaled_violation'], atol=1e-10, rtol=0):
        raise ValueError('Component constraints changed terminal acceptance')
    assessment.update(extra_constraint_vector=values.tolist(),
        component_joint_velocity_rad_s=velocity.tolist(), representation='signed_components_v2')
    return assessment
