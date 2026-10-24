"""Causal preparation relative to a measured TRAIN reference, with no state writes."""
import numpy as np

from .full_racket_arm_search_v4 import bounded_prepose_step


def validate_relative_prepose(config):
    if config is None:
        return None
    reference = np.asarray(config['reference_target_body_m'], dtype=float)
    if reference.shape != (3,) or not np.isfinite(reference).all() or np.linalg.norm(reference) > 4.:
        raise ValueError('Require a finite three-dimensional reference in the robot root body frame')
    if config.get('frame') != 'robot_root_body':
        raise ValueError('Relative preparation reference must use the robot root body frame')
    clock = float(config['reference_capture_clock_s'])
    plane = float(config['reference_intercept_plane_x_m'])
    if not np.isfinite(clock) or not 2.-1e-5 <= clock <= 2.01+1e-5:
        raise ValueError('Reference must use the first control observation at or after the two-second planning trigger')
    if not np.isfinite(plane) or not .8 <= plane <= 1.45:
        raise ValueError('Reference interception plane outside the existing policy range')
    return dict(config, reference_target_body_m=reference.tolist())


def public_target_body(target_world, public):
    """Convert a forecast using the observed known command, gravity and forward.

    The fixed known world command is [1.10,0,1.80]. Its measured body-frame
    coordinates carry current root translation. The observed axes also carry
    current yaw/tilt. No root ground truth or future simulator value is read.
    """
    public, target_world = np.asarray(public, dtype=float), np.asarray(target_world, dtype=float)
    if public.ndim != 2 or public.shape[1] != 794 or target_world.shape != (len(public), 3):
        raise ValueError('Require N794 public inputs and N3 forecast positions')
    forward, up = public[:, 763:766], -public[:, 6:9]
    left = np.cross(up, forward)
    offset = target_world - np.array([1.10, 0., 1.80])
    body = public[:, 760:763] + offset[:, :1]*forward + offset[:, 1:2]*left + offset[:, 2:3]*up
    ready = np.isfinite(body).all(1) & np.isfinite(target_world).all(1)
    ready &= np.isfinite(forward).all(1) & np.isfinite(up).all(1)
    ready &= (np.linalg.norm(forward, axis=1) > .5) & (np.linalg.norm(up, axis=1) > .5)
    ready &= np.linalg.norm(left, axis=1) > .2
    return body, ready


def relative_prepose_delta(target_world, public, reference_body, position_jacobians,
                           normal_jacobians, valid, gain, lower, upper):
    """A measured reference has zero correction; other current inputs adapt.

    The returned value is a bounded position command correction. It changes
    neither physical robot state nor the ball forecast and uses no case IDs.
    """
    if not np.isfinite(gain) or not 0. <= gain <= 1.:
        raise ValueError('Relative preparation gain must be in[0,1]')
    body, ready = public_target_body(target_world, public)
    count = len(body)
    reference_body = np.asarray(reference_body, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    position_jacobians, normal_jacobians, lower, upper = [np.asarray(x) for x in
        (position_jacobians, normal_jacobians, lower, upper)]
    if reference_body.shape != (3,) or not np.isfinite(reference_body).all() or valid.shape != (count,):
        raise ValueError('Require a finite body-frame reference and N validity flags')
    if (position_jacobians.shape != (count, 3, 6) or normal_jacobians.shape != (count, 3, 6)
            or lower.shape != (count, 6) or upper.shape != (count, 6)):
        raise ValueError('Require per-lane kinematic Jacobians and six-joint correction bounds')
    result = np.zeros((count, 6))
    if gain == 0.:
        return result
    ready &= valid
    for index in np.flatnonzero(ready):
        delta = gain*(body[index]-reference_body)
        if np.array_equal(delta, np.zeros(3)):
            continue
        result[index] = bounded_prepose_step(delta, position_jacobians[index], normal_jacobians[index],
                                              lower[index], upper[index])
    return result
