"""Independent CPU continuous sphere/box reference, including box rotation.

Conservative advancement uses a translation plus angular Hausdorff bound on
the actual box. It does not enlarge the physical blade to manufacture hits.
"""
import math
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sphere_box_sweep(start, end, center_start, center_end, quat_start, quat_end,
                     half_extents=(0.0075, 0.11, 0.11), radius=0.025, tolerance=1e-8):
    start, end, c0, c1, half = [np.asarray(v, dtype=float) for v in (start, end, center_start, center_end, half_extents)]
    q0, q1 = [np.asarray(v, dtype=float) for v in (quat_start, quat_end)]
    if not all(np.isfinite(v).all() for v in (start, end, c0, c1, half, q0, q1)) or radius <= 0 or np.any(half <= 0):
        raise ValueError("finite geometry and positive dimensions required")
    if np.linalg.norm(q0) < 1e-12 or np.linalg.norm(q1) < 1e-12:
        raise ValueError("nonzero quaternion required")
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    if q0 @ q1 < 0:
        q1 = -q1
    theta = math.acos(float(np.clip(q0 @ q1, -1, 1)))
    bound = np.linalg.norm((end-start)-(c1-c0)) + 2*theta*np.linalg.norm(half)
    fraction = 0.0
    for _ in range(512):
        if theta < 1e-8:
            q = (1-fraction)*q0 + fraction*q1
        else:
            q = (math.sin((1-fraction)*theta)*q0 + math.sin(fraction*theta)*q1) / math.sin(theta)
        point = start + fraction*(end-start)
        center = c0 + fraction*(c1-c0)
        local = rotation(q).T @ (point-center)
        delta = abs(local)-half
        gap = np.linalg.norm(np.maximum(delta, 0)) + min(float(max(delta)), 0) - radius
        if gap <= tolerance:
            return fraction, point, center, q
        if bound < 1e-15 or fraction >= 1:
            return None
        advance = 0.9 * gap / bound
        if fraction + advance > 1:
            return None
        fraction += max(advance, 1e-12)
    raise RuntimeError("sweep reference did not converge; classify this boundary case explicitly")
