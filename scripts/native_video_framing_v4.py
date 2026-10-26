"""Fit orthographic views to the meshes at the actual recorded body poses.

Only camera matrices are computed here. Native meshes, body/ball motion,
contact/landing events, court dimensions and outcomes are never changed.
"""
import json
from pathlib import Path

import numpy as np

from render_current_v4_progress import pose_matrix


PANE_HEIGHT = 420
PANE_TOP = 126
MARGIN = 16


def body_mesh_corners(run):
    metadata = json.loads((Path(run) / 'geometry.json').read_text())
    grouped = {}
    with np.load(Path(run) / 'geometry.npz') as arrays:
        for item in metadata['meshes']:
            vertices = arrays[item['key'] + '_vertices']
            assert vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
            lo, hi = vertices.min(0), vertices.max(0)
            body = item['body']
            if body in grouped:
                lo, hi = np.minimum(lo, grouped[body][0]), np.maximum(hi, grouped[body][1])
            grouped[body] = (lo, hi)
    bodies = sorted(grouped)
    corners = []
    for body in bodies:
        lo, hi = grouped[body]
        corners.append([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
    return np.asarray(bodies), np.asarray(corners, dtype=float), len(metadata['meshes'])


def at_poses(poses, bodies, corners):
    poses = np.asarray(poses)
    single = poses.ndim == 2
    if single:
        poses = poses[None]
    assert poses.ndim == 3 and poses.shape[-1] == 7 and np.isfinite(poses).all()
    matrices = np.asarray([[pose_matrix(frame[body]) for body in bodies] for frame in poses])
    points = np.einsum('tbij,bkj->tbki', matrices[:, :, :3, :3], corners) + matrices[:, :, None, :3, 3]
    return points[0] if single else points


def fit(points, width, height, side, azimuth=0.):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    assert len(points) and np.isfinite(points).all()
    if side:
        tilt = np.deg2rad(12.)
        angle = np.deg2rad(azimuth)
        right = np.asarray([np.cos(angle), -np.sin(angle), 0.])
        back = np.asarray([-np.sin(angle)*np.cos(tilt), -np.cos(angle)*np.cos(tilt), np.sin(tilt)])
        basis = np.asarray([right, np.cross(back, right), back])
    else:
        basis = np.eye(3)
    aligned = points @ basis.T
    lo, hi = aligned.min(0), aligned.max(0)
    center = (lo + hi) / 2
    vertical = max(hi[1] - lo[1], 2.1 if side else 1.9)
    units = max((hi[0] - lo[0]) / (width - 2 * MARGIN), vertical / (height - 2 * MARGIN))
    distance = max(10., (hi[2] - lo[2]) / 2 + 2.)
    eye_aligned = center + [0., 0., distance]
    view = np.eye(4)
    view[:3, :3] = basis
    view[:3, 3] = -eye_aligned
    near, far = .01, distance + (hi[2] - lo[2]) / 2 + 20.
    projection = np.eye(4)
    projection[0, 0], projection[1, 1] = 2 / (units * width), 2 / (units * height)
    projection[2, 2], projection[2, 3] = -2 / (far - near), -(far + near) / (far - near)
    vp = projection @ view
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ vp.T
    ndc = homogeneous[:, :3] / homogeneous[:, 3:]
    assert np.max(np.abs(ndc)) < 1
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
    assert pixels[:, 0].min() >= MARGIN - 1e-5 and pixels[:, 0].max() <= width - MARGIN + 1e-5
    assert pixels[:, 1].min() >= MARGIN - 1e-5 and pixels[:, 1].max() <= height - MARGIN + 1e-5
    return vp, dict(points_checked=len(points), pixel_bounds_min=pixels.min(0).tolist(),
                    pixel_bounds_max=pixels.max(0).tolist(), units_per_pixel=units,
                    side_azimuth_degrees=azimuth if side else None,
                    minimum_margin_pixels=MARGIN, all_fitted_points_in_viewport=True)


def case_views(run, case, recorded, court):
    bodies, corners, meshes = body_mesh_corners(run)
    actual = at_poses(recorded['link_poses'], bodies, corners).reshape(-1, 3)
    events = []
    if case.get('contact'):
        events.append(at_poses(case['contact']['body_link_poses_local'], bodies, corners).reshape(-1, 3))
    state = case.get('last_recorded_state') or case['row']['terminal_native_state']
    terminal = np.column_stack((state['body_link_positions_m'], state['body_link_quaternions_wxyz']))
    events.append(at_poses(terminal, bodies, corners).reshape(-1, 3))
    robot = np.concatenate([actual, *events])
    # These fixed cameras cover the entire actual episode rather than following
    # or changing the robot's motion to keep a particular outcome visible.
    anchors = np.asarray([[robot[:, 0].mean(), robot[:, 1].mean(), 0.],
                          [court['net_plane_x_m'], 0., court['net_center_top_m']]])
    near = np.concatenate([robot, anchors])
    times = recorded['episode_clock_s']
    mask = times >= 2.
    if case['row']['ground_seen']:
        mask &= times <= case['row']['first_landing_time_s']
    ball = recorded['ball_position_m'][mask]
    ball = ball[np.isfinite(ball).all(1)]
    x0, x1 = court['target_x_limits_m']
    yh = court['target_y_half_width_m']
    target = np.asarray([[x, y, .002] for x in [x0, x1] for y in [-yh, yh]])
    extra = [target, ball]
    if case['row']['ground_seen']:
        landing = np.asarray(case['row']['first_landing_m'])
        assert np.isfinite(landing).all()
        extra.append(landing[None])
    wide = np.concatenate([near, *extra])
    contact = near
    if case.get('contact'):
        contact = np.concatenate([events[0], np.asarray(case['contact']['ball_position_m'])[None]])
    matrices, checks = {}, {}
    for prefix, points in [('', near), ('wide_', wide), ('court_', wide), ('contact_', contact)]:
        for name, width, side in [('side', 760, True), ('top', 504, False)]:
            matrices[prefix + name], checks[prefix + name] = fit(points, width, PANE_HEIGHT, side,
                                                               30. if prefix == 'contact_' and side else 0.)
    return matrices, dict(rendered_meshes=meshes, rendered_bodies=len(bodies),
                          actual_50Hz_samples=len(recorded['link_poses']), views=checks,
                          scope='All recorded robot mesh bounding corners, exact contact pose and last recorded pose. Viewport coverage does not prove every surface is unoccluded.')
