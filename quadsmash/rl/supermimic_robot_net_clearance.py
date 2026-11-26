"""Finite native net-band geometry for a differentiable robot clearance proxy."""
import itertools

import torch


def net_proxy_geometry(env, geometry):
    from quadsmash.physics.full_racket_v4 import parts, string_polygon, string_strips

    local = env.safety_points.squeeze(0)
    ids = [geometry.ids[env.robot.body_names[i]] for i in env.safety_body_ids.tolist()]
    box_points = local.new_tensor(list(itertools.product((-1., 1.), repeat=3)))
    racket = []
    config = env.racket_config
    for part in parts(config):
        if not part['collision'] or part['name'] == 'Clamp':
            continue
        if part['kind'] == 'string_prism':
            strips = string_strips(config)
            if not strips:
                polygon = local.new_tensor(string_polygon(config))
                yz = polygon.abs().amax(0).tolist()
                strips = [dict(center=[0., 0., 0.], size=[config['assumptions']['string_thickness_m'],
                                                         2. * yz[0], 2. * yz[1]])]
            for strip in strips:
                racket.append(local.new_tensor(strip['center']) + box_points * local.new_tensor(strip['size']) / 2.)
        else:
            axis = local.new_tensor(part['axis'])
            half = axis.abs() * (part['length'] / 2.) + part['radius']
            racket.append(local.new_tensor(part['center']) + box_points * half)
    ids.extend([geometry.ids['Racket']] * len(racket))
    return torch.tensor(ids, device=local.device), torch.cat((local, torch.stack(racket)))


def robot_net_clearance(geometry, proxy, court):
    """Positive gap outside the solid band. A racket can pass above the tape."""
    from quadsmash.physics.court_v4 import top_height

    ids, corners = proxy
    rotation = geometry['rotation'][:, ids]
    world = corners[None] @ rotation.transpose(-1, -2) + geometry['position'][:, ids, None]
    lo, hi = world.amin(-2), world.amax(-2)
    width = court['net_width_m'] / 2.
    closest = torch.maximum(lo[..., 1], -hi[..., 1]).clamp_min(0.).clamp_max(width)
    farthest = torch.maximum(lo[..., 1].abs(), hi[..., 1].abs()).clamp_max(width)
    low_top = top_height(closest, court)
    high_top = top_height(farthest, court)
    plane = court['net_plane_x_m']
    depth = court['net_equivalent_thickness_m'] / 2.
    dx = torch.maximum(lo[..., 0] - (plane + depth), (plane - depth) - hi[..., 0])
    dy = torch.maximum(lo[..., 1] - width, -width - hi[..., 1])
    dz = torch.maximum(lo[..., 2] - high_top, low_top - court['net_depth_m'] - hi[..., 2])
    band = torch.maximum(torch.maximum(dx, dy), dz)
    post_y = lo.new_tensor([-court['post_y_center_abs_m'], court['post_y_center_abs_m']])
    radius = court['post_radius_m']
    post_low = torch.stack((torch.full_like(post_y, plane), post_y, torch.zeros_like(post_y)), dim=-1)
    post_high = post_low + lo.new_tensor([radius, radius, court['post_height_m']])
    post_low = post_low - lo.new_tensor([radius, radius, 0.])
    post_gap = torch.maximum(lo[:, :, None] - post_high, post_low - hi[:, :, None]).amax(dim=-1)
    return torch.minimum(band, post_gap.amin(dim=-1)).amin(dim=-1)


def robot_net_cost(clearance, *, margin=.045):
    return ((margin - clearance).relu() / margin).square()


def racket_back_on_own_side(geometry, proxy, racket_id, court):
    ids, corners = proxy
    rotation = geometry['rotation'][:, ids, 0]
    x = (corners[None] * rotation[:, :, None]).sum(-1) + geometry['position'][:, ids, 0, None]
    edge = court['net_plane_x_m'] - court['net_equivalent_thickness_m'] / 2.
    return x.masked_fill(ids[None, :, None] != racket_id, -torch.inf).amax(dim=(1, 2)) < edge


def check_robot_net_clearance(before, after, *, tolerance=1.e-6):
    selected = before.isfinite()
    if not selected.any():
        return True, dict(compared=0)
    old = before[selected]
    new = after[selected]
    lost_clearance = ((old >= 0.) & (new < 0.)).sum()
    old_risk = robot_net_cost(old).mean()
    new_risk = robot_net_cost(new).mean()
    valid = bool(new.isfinite().all() and lost_clearance == 0 and new_risk <= old_risk + tolerance)
    return valid, dict(compared=len(old), new_intersections=int(lost_clearance),
                       mean_risk_before=float(old_risk),
                       mean_risk_after=float(new_risk) if new.isfinite().all() else None)
