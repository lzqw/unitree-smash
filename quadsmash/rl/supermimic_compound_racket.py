"""Impulse on a spinning shuttle from multiple analytic racket collision shapes."""
import torch

from .supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from .supermimic_ball_impulse_prior_v1 import box_normal, round_normal
from .supermimic_contact_pilot_v1 import _quat_rotate, _quat_rotate_inverse


def solve_patch(normals, mu, restitution, velocity, spin, surface, radius, active,
                normal_scale, iterations, *, return_spin=False, separation_velocity=None, load_trace=None, inertia_ratio=None, friction_load_floor=None):
    """Projected normal and friction impulses in velocity units, with per-shape Coulomb bounds."""
    if friction_load_floor is not None:
        friction_load_floor = torch.where(active, friction_load_floor.clamp_min(0), 0.)
    arm = -radius * normals
    supports = normals.shape[1]
    sticking_response = (2/7) if inertia_ratio is None else (inertia_ratio/(1+inertia_ratio))[:,None]
    inertia_per_mass = .4*radius**2 if inertia_ratio is None else inertia_ratio[:,None]*radius**2
    loads = list(torch.zeros_like(mu).unbind(1))
    friction = list(torch.zeros_like(normals).unbind(1))
    linear, angular = velocity, spin
    closing = ((velocity[:, None] + torch.linalg.cross(spin[:, None].expand_as(arm), arm)
                - surface) * normals).sum(-1)
    outgoing = closing + (1 + restitution) * normal_scale * (-closing).clamp_min(0)
    if separation_velocity is not None:
        # Positive separation spends part of the interval closing the gap.
        outgoing = torch.maximum(closing, outgoing - separation_velocity)
    for _ in range(iterations):
        for j in range(supports):
            normal, lever = normals[:, j], arm[:, j]
            relative = linear + torch.linalg.cross(angular, lever) - surface[:, j]
            vn = (relative * normal).sum(-1)
            load = torch.where(active[:, j], (loads[j] + outgoing[:, j] - vn).clamp_min(0), 0.)
            linear = linear + (load - loads[j])[:, None] * normal
            loads[j] = load
            proposed = friction[j] - sticking_response * (relative - vn[:, None] * normal)
            friction_load = load if friction_load_floor is None else torch.maximum(load, friction_load_floor[:, j])
            bound = friction_load * mu[:, j]
            tangent = proposed * (bound / proposed.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)[:, None]
            change = tangent - friction[j]
            linear = linear + change
            angular = angular + torch.linalg.cross(lever, change) / inertia_per_mass
            friction[j] = tangent
    normal_loads = torch.stack(loads, 1)
    tangents = torch.stack(friction, 1)
    if load_trace is not None:
        # One discrete state per shape: unloaded, sticking, sliding.
        loaded = normal_loads > 1.e-8
        friction_load = normal_loads if friction_load_floor is None else torch.maximum(normal_loads,friction_load_floor)
        loaded = loaded | (tangents.norm(dim=-1)>1.e-8)
        sliding = tangents.norm(dim=-1) >= (mu * friction_load - 1.e-8).clamp_min(0)
        load_trace['regime'] = loaded.to(torch.uint8) * (1 + sliding.to(torch.uint8))
    total = ((normal_loads[..., None] * normals).sum(1), tangents.sum(1))
    return (*total, angular) if return_spin else total


def compound_step(prior, position, velocity, spin, center, rotation, linear, angular,
                  material, offset, normal_scale, *, compound=True, separation_gain=0., impact_geometry_fraction=0., return_impact_time=False, load_trace=None, local_mobility=None, shuttle_mass=None, temporal_solver=None, patch_inertia=None, articulation_friction_floor=False):
    """Predict one native-length substep from the incoming shuttle and current racket state."""
    geometry = prior.preview
    local = _quat_rotate_inverse(rotation, position - center)
    rounded = (geometry.racket_round_centers, geometry.racket_round_axes, geometry.racket_round_halves,
               geometry.racket_round_radii, geometry.racket_round_capsule)
    gaps = torch.cat((box_gap(local, geometry.racket_box_centers, geometry.racket_box_halves),
                      round_gap(local, *rounded)), -1) - geometry.radius
    near_count = (gaps <= offset).sum(-1)
    count = int(near_count.max()) if compound else 1
    time = position.new_zeros(len(position))
    if load_trace is not None:
        load_trace['regime'] = torch.zeros_like(gaps, dtype=torch.uint8)
        load_trace['near_shape_count'] = near_count
    if count == 0:
        result = torch.zeros_like(velocity), torch.zeros_like(velocity), spin, near_count
        return (*result, time) if return_impact_time else result
    gap, ids = gaps.topk(count, largest=False)
    normals = torch.cat((box_normal(local, geometry.racket_box_centers, geometry.racket_box_halves),
                         round_normal(local, *rounded)), 1)
    n = _quat_rotate(rotation[:, None], normals.gather(1, ids[..., None].expand(-1, -1, 3)))
    lever = position[:, None] - center[:, None] - geometry.radius * n
    surface = linear[:, None] + torch.linalg.cross(angular[:, None].expand_as(lever), lever)
    vn = ((velocity[:, None] - surface) * n).sum(-1)
    active = (gap <= offset) & (gap + prior.sub_dt * vn <= 0) & (vn < 0)
    if impact_geometry_fraction:
        if not 0 < impact_geometry_fraction <= 1:
            raise ValueError('Impact reconstruction fraction must be between zero and one')
        # Solve for the earliest incoming surface crossing within the registered substep.
        # An initial overlap begins at time zero, including when other shapes load later.
        closing_time = gap.clamp_min(0) / (-vn).clamp_min(1e-10)
        earliest = torch.where(active, closing_time, prior.sub_dt).amin(-1).clamp(max=prior.sub_dt)
        time = torch.where(active.any(-1), earliest * impact_geometry_fraction, 0.)
        angle = angular * time[:,None]
        theta = angle.norm(dim=-1, keepdim=True)
        dq = torch.cat(((theta*.5).cos(), .5*torch.sinc(theta/(2*torch.pi))*angle), -1)
        # World-frame angular velocity left-multiplies the initial racket orientation.
        impact_rotation = torch.cat((dq[:,:1]*rotation[:,:1]-(dq[:,1:]*rotation[:,1:]).sum(-1,keepdim=True),
            dq[:,:1]*rotation[:,1:] + rotation[:,:1]*dq[:,1:] + torch.linalg.cross(dq[:,1:],rotation[:,1:])), -1)
        impact_delta = position - center + time[:,None]*(velocity-linear)
        local_hit = _quat_rotate_inverse(impact_rotation, impact_delta)
        axes_hit = torch.cat((box_normal(local_hit, geometry.racket_box_centers, geometry.racket_box_halves),
                              round_normal(local_hit, *rounded)), 1)
        n = _quat_rotate(impact_rotation[:,None], axes_hit.gather(1, ids[...,None].expand(-1,-1,3)))
        lever = impact_delta[:,None] - geometry.radius*n
        surface = linear[:,None] + torch.linalg.cross(angular[:,None].expand_as(lever),lever)
        gap = torch.cat((box_gap(local_hit,geometry.racket_box_centers,geometry.racket_box_halves),
                         round_gap(local_hit,*rounded)), -1).gather(1,ids) - geometry.radius
    patch_trace = {} if load_trace is not None else None
    separation = separation_gain * gap / prior.sub_dt if separation_gain else None
    ratio = None
    if patch_inertia is not None:
        single,multiple=patch_inertia
        if not (0 < single <= 1 and 0 < multiple <= 1) or temporal_solver is not None or local_mobility is not None:
            raise ValueError('Expected two positive effective tangential inertias for a prescribed-surface solve')
        ratio = torch.where(active.sum(-1)>1, velocity.new_tensor(multiple), velocity.new_tensor(single))
    friction_floor = None
    if articulation_friction_floor:
        if local_mobility is not None:
            raise ValueError('The prescribed-surface friction load estimate requires zero racket mobility')
        # Per-shape approximation to the articulation solver's prepared minimum friction load.
        # Expressed as delta velocity for a sphere; the native solver uses articulated unit response.
        vn = ((velocity[:,None]-surface)*n).sum(-1)
        bounce = torch.where((vn < -.05) & (-vn*prior.sub_dt > gap), -vn*material[ids,2], 0.)
        estimate = (bounce-vn-gap/prior.sub_dt).clamp_min(0)
        friction_floor = torch.where(active, .25*estimate, 0.)
    if temporal_solver is not None:
        if local_mobility is not None or impact_geometry_fraction or separation_gain:
            raise ValueError('Temporal solve supplies its own separation dynamics on prescribed surfaces')
        from .supermimic_temporal_contact_patch import solve_temporal_patch
        normal,tangent,spin = solve_temporal_patch(n,gap,material[ids,1],material[ids,2],velocity,spin,surface,
            geometry.radius,active,normal_scale,prior.sub_dt,**temporal_solver,load_trace=patch_trace,friction_load_floor=friction_floor)
    elif local_mobility is None:
        normal, tangent, spin = solve_patch(n,material[ids,1],material[ids,2],velocity,spin,surface,
                                            geometry.radius,active,normal_scale,4 if compound else 1,
                                            return_spin=True,separation_velocity=separation,load_trace=patch_trace,inertia_ratio=ratio,friction_load_floor=friction_floor)
    else:
        from .supermimic_full_body_geometry_v1 import rotation as rotation_matrix
        from .supermimic_mobile_racket_contact import solve_mobile_patch
        R = rotation_matrix(impact_rotation if impact_geometry_fraction else rotation)
        zeros = torch.zeros_like(R)
        transport = torch.cat((torch.cat((R,zeros),-1),torch.cat((zeros,R),-1)),-2)
        spatial = transport @ local_mobility @ transport.transpose(-1,-2)
        normal,tangent,spin = solve_mobile_patch(n,material[ids,1],material[ids,2],velocity,spin,surface,
            lever,geometry.radius,active,normal_scale,4 if compound else 1,spatial,shuttle_mass,separation,patch_trace)
    if load_trace is not None:
        load_trace['regime'].scatter_(1, ids, patch_trace['regime'])
    result = normal, tangent, spin, active.sum(-1)
    return (*result, time) if return_impact_time else result
