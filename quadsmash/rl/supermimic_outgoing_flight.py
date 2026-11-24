"""Differentiable research cost from the shared transition's outgoing ball state."""
from math import ceil, isclose

import torch
from torch.nn import functional as F


def predict_flight_events(release_position, release_velocity, prior, court, *,
                          duration=8., dt=.01, radius=None, sample_times=()):
    """Advance free flight to the first ground event or explicit time limit.

    Positions and times for absent events are masked as NaN at the API boundary.
    Net clearance is signed against the local net height and shuttle radius.
    """
    if not 0 < dt <= duration:
        raise ValueError('Invalid flight integration interval')
    sample_ticks = {round(time / dt): time for time in sample_times}
    if any(not 0 < time <= duration or not isclose(time, tick * dt)
           for tick, time in sample_ticks.items()):
        raise ValueError('Flight samples must be positive integration boundaries within duration')
    samples = {}
    radius = prior.preview.radius if radius is None else radius
    p, v = release_position, release_velocity
    landed = p[:, 2] <= radius
    crossed = torch.zeros_like(landed)
    net_contact = torch.zeros_like(landed)
    landing = p
    landing_time = torch.zeros_like(p[:, 0])
    crossing_time = torch.zeros_like(landing_time)
    clearance = torch.zeros_like(landing_time)
    net_y = torch.zeros_like(landing_time)
    furthest_x = p[:, 0]
    net_x = court['net_plane_x_m'] + radius
    net_front = court['net_plane_x_m'] - radius - court.get('net_equivalent_thickness_m', 0.) / 2
    for index in range(ceil(duration / dt)):
        start = index * dt
        step = min(dt, duration - start)
        relative = v - prior.wind
        v = v + step * (prior.gravity - prior.drag * relative.norm(dim=-1, keepdim=True) * relative)
        q = p + step * v
        # Sweep the incoming segment against the front of the net band. The
        # free-flight target must never pass through a blocking net event.
        if 'net_width_m' in court:
            front = ~landed & ~net_contact & (p[:, 0] <= net_front) & (q[:, 0] >= net_front)
            u = ((net_front - p[:, 0]) / (q[:, 0] - p[:, 0]).clamp_min(1e-8)).clamp(0, 1)
            entry = torch.lerp(p, q, u[:, None])
            top = court['net_center_top_m'] + (court['net_side_top_m'] - court['net_center_top_m']) * (
                entry[:, 1] / court['post_y_center_abs_m']).square().clamp_max(1)
            contact = (front & (entry[:, 1].abs() <= court['net_width_m'] / 2 + radius)
                       & (entry[:, 2] <= top + radius) & (entry[:, 2] >= top - court['net_depth_m'] - radius))
            face_velocity = torch.cat((-prior.restitution * v[:, :1], v[:, 1:]), dim=-1)
            q = torch.where(contact[:, None], entry + step * (1 - u[:, None]) * face_velocity, q)
            v = torch.where(contact[:, None], face_velocity, v)
            net_contact |= contact
            clearance = torch.where(contact, entry[:, 2] - top - radius, clearance)
            net_y = torch.where(contact, entry[:, 1], net_y)
        ground = ~landed & (q[:, 2] <= radius)
        ground_fraction = ((p[:, 2] - radius) / (p[:, 2] - q[:, 2]).clamp_min(1e-8)).clamp(0, 1)
        end = torch.where(ground[:, None], torch.lerp(p, q, ground_fraction[:, None]), q)
        furthest_x = torch.where(landed, furthest_x, torch.maximum(furthest_x, end[:, 0]))
        crossing = ~landed & ~crossed & (p[:, 0] <= net_x) & (end[:, 0] >= net_x)
        fraction = ((net_x - p[:, 0]) / (end[:, 0] - p[:, 0]).clamp_min(1e-8)).clamp(0, 1)
        witness = torch.lerp(p, end, fraction[:, None])
        crossing_time = torch.where(crossing, start + step * fraction * torch.where(ground, ground_fraction, 1.),
                                    crossing_time)
        top = court['net_center_top_m'] + (court['net_side_top_m'] - court['net_center_top_m']) * (
            witness[:, 1] / court['post_y_center_abs_m']).square().clamp_max(1)
        clearance = torch.where(crossing, witness[:, 2] - top - radius, clearance)
        net_y = torch.where(crossing, witness[:, 1], net_y)
        landing = torch.where(ground[:, None], end, landing)
        landing_time = torch.where(ground, start + step * ground_fraction, landing_time)
        landed = landed | ground
        crossed = crossed | crossing
        p = end
        if index + 1 in sample_ticks:
            samples[sample_ticks[index + 1]] = dict(position=p, velocity=v, airborne=~landed,
                                                     net_contact=net_contact.clone())
        if (index + 1) % 16 == 0 and landed.all():
            break
    return dict(release_position=release_position, release_velocity=release_velocity,
                crossed_net=crossed, net_contact=net_contact, samples=samples,
                crossing_time=crossing_time.masked_fill(~crossed, float('nan')),
                net_clearance=clearance.masked_fill(~(crossed | net_contact), float('nan')),
                net_lateral_position=net_y.masked_fill(~(crossed | net_contact), float('nan')), furthest_x=furthest_x,
                landed=landed, landing_position=landing.masked_fill(~landed[:, None], float('nan')),
                landing_time=landing_time.masked_fill(~landed, float('nan')), timeout=~landed)


def calibrated_speed_loss(velocity, bias, *, minimum=4.5, task_margin=0., temperature=.15):
    """Keep the prediction correction fixed while differentiating the speed deficit."""
    speed = velocity.norm(dim=-1)
    calibrated = speed - torch.as_tensor(bias, dtype=speed.dtype, device=speed.device).detach()
    return temperature * F.softplus((minimum + task_margin - calibrated) / temperature)


def ball_only_flight_terms(position, velocity, prior, court, *, speed_bias, minimum=4.5, speed_margin=0.,
                           net_margin=.02, landing_margin=.03, center_landing=False):
    """Separate speed, direction, net and landing costs on actual predicted events."""
    events = predict_flight_events(position, velocity, prior, court)
    # Replace absent witnesses before computing penalties to keep backward finite.
    clearance = events['net_clearance'].nan_to_num()
    landing = events['landing_position'].nan_to_num()
    low, high = court['target_x_limits_m']
    width = court['target_y_half_width_m']
    width_cost = (events['net_lateral_position'].nan_to_num().abs() - court['net_width_m'] / 2).relu()
    shortfall = (court['net_plane_x_m'] + prior.preview.radius - events['furthest_x']).relu()
    net = torch.where(events['crossed_net'] | events['net_contact'],
                      (net_margin - clearance).relu() + width_cost, .1 + shortfall)
    lane_landing = ((low + landing_margin - landing[:, 0]).relu() + (landing[:, 0] - high + landing_margin).relu()
                    + (landing[:, 1].abs() - width + landing_margin).relu())
    if center_landing:
        center = landing.new_tensor(((low + high) / 2, 0.))
        lane_landing = lane_landing + .25 * (landing[:, :2] - center).abs().sum(-1)
    lane_landing = torch.where(events['landed'], lane_landing, torch.ones_like(lane_landing))
    return dict(speed=calibrated_speed_loss(velocity, speed_bias, minimum=minimum, task_margin=speed_margin),
                direction=(.1 - velocity[:, 0]).relu(), net=net, landing=lane_landing)


def outgoing_cost(ball, prior, court, *, dt=.02, steps=100, scoring_margins=False,
                  release_velocity=None, speed_goal=None):
    """Free flight in court coordinates. Interpolate net/ground crossings when reached."""
    p,v=ball.split(3,-1)
    radius = prior.preview.radius if scoring_margins else 0.
    net_x=court['net_plane_x_m'] + radius
    xmin,xmax=court['target_x_limits_m']
    landed=torch.zeros_like(p[:,0],dtype=torch.bool)
    crossed=torch.zeros_like(landed)
    landing=p
    net=p
    max_x=p[:,0]
    for step in range(steps):
        rel=v-prior.wind
        v=v+dt*(prior.gravity-prior.drag*rel.norm(dim=-1,keepdim=True)*rel)
        q=p+dt*v
        hit=(q[:,2]<=radius)&~landed
        ground_fraction=((p[:,2]-radius)/(p[:,2]-q[:,2]).clamp_min(1e-6)).clamp(0,1)
        end=torch.where(hit[:,None],torch.lerp(p,q,ground_fraction[:,None]),q)
        max_x=torch.where(landed,max_x,torch.maximum(max_x,end[:,0]))
        cross=(p[:,0]<=net_x)&(end[:,0]>=net_x)&~landed&~crossed
        fraction=((net_x-p[:,0])/(end[:,0]-p[:,0]).clamp_min(1e-6)).clamp(0,1)
        net=torch.where(cross[:,None],torch.lerp(p,end,fraction[:,None]),net)
        landing=torch.where(landed[:,None],landing,end)
        landed=landed|hit
        crossed=crossed|cross
        p=q
        # Landing freezes every scored quantity. Check in groups to avoid a
        # host synchronization at every flight step during GPU imagination.
        if (step + 1) % 16 == 0 and step + 1 < steps and landed.all():
            break
    top=court['net_center_top_m']+(court['net_side_top_m']-court['net_center_top_m'])*(net[:,1]/court['post_y_center_abs_m']).square().clamp_max(1.)
    extra = 0.
    width = court['target_y_half_width_m']
    if scoring_margins:
        top = top + prior.preview.radius + .02
        xmin, xmax, width = xmin + .03, xmax - .03, width - .03
        scored_velocity = ball[:, 3:] if release_velocity is None else release_velocity
        target_speed = 5.25 if speed_goal is None else speed_goal
        extra = ((target_speed - scored_velocity.norm(dim=-1)).relu() + (.1 - scored_velocity[:, 0]).relu()) / 5.
        if release_velocity is not None and speed_goal is None:
            # Native scoring latches velocity at release. Subsequent flight keeps the full WM endpoint.
            extra = extra + (scored_velocity[:, 2] + 1.).relu() / 5.
    clearance=torch.where(crossed,(top-net[:,2]).relu(),(net_x-max_x).relu())
    return (extra+clearance+(xmin-landing[:,0]).relu()+(landing[:,0]-xmax).relu()
            +(landing[:,1].abs()-width).relu())


def first_string_flight_loss(ball, selected, first_string, prior, court, *, release_velocity=None, speed_goal=None):
    """Score flight only after a legal first racket impact, on a fixed lane denominator."""
    selected = selected.reshape(-1).bool()
    legal = selected & first_string.reshape(-1)
    if not legal.any():
        return ball.sum() * 0. + selected.any().to(ball.dtype)
    velocity = None if release_velocity is None else release_velocity[legal]
    cost = outgoing_cost(ball[legal], prior, court, scoring_margins=True, release_velocity=velocity, speed_goal=speed_goal)
    bounded_cost = -torch.expm1(-cost)
    return (bounded_cost.sum() + (selected & ~legal).sum()) / selected.sum()
