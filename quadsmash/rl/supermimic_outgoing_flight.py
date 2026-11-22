"""Differentiable research cost from the shared transition's outgoing ball state."""
import torch


def outgoing_cost(ball, prior, court, *, dt=.02, steps=100, scoring_margins=False, release_velocity=None):
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
        extra = ((5.25 - scored_velocity.norm(dim=-1)).relu() + (.1 - scored_velocity[:, 0]).relu()) / 5.
        if release_velocity is not None:
            # Native scoring latches velocity at release. Subsequent flight keeps the full WM endpoint.
            extra = extra + (scored_velocity[:, 2] + 1.).relu() / 5.
    clearance=torch.where(crossed,(top-net[:,2]).relu(),(net_x-max_x).relu())
    return (extra+clearance+(xmin-landing[:,0]).relu()+(landing[:,0]-xmax).relu()
            +(landing[:,1].abs()-width).relu())


def first_string_flight_loss(ball, selected, first_string, prior, court, *, release_velocity=None):
    """Score flight only after a legal first racket impact, on a fixed lane denominator."""
    selected = selected.reshape(-1).bool()
    legal = selected & first_string.reshape(-1)
    if not legal.any():
        return ball.sum() * 0. + selected.any().to(ball.dtype)
    velocity = None if release_velocity is None else release_velocity[legal]
    cost = outgoing_cost(ball[legal], prior, court, scoring_margins=True, release_velocity=velocity)
    bounded_cost = -torch.expm1(-cost)
    return (bounded_cost.sum() + (selected & ~legal).sum()) / selected.sum()
