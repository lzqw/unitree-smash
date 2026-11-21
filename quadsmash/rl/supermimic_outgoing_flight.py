"""Differentiable research cost from the shared transition's outgoing ball state."""
import torch


def outgoing_cost(ball, prior, court, *, dt=.02, steps=100):
    """Free flight in court coordinates. Interpolate net/ground crossings when reached."""
    p,v=ball.split(3,-1)
    net_x=court['net_plane_x_m']
    xmin,xmax=court['target_x_limits_m']
    landed=torch.zeros_like(p[:,0],dtype=torch.bool)
    crossed=torch.zeros_like(landed)
    landing=p
    net=p
    max_x=p[:,0]
    for _ in range(steps):
        rel=v-prior.wind
        v=v+dt*(prior.gravity-prior.drag*rel.norm(dim=-1,keepdim=True)*rel)
        q=p+dt*v
        hit=(q[:,2]<=0)&~landed
        ground_fraction=(p[:,2]/(p[:,2]-q[:,2]).clamp_min(1e-6)).clamp(0,1)
        end=torch.where(hit[:,None],torch.lerp(p,q,ground_fraction[:,None]),q)
        max_x=torch.where(landed,max_x,torch.maximum(max_x,end[:,0]))
        cross=(p[:,0]<=net_x)&(end[:,0]>=net_x)&~landed&~crossed
        fraction=((net_x-p[:,0])/(end[:,0]-p[:,0]).clamp_min(1e-6)).clamp(0,1)
        net=torch.where(cross[:,None],torch.lerp(p,end,fraction[:,None]),net)
        landing=torch.where(landed[:,None],landing,end)
        landed=landed|hit
        crossed=crossed|cross
        p=q
    top=court['net_center_top_m']+(court['net_side_top_m']-court['net_center_top_m'])*(net[:,1]/court['post_y_center_abs_m']).square().clamp_max(1.)
    clearance=torch.where(crossed,(top-net[:,2]).relu(),(net_x-max_x).relu())
    return (clearance+(xmin-landing[:,0]).relu()+(landing[:,0]-xmax).relu()
            +(landing[:,1].abs()-court['target_y_half_width_m']).relu())
