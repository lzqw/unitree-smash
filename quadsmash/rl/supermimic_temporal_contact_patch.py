"""Reduced temporal contact constraints for a spherical shuttle and prescribed racket surfaces."""
import torch


def solve_temporal_patch(n, gap, mu, restitution, velocity, spin, surface, radius, active,
                         normal_scale, dt, position_iterations, velocity_iterations, bounce_speed,
                         *, load_trace=None):
    """Integrate the relative constraint displacement during position iterations.

    Normal impulses use a speculative separation and an incoming-speed rebound target.
    Friction acts on the spinning sphere at each surface. Racket twists remain prescribed.
    """
    if position_iterations < 1 or velocity_iterations < 0 or dt <= 0 or bounce_speed < 0:
        raise ValueError('Invalid temporal contact solver configuration')
    dt_iter = dt/position_iterations
    arm = -radius*n
    linear, angular = velocity, spin
    closing = ((velocity[:,None]-surface)*n).sum(-1)
    bounce = active & (closing < -bounce_speed) & ((-closing)*dt > gap)
    target = torch.where(bounce, -restitution*closing, 0.)
    # Anchor the restitution trajectory to the crossing within this physical interval.
    safe_closing = torch.where(bounce, closing, -1.)
    separation = gap + target*(dt+gap/safe_closing)
    normal_velocity_scale = (gap>0).to(n.dtype)*.2+.8
    elapsed_displacement = torch.zeros_like(n)
    loads = [torch.zeros_like(g) for g in gap.unbind(1)]
    tangents = [torch.zeros_like(axis) for axis in n.unbind(1)]
    for iteration in range(position_iterations+velocity_iterations):
        phase_time = min(iteration,position_iterations)*dt_iter
        separation_now = separation+(elapsed_displacement*n).sum(-1)
        if iteration>=position_iterations:
            separation_now=separation_now.clamp_min(0.)
        bias = -(separation_now-target*phase_time)*normal_velocity_scale/dt_iter
        for j,(axis,lever) in enumerate(zip(n.unbind(1),arm.unbind(1),strict=True)):
            rel = linear + torch.linalg.cross(angular,lever) - surface[:,j]
            vn = (rel*axis).sum(-1)
            load = torch.where(active[:,j],(loads[j]+target[:,j]+bias[:,j]-vn).clamp_min(0.),0.)
            linear = linear+(load-loads[j])[:,None]*axis
            loads[j]=load
            slip = rel-vn[:,None]*axis
            drift = elapsed_displacement[:,j]
            drift = drift-(drift*axis).sum(-1,keepdim=True)*axis
            request=tangents[j]-(2./7)*(slip+.8*drift/dt_iter)
            bound=mu[:,j]*load
            tangent=request*(bound/request.norm(dim=-1).clamp_min(1.e-12)).clamp(max=1)[:,None]
            delta=tangent-tangents[j]
            linear=linear+delta
            angular=angular+torch.linalg.cross(lever,delta)/(.4*radius**2)
            tangents[j]=tangent
        if iteration < position_iterations:
            elapsed_displacement=elapsed_displacement+dt_iter*(linear[:,None]+torch.linalg.cross(angular[:,None].expand_as(arm),arm)-surface)
    loads=torch.stack(loads,1)
    tangents=torch.stack(tangents,1)
    if load_trace is not None:
        loaded=loads>1.e-8
        sliding=tangents.norm(dim=-1)>=(mu*loads-1.e-8).clamp_min(0)
        load_trace['regime']=loaded.to(torch.uint8)*(1+sliding.to(torch.uint8))
    # Use the calibrated finite-mass scale on the net impulse once.
    return ((loads[...,None]*n).sum(1)*normal_scale,tangents.sum(1)*normal_scale,
            spin+normal_scale*(angular-spin))
