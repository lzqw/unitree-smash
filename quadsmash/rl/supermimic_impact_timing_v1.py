"""Causal one-tick interpolation for a learned collision response.

This is not a reconstruction of PhysX substeps. Native endpoints/timestamps
must never be supplied as imagined future state. Crossing-bin selection is
discrete; selected geometry and within-bin timing retain input gradients.
"""
import torch


def hermite_position_velocity(p0,v0,p1,v1,time,dt):
    a=(time.reshape(-1,1)/dt).clamp(0.,1.)
    a2=a*a;a3=a2*a
    position=(2*a3-3*a2+1)*p0+(a3-2*a2+a)*dt*v0+(-2*a3+3*a2)*p1+(a3-a2)*dt*v1
    velocity=((6*a2-6*a)*p0+(-6*a2+6*a)*p1)/dt+(3*a2-4*a+1)*v0+(3*a2-2*a)*v1
    return position,velocity


def quaternion_world_derivative(q,omega):
    scalar=-.5*(omega*q[:,1:]).sum(-1,keepdim=True)
    vector=.5*(q[:,:1]*omega+torch.linalg.cross(omega,q[:,1:]))
    return torch.cat((scalar,vector),-1)


def interpolate_physical_state(state,predicted_endpoint,time,dt):
    """Cubic position interpolation with endpoint velocities; WXYZ rotation.

    Rotation uses normalized quaternion Hermite with world angular-velocity
    endpoint tangents. Angular velocity itself is linearly interpolated.
    These approximations change model inputs only, never native integration.
    """
    if state.ndim!=2 or state.shape[1]!=55 or predicted_endpoint.shape!=state.shape:
        raise ValueError('Impact interpolation requires aligned [batch,55] states')
    if time.shape!=(len(state),) or dt<=0:
        raise ValueError('Impact interpolation time/clock mismatch')
    p,v=hermite_position_velocity(state[:,:3],state[:,7:10],predicted_endpoint[:,:3],predicted_endpoint[:,7:10],time,dt)
    q0=state[:,3:7];q1=predicted_endpoint[:,3:7]
    q1=torch.where((q0*q1).sum(-1,keepdim=True)>=0,q1,-q1)
    q,_=hermite_position_velocity(q0,quaternion_world_derivative(q0,state[:,10:13]),
        q1,quaternion_world_derivative(q1,predicted_endpoint[:,10:13]),time,dt)
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    a=(time[:,None]/dt).clamp(0.,1.)
    omega=(1-a)*state[:,10:13]+a*predicted_endpoint[:,10:13]
    joint,djoint=hermite_position_velocity(state[:,13:31],state[:,31:49],predicted_endpoint[:,13:31],predicted_endpoint[:,31:49],time,dt)
    ball,ball_v=hermite_position_velocity(state[:,49:52],state[:,52:55],predicted_endpoint[:,49:52],predicted_endpoint[:,52:55],time,dt)
    return torch.cat((p,q,v,omega,joint,djoint,ball,ball_v),-1)


def causal_sweep_time(first,velocity,last,preview):
    """First modeled sphere/compound-racket gap crossing in the native grid.

    The current-frame quadratic uses current relative velocity and a learned
    endpoint. A no-crossing path uses its closest grid node, explicitly
    reported as such; no collision is certified by this fallback.
    """
    if first.shape!=velocity.shape or first.shape!=last.shape or first.ndim!=2 or first.shape[1]!=3:
        raise ValueError('Sweep requires aligned [batch,3] local motion')
    times=preview.times.to(first)
    dt=times[-1]
    if len(times)<2 or not (dt>0):raise ValueError('Missing model/native control grid')
    curvature=last-first-dt*velocity
    with torch.no_grad():
        t=times[None,:,None]
        path=first[:,None]+t*velocity[:,None]+(t/dt).square()*curvature[:,None]
        gap=preview.racket_gap(path)
        crossed=(gap<=0).any(1)
        first_index=(gap<=0).long().argmax(1)
        selected=torch.where(crossed,first_index,gap.argmin(1))
        lower=(selected-1).clamp_min(0)
    t0=times[lower];t1=times[selected]
    def point(time):
        return first+time[:,None]*velocity+(time[:,None]/dt).square()*curvature
    g0=preview.racket_gap(point(t0));g1=preview.racket_gap(point(t1))
    fraction=(g0/(g0-g1).clamp_min(1e-9)).clamp(0.,1.)
    root=t0+fraction*(t1-t0)
    root=torch.where(crossed,root,t1)
    return root,dict(crossing_found=crossed,first_bin=selected)
