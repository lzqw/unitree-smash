"""CPU free-flight planning from a measured release; never a rally scorer.

This integrates the current rigid shuttle surrogate and screens finite-radius
clearance against the actual64 suspended net boxes and both posts. A positive
screen is a conservative collision-avoiding planning subset; legal net-touch
returns remain legal under the original native rules. Racket/robot dynamics,
reachable output velocity and recovery are outside this model.
"""
from dataclasses import dataclass
import math

import numpy as np

from quadsmash.physics.court_v4 import net_band_segments


@dataclass(frozen=True)
class FlightScreen:
    landing_m: np.ndarray
    landing_time_s: np.ndarray
    trailing_crossing_m: np.ndarray
    minimum_sampled_net_gap_m: np.ndarray
    maximum_half_step_travel_bound_m: np.ndarray

    @property
    def buffered_net_gap_m(self):
        return self.minimum_sampled_net_gap_m-self.maximum_half_step_travel_bound_m


def rk4(position, velocity, dt, drag_k, gravity, wind):
    def acceleration(v):
        relative = v-wind
        return gravity-drag_k*np.linalg.norm(relative,axis=1,keepdims=True)*relative
    a1 = acceleration(velocity)
    v2 = velocity+.5*dt*a1; a2 = acceleration(v2)
    v3 = velocity+.5*dt*a2; a3 = acceleration(v3)
    v4 = velocity+dt*a3; a4 = acceleration(v4)
    return (position+dt*(velocity+2*v2+2*v3+v4)/6,
            velocity+dt*(a1+2*a2+2*a3+a4)/6)


def net_gap(points, court, radius):
    """Exact point-to-static-box/cylinder signed distance minus sphere radius."""
    boxes = np.asarray([p for p,_ in net_band_segments(court)],dtype=np.float64)
    boxes[:,:,0] += court['net_plane_x_m']
    low,high = boxes.min(1),boxes.max(1)
    delta = np.maximum(np.maximum(low[None]-points[:,None],points[:,None]-high[None]),0.)
    distance = np.linalg.norm(delta,axis=2).min(1)-radius
    # Outside distances are exact; inside a box distance is0 and therefore
    # correctly fails the positive-clearance planning screen.
    for sign in [-1,1]:
        radial = np.hypot(points[:,0]-court['net_plane_x_m'],
            points[:,1]-sign*court['post_y_center_abs_m'])-court['post_radius_m']
        vertical = np.abs(points[:,2]-court['post_height_m']/2)-court['post_height_m']/2
        signed = np.hypot(np.maximum(radial,0),np.maximum(vertical,0))+np.minimum(np.maximum(radial,vertical),0)
        distance = np.minimum(distance,signed-radius)
    return distance


def free_flight(position, velocity, physics, court, *, dt=.0005, horizon_s=2., gravity=(0.,0.,-9.81)):
    p,v = [np.asarray(x,dtype=np.float64).copy() for x in [position,velocity]]
    if p.ndim != 2 or p.shape != v.shape or p.shape[1] != 3 or not len(p):
        raise ValueError('Require nonempty matching [N,3] release positions and velocities')
    if not np.isfinite(p).all() or not np.isfinite(v).all() or not 0 < dt <= .01 or not 0 < horizon_s <= 8.:
        raise ValueError('Finite release states and explicit finite integration range required')
    radius,drag_k = physics['collision_radius_m'],physics['quadratic_drag_k_per_m']
    g,w = np.asarray(gravity,dtype=np.float64),np.asarray(physics['wind_m_s'],dtype=np.float64)
    if (w != 0).any():
        raise ValueError('The speed-step bound is currently validated only for the unchanged zero-wind task')
    if (p[:,2] <= radius).any() or (np.linalg.norm(v,axis=1) > physics['max_speed_m_s']).any():
        raise ValueError('Release must be above ground and within the declared surrogate speed range')
    count = len(p); active = np.ones(count,dtype=bool)
    landing,crossing = [np.full((count,3),np.nan) for _ in range(2)]
    landing_time = np.full(count,np.nan)
    net_min = np.full(count,np.inf); half_travel = np.zeros(count)
    trailing_plane = court['net_plane_x_m']+radius
    for tick in range(math.ceil(horizon_s/dt)):
        if not active.any(): break
        step = min(dt,horizon_s-tick*dt)
        # With zero wind, drag cannot increase speed and |v|'<=|g|. Distance
        # to a closed obstacle is1-Lipschitz. Half-step travel buffers between
        # sampled endpoints; integration error is separately refined by callers.
        travel = .5*step*(np.linalg.norm(v,axis=1)+np.linalg.norm(g)*step)
        half_travel[active] = np.maximum(half_travel[active],travel[active])
        near = active & (np.abs(p[:,0]-court['net_plane_x_m']) <=
            radius+court['net_equivalent_thickness_m']/2+2*travel)
        if near.any(): net_min[near] = np.minimum(net_min[near],net_gap(p[near],court,radius))
        next_p,next_v = rk4(p,v,step,drag_k,g,w)
        if not np.isfinite(next_p).all() or not np.isfinite(next_v).all():
            raise ValueError('Nonfinite free-flight integration')
        crossed = active & np.isnan(crossing[:,0]) & (p[:,0] < trailing_plane) & (next_p[:,0] >= trailing_plane)
        if crossed.any():
            fraction = (trailing_plane-p[crossed,0])/(next_p[crossed,0]-p[crossed,0])
            crossing[crossed] = p[crossed]+fraction[:,None]*(next_p[crossed]-p[crossed])
        landed = active & (next_p[:,2] <= radius)
        if landed.any():
            fraction = (p[landed,2]-radius)/(p[landed,2]-next_p[landed,2])
            landing[landed] = p[landed]+fraction[:,None]*(next_p[landed]-p[landed])
            landing_time[landed] = tick*dt+fraction*step
        p[active],v[active] = next_p[active],next_v[active]
        active[landed] = False
    return FlightScreen(landing,landing_time,crossing,net_min,half_travel)


def target_screen(flight, velocity, court):
    """Planning conditions only; never substitutes native complete/legality flags."""
    p,v = flight.landing_m,np.asarray(velocity)
    return (np.isfinite(p).all(1) & np.isfinite(flight.trailing_crossing_m).all(1)
        & (v[:,0] > 0.) & (np.linalg.norm(v,axis=1) >= 8.-1e-12) & (v[:,2] <= -1.)
        & (p[:,0] >= court['target_x_limits_m'][0]) & (p[:,0] <= court['target_x_limits_m'][1])
        & (np.abs(p[:,1]) <= court['target_y_half_width_m']) & (flight.buffered_net_gap_m > 0.))
