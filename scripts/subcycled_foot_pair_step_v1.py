"""Causal refined integration of the coupled contact/implicit-drive model.

Subdivision changes numerical prediction only. It does not change Isaac's
physical timestep, hardware bounds, contact geometry, or native commands.
It is not an implementation of PhysX's TGS iterations.
"""
import copy

import numpy as np
import pinocchio as pin

from scripts.foot_pair_contact_step_v1 import foot_pair_velocity_step
from scripts.endpoint_contact_velocity_probe_v1 import endpoint_velocity_probe


def subcycled_foot_pair_step(model, interface, state, target, spheres, dt, *,
                            foot_friction, subdivisions=8, endpoint_velocity=False):
    if type(subdivisions) is not int or not 1 <= subdivisions <= 16:
        raise ValueError('Explicit bounded integration resolution required')
    current=copy.deepcopy(state);delta=dt/subdivisions
    ground=np.zeros((4,3));residual=0.;last=None
    for i in range(subdivisions):
        if endpoint_velocity and i==subdivisions-1:
            r=endpoint_velocity_probe(model,interface,current,target,spheres,delta,foot_friction=foot_friction)
            q=r['q']
        else:
            r=foot_pair_velocity_step(model,interface,current,target,spheres,delta,foot_friction=foot_friction)
            q=pin.integrate(model.model,current.q,delta*r['velocity'])
        current.q=q;current.v=r['velocity'];last=r
        ground+=r['ground_impulse'];residual=max(residual,r['momentum_residual'])
    return dict(q=current.q,velocity=current.v,ground_impulse=ground,
        endpoint_ground_impulse=last['ground_impulse'],momentum_residual=residual,
        subdivisions=subdivisions,endpoint_velocity_correction=endpoint_velocity,native_equivalent=False)
