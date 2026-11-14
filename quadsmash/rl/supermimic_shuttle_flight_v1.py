"""Differentiable free-flight reference using the pinned native drag force.

Only free flight: no launch teleport, racket collision, net or ground response.
The semi-implicit integration is a candidate for native validation, not a claim
that PhysX contact dynamics has been reproduced. Native execution is unchanged.
"""
import math

import torch
from torch import nn

from quadsmash.physics.native import drag_force, physics_contract


class ShuttleFreeFlightV1(nn.Module):
    def __init__(self, *, control_dt_s=.01, physics_dt_s=.0000625,
                 gravity=(0., 0., -9.81), contract=None, dtype=torch.float32):
        super().__init__()
        c = physics_contract() if contract is None else contract
        if control_dt_s <= 0 or physics_dt_s <= 0:
            raise ValueError('Positive time steps required')
        steps = round(control_dt_s / physics_dt_s)
        if steps < 1 or not math.isclose(steps*physics_dt_s,control_dt_s,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Control period must contain whole native substeps')
        self.steps = steps
        self.dt = float(physics_dt_s)
        self.mass = float(c['mass_kg'])
        self.coefficient = float(c['quadratic_drag_k_per_m'])
        self.register_buffer('gravity',torch.tensor(gravity,dtype=dtype))
        self.register_buffer('wind',torch.tensor(c['wind_m_s'],dtype=dtype))

    def forward(self, position, velocity):
        if position.shape != velocity.shape or position.shape[-1] != 3:
            raise ValueError('Require matching position/velocity with three components')
        p, v = position, velocity
        for _ in range(self.steps):
            acceleration = self.gravity + drag_force(v,self.mass,self.coefficient,self.wind)/self.mass
            v = v + self.dt*acceleration
            p = p + self.dt*v
        return p, v
