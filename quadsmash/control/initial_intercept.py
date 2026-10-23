"""Causal initial-ball prediction from the public observation and locked model.

The known command position/direction defines the world reference. No simulator
state, future labels, recorded trajectory or collision outcome is an input.
"""
import numpy as np


def observed_world_heading(observation, command_normal):
    """Recover heading from current public gravity/known command direction."""
    import torch
    gravity = torch.nn.functional.normalize(observation[:, 6:9], dim=-1)
    normal = observation[:, 763:766]
    tangent = torch.nn.functional.normalize(normal-(normal*gravity).sum(1, keepdim=True)*gravity, dim=-1)
    body_basis = torch.stack((gravity, tangent, torch.linalg.cross(gravity, tangent)), -1)
    g = torch.tensor([0., 0., -1.], device=observation.device, dtype=observation.dtype)
    n = torch.as_tensor(command_normal, device=observation.device, dtype=observation.dtype)
    t = torch.nn.functional.normalize(n-(n*g).sum()*g, dim=-1)
    world_basis = torch.stack((g, t, torch.linalg.cross(g, t)), -1)
    rotation = world_basis@body_basis.transpose(1, 2)
    return torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])


def _unit(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate public direction reference')
    return vector/norm


def observed_initial_ball(observation, command_position, command_normal):
    obs = np.asarray(observation, dtype=np.float64)
    count = len(obs)
    gravity_body = _unit(obs[:, 6:9])
    normal_body = obs[:, 763:766]
    tangent_body = _unit(normal_body-(normal_body*gravity_body).sum(1, keepdims=True)*gravity_body)
    body_basis = np.stack((gravity_body, tangent_body, np.cross(gravity_body, tangent_body)), -1)
    gravity_world = np.tile([0., 0., -1.], (count, 1))
    normal_world = np.broadcast_to(command_normal, (count, 3)).copy()
    tangent_world = _unit(normal_world-(normal_world*gravity_world).sum(1, keepdims=True)*gravity_world)
    world_basis = np.stack((gravity_world, tangent_world, np.cross(gravity_world, tangent_world)), -1)
    rotation = world_basis@body_basis.transpose(0, 2, 1)
    root = np.asarray(command_position)-np.einsum('nij,nj->ni', rotation, obs[:, 760:763])
    position = root+np.einsum('nij,nj->ni', rotation, obs[:, 61:64])
    velocity_body = obs[:, 64:67]+obs[:, :3]+np.cross(obs[:, 3:6], obs[:, 61:64])
    velocity = np.einsum('nij,nj->ni', rotation, velocity_body)
    return position, velocity


def predict_initial_intercept(observation, command_position, command_normal,
                              intercept_x, physics, dt=.002, max_seconds=8.):
    position, velocity = observed_initial_ball(observation, command_position, command_normal)
    initial_position, initial_velocity = position.copy(), velocity.copy()
    count = len(position)
    x = np.broadcast_to(intercept_x, (count,))
    wind = np.asarray(physics['wind_m_s'])
    drag = physics['quadratic_drag_k_per_m']
    valid = np.zeros(count, dtype=bool)
    alive = (position[:, 0] > x) & (position[:, 2] > .025)
    arrival = np.full(count, np.nan)
    intercept = np.full((count, 3), np.nan)
    incoming = intercept.copy()

    def acceleration(v):
        relative = v-wind
        return np.array([0., 0., -9.81])-drag*np.linalg.norm(relative, axis=1, keepdims=True)*relative

    for step in range(int(np.ceil(max_seconds/dt))):
        a1 = acceleration(velocity)
        v2 = velocity+.5*dt*a1
        a2 = acceleration(v2)
        v3 = velocity+.5*dt*a2
        a3 = acceleration(v3)
        v4 = velocity+dt*a3
        a4 = acceleration(v4)
        next_position = position+dt*(velocity+2*v2+2*v3+v4)/6
        next_velocity = velocity+dt*(a1+2*a2+2*a3+a4)/6
        crossing = alive & (~valid) & (position[:, 0] >= x) & (next_position[:, 0] <= x)
        alpha = np.clip((position[:, 0]-x)/np.maximum(position[:, 0]-next_position[:, 0], 1e-12), 0, 1)
        point = position+alpha[:, None]*(next_position-position)
        cross_velocity = velocity+alpha[:, None]*(next_velocity-velocity)
        crossing &= (cross_velocity[:, 2] < 0.) & (point[:, 2] > .025)
        arrival[crossing] = (step+alpha[crossing])*dt
        intercept[crossing] = point[crossing]
        incoming[crossing] = cross_velocity[crossing]
        valid |= crossing
        alive &= next_position[:, 2] > .025
        position, velocity = next_position, next_velocity
        if np.all(valid | (~alive)):
            break
    return dict(valid=valid, arrival_time_s=arrival, position_m=intercept,
                incoming_velocity_mps=incoming, initial_position_m=initial_position,
                initial_velocity_mps=initial_velocity)
