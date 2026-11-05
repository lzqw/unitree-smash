"""Optional JAX implementation of the batched VHSIP capture calculation.

This module mirrors ``ReactiveLandingV1._vhsip_capture`` without importing
JAX at package import time.  It is intentionally limited to the analytical
horizontal VHSIP/ZMP subproblem; FK, contact detection and the native action
interface remain outside this module.  When JAX is installed, callers can use
``jax.jit``/``jax.vmap`` or the provided ``jit_vhsip_capture`` directly.

The Torch implementation remains the reference until a parity test is run on
the same inputs and the native landing trial shows an actual benefit.
"""
from __future__ import annotations

try:  # Keep the rest of the repository importable without optional JAX.
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised only without JAX.
    jax = None
    jnp = None


def _require_jax():
    if jax is None or jnp is None:
        raise ImportError('Install jax/jaxlib to use the optional VHSIP backend')


def vhsip_capture(velocity_xy, position_xy, height, vertical_velocity, mass,
                  dt=.01, settling=.6, gravity=9.81):
    """Return the batched horizontal VHSIP capture/ZMP offset.

    Arguments are arrays with shapes ``[batch, 2]``, ``[batch, 2]``,
    ``[batch]``, ``[batch]`` and scalar mass.  The output has shape
    ``[batch, 2]`` and uses the same sign convention as the Torch reference.
    """
    _require_jax()
    velocity_xy = jnp.asarray(velocity_xy)
    position_xy = jnp.asarray(position_xy)
    height = jnp.asarray(height)
    vertical_velocity = jnp.asarray(vertical_velocity)
    mass = jnp.asarray(mass, dtype=height.dtype)
    if velocity_xy.ndim != 2 or velocity_xy.shape[-1] != 2:
        raise ValueError('velocity_xy must have shape [batch,2]')
    if position_xy.shape != velocity_xy.shape:
        raise ValueError('position_xy must match velocity_xy')

    compression = .4 * jnp.maximum(height, .15)
    stiffness = mass * (8. / settling) ** 2
    eig = -jnp.sqrt(stiffness / mass)
    limit = jnp.e * compression * eig
    fast = vertical_velocity < limit
    fast_k = mass * (vertical_velocity / jnp.maximum(jnp.e * compression, 1e-4)) ** 2
    stiffness = jnp.where(fast, fast_k, stiffness)
    damping = 2. * jnp.sqrt(mass * stiffness)
    eig = -jnp.sqrt(stiffness / mass)
    steps = int(round(float(settling / dt)))
    time = jnp.arange(steps, dtype=height.dtype) * dt
    et = jnp.exp(eig[:, None] * time[None])
    vz_t = et * (eig[:, None] * time[None] + 1.) * vertical_velocity[:, None]
    z = et * time[None] * vertical_velocity[:, None] + height[:, None]
    az = (-(damping / mass)[:, None] * vz_t
          - (stiffness / mass)[:, None]
          * (et * time[None] * vertical_velocity[:, None]))
    omega = jnp.maximum((gravity + az) / jnp.maximum(z, .12), 0.)

    batch = height.shape[0]
    eye = jnp.broadcast_to(jnp.eye(2, dtype=height.dtype), (batch, 2, 2))
    p0 = eye
    u0 = jnp.zeros((batch, 2, 1), dtype=height.dtype)

    def step(carry, om):
        p, u = carry
        a = jnp.zeros_like(p).at[:, 0, 0].set(1.).at[:, 1, 1].set(1.)
        a = a.at[:, 0, 1].set(dt * om)
        b = jnp.zeros_like(u).at[:, 0, 0].set(-dt * om)
        return (a @ p, a @ u + b), None

    # The final omega sample is not used by the discrete rollout, matching the
    # Torch loop over range(steps-1).
    (p, u), _ = jax.lax.scan(step, (p0, u0), omega[:, :-1].T)
    cvp, cpp = p[:, 0, :], p[:, 1, :]
    cvu, cpu = u[:, 0, 0], u[:, 1, 0]
    qu = cvu * cvu + (cpu - 1.) ** 2 + 1e-6
    qxu = cvu[:, None] * cvp + (cpu - 1.)[:, None] * cpp
    x0 = jnp.stack((velocity_xy, position_xy), axis=-1)
    return -(qxu[:, None, :] * x0).sum(axis=-1) / qu[:, None]


def jit_vhsip_capture(*args, **kwargs):
    """JIT-compiled convenience wrapper; errors clearly when JAX is absent."""
    _require_jax()
    return jax.jit(vhsip_capture)(*args, **kwargs)


__all__ = ['vhsip_capture', 'jit_vhsip_capture']
