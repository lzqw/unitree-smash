"""FP32 shuttle integration and its exact discrete, first-order adjoint."""
from hashlib import sha256
from pathlib import Path

import torch
import triton
import triton.language as tl

from .supermimic_research_flight_cuda import ResearchCudaImpulsePrior, _flight, _load_xyz, _store_xyz


@triton.jit
def _variable_path(P, V, D, G, W, POSITION, VELOCITY, REMAINING, N: tl.constexpr,
                   STEPS: tl.constexpr, DT: tl.constexpr, DRAG: tl.constexpr):
    lane = tl.program_id(0) * 32 + tl.arange(0, 32)
    valid = lane < N
    px, py, pz = _load_xyz(P, lane * 3, 1, valid)
    vx, vy, vz = _load_xyz(V, lane * 3, 1, valid)
    gx, gy, gz = _load_xyz(G, 0, 1, True)
    wx, wy, wz = _load_xyz(W, 0, 1, True)
    left = tl.load(D + lane, valid, 0.)
    start = lane * (STEPS + 1) * 3
    _store_xyz(VELOCITY, start, valid, vx, vy, vz)
    for step in range(STEPS):
        tl.store(REMAINING + lane * STEPS + step, left, valid)
        dt = tl.where(left != left, left, tl.minimum(tl.maximum(left, 0.), DT))
        left = left - dt
        rx, ry, rz = vx - wx, vy - wy, vz - wz
        drag_speed = DRAG * tl.sqrt_rn((rx * rx + rz * rz) + ry * ry)
        vx = vx + dt * (gx - drag_speed * rx)
        vy = vy + dt * (gy - drag_speed * ry)
        vz = vz + dt * (gz - drag_speed * rz)
        px = px + dt * vx
        py = py + dt * vy
        pz = pz + dt * vz
        _store_xyz(VELOCITY, start + (step + 1) * 3, valid, vx, vy, vz)
    _store_xyz(POSITION, lane * 3, valid, px, py, pz)


@triton.jit
def _reverse_flight(V, R, G, W, DP, DV, OUT_P, OUT_V, OUT_D, N: tl.constexpr,
                    STEPS: tl.constexpr, DT: tl.constexpr, DRAG: tl.constexpr, PATH: tl.constexpr):
    lane = tl.program_id(0) * 32 + tl.arange(0, 32)
    valid = lane < N
    start = lane * (STEPS + 1) * 3
    wx, wy, wz = _load_xyz(W, 0, 1, True)
    ax, ay, az = _load_xyz(G, 0, 1, True)
    if PATH:
        zero = tl.full((32,), 0., tl.float32)
        dx, dy, dz = zero, zero, zero
        ux, uy, uz = zero, zero, zero
    else:
        dx, dy, dz = _load_xyz(DP, lane * 3, 1, valid)
        ux, uy, uz = _load_xyz(DV, lane * 3, 1, valid)
    duration_grad = tl.full((32,), 0., tl.float32)
    for rev in range(STEPS):
        step = STEPS - rev - 1
        prior = start + step * 3
        if PATH:
            dt = DT
            kx, ky, kz = _load_xyz(DP, prior + 3, 1, valid)
            lx, ly, lz = _load_xyz(DV, prior + 3, 1, valid)
            dx, dy, dz = dx + kx, dy + ky, dz + kz
            ux, uy, uz = ux + lx, uy + ly, uz + lz
        else:
            left = tl.load(R + lane * STEPS + step, valid, 0.)
            dt = tl.minimum(tl.maximum(left, 0.), DT)
        vx, vy, vz = _load_xyz(V, prior, 1, valid)
        rx, ry, rz = vx - wx, vy - wy, vz - wz
        speed = tl.sqrt_rn((rx * rx + rz * rz) + ry * ry)
        ux, uy, uz = ux + dt * dx, uy + dt * dy, uz + dt * dz
        if not PATH:
            nx, ny, nz = _load_xyz(V, prior + 3, 1, valid)
            grad_dt = dx * nx + dy * ny + dz * nz
            grad_dt += ux * (ax - DRAG * speed * rx) + uy * (ay - DRAG * speed * ry) + uz * (az - DRAG * speed * rz)
            # PyTorch clamp uses the inclusive interval for its input derivative.
            duration_grad = tl.where((left >= 0.) & (left <= DT), grad_dt, duration_grad)
        dot = ux * rx + uy * ry + uz * rz
        direction = tl.where(speed > 0., dot / speed, 0.)
        ux = ux - dt * DRAG * (speed * ux + rx * direction)
        uy = uy - dt * DRAG * (speed * uy + ry * direction)
        uz = uz - dt * DRAG * (speed * uz + rz * direction)
    if PATH:
        kx, ky, kz = _load_xyz(DP, start, 1, valid)
        lx, ly, lz = _load_xyz(DV, start, 1, valid)
        dx, dy, dz = dx + kx, dy + ky, dz + kz
        ux, uy, uz = ux + lx, uy + ly, uz + lz
    else:
        tl.store(OUT_D + lane, duration_grad, valid)
    _store_xyz(OUT_P, lane * 3, valid, dx, dy, dz)
    _store_xyz(OUT_V, lane * 3, valid, ux, uy, uz)


class _DifferentiableFlight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, v, duration, gravity, wind, steps, dt, drag):
        n = len(p)
        path = duration is None
        shape = (n, steps + 1, 3)
        velocities = v.new_empty(shape)
        grid = (triton.cdiv(n, 32),)
        options = dict(num_warps=1, enable_fp_fusion=False)
        if path:
            positions = p.new_empty(shape)
            with torch.cuda.device(p.device):
                _flight[grid](p, v, p, gravity, wind, positions, velocities, n,
                              *p.stride(), *v.stride(), 0, steps, drag, dt, True, 32, **options)
            outputs = positions, velocities
            remaining = v.new_empty(0)
        else:
            positions = p.new_empty(p.shape)
            remaining = duration.new_empty((n, steps))
            with torch.cuda.device(p.device):
                _variable_path[grid](p.contiguous(), v.contiguous(), duration.contiguous(), gravity,
                                     wind, positions, velocities, remaining, n, steps, dt, drag, **options)
            outputs = positions, velocities[:, -1].contiguous()
        ctx.save_for_backward(velocities, remaining, gravity, wind)
        ctx.settings = n, steps, dt, drag, path
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dp, dv):
        velocity, remaining, gravity, wind = ctx.saved_tensors
        n, steps, dt, drag, path = ctx.settings
        gp, gv = velocity.new_empty((n, 3)), velocity.new_empty((n, 3))
        gd = velocity.new_empty(n)
        with torch.cuda.device(velocity.device):
            _reverse_flight[(triton.cdiv(n, 32),)](
                velocity, remaining, gravity, wind, dp.contiguous(), dv.contiguous(), gp, gv, gd,
                n, steps, dt, drag, path, num_warps=1, enable_fp_fusion=False)
        return gp, gv, None if path else gd, None, None, None, None, None


class DifferentiableCudaImpulsePrior(ResearchCudaImpulsePrior):
    def _differentiable(self, p, v, duration=None):
        inputs = (p, v) if duration is None else (p, v, duration)
        return (torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
                and not self.gravity.requires_grad and not self.wind.requires_grad
                and torch.autograd.forward_ad._current_level < 0
                and not torch._C._are_functorch_transforms_active())

    def _supports_gradient_kernel(self, p, v, duration=None):
        if not self._differentiable(p, v, duration):
            return False
        with torch.no_grad():
            return self._can_fuse(p, v, duration)

    def advance(self, p, v, duration):
        if self._supports_gradient_kernel(p, v, duration):
            return _DifferentiableFlight.apply(p, v, duration, self.gravity, self.wind,
                                              self.steps, self.sub_dt, self.drag)
        return super().advance(p, v, duration)

    def advance_trajectory(self, p, v):
        if self._supports_gradient_kernel(p, v):
            return _DifferentiableFlight.apply(p, v, None, self.gravity, self.wind,
                                              self.steps, self.sub_dt, self.drag)
        return super().advance_trajectory(p, v)


def enable_differentiable_flight(model, expected_backend=None):
    prior = model.impulse_prior
    backend = dict(schema='research-fp32-flight-adjoint-v1',
                   source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch=str(torch.__version__), triton=triton.__version__)
    if expected_backend is not None and backend != expected_backend:
        raise ValueError('Differentiable flight backend differs from checkpoint')
    fused = DifferentiableCudaImpulsePrior(**prior.calibration_manifest()).to(prior.gravity)
    fused.gravity.copy_(prior.gravity)
    fused.wind.copy_(prior.wind)
    model.impulse_prior = fused.requires_grad_(False).train(prior.training)
    return backend
