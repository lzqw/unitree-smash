"""Fused FP32 CUDA shuttle integration for research rollouts without flight gradients."""
from hashlib import sha256
from pathlib import Path

import torch
import triton
import triton.language as tl

from .supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


@triton.jit
def _load_xyz(ptr, offset, stride, mask):
    return (tl.load(ptr + offset, mask, 0.), tl.load(ptr + offset + stride, mask, 0.),
            tl.load(ptr + offset + 2 * stride, mask, 0.))


@triton.jit
def _store_xyz(ptr, offset, mask, x, y, z):
    tl.store(ptr + offset, x, mask)
    tl.store(ptr + offset + 1, y, mask)
    tl.store(ptr + offset + 2, z, mask)


@triton.jit
def _flight(P, V, D, G, W, OUT_P, OUT_V, count, ps0, ps1, vs0, vs1, ds,
            STEPS: tl.constexpr, DRAG: tl.constexpr, DT: tl.constexpr,
            PATH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < count
    px, py, pz = _load_xyz(P, lane * ps0, ps1, mask)
    vx, vy, vz = _load_xyz(V, lane * vs0, vs1, mask)
    gx, gy, gz = _load_xyz(G, 0, 1, True)
    wx, wy, wz = _load_xyz(W, 0, 1, True)
    if PATH:
        start = lane * (STEPS + 1) * 3
        _store_xyz(OUT_P, start, mask, px, py, pz)
        _store_xyz(OUT_V, start, mask, vx, vy, vz)
    else:
        remaining = tl.load(D + lane * ds, mask, 0.)

    for step in range(STEPS):
        if PATH:
            dt = DT
        else:
            dt = tl.where(remaining != remaining, remaining, tl.minimum(tl.maximum(remaining, 0.), DT))
            remaining = remaining - dt
        rx, ry, rz = vx - wx, vy - wy, vz - wz
        # Match the reduction tree of torch.linalg.vector_norm on the three CUDA components.
        drag_speed = DRAG * tl.sqrt_rn((rx * rx + rz * rz) + ry * ry)
        vx = vx + dt * (gx - drag_speed * rx)
        vy = vy + dt * (gy - drag_speed * ry)
        vz = vz + dt * (gz - drag_speed * rz)
        px, py, pz = px + dt * vx, py + dt * vy, pz + dt * vz
        if PATH:
            offset = start + (step + 1) * 3
            _store_xyz(OUT_P, offset, mask, px, py, pz)
            _store_xyz(OUT_V, offset, mask, vx, vy, vz)

    if not PATH:
        _store_xyz(OUT_P, lane * 3, mask, px, py, pz)
        _store_xyz(OUT_V, lane * 3, mask, vx, vy, vz)


class ResearchCudaImpulsePrior(BallImpulsePriorV1):
    def _can_fuse(self, p, v, duration=None):
        if not p.is_cuda or p.ndim != 2 or p.shape[1] != 3 or v.shape != p.shape or not len(p):
            return False
        if (torch.autograd.forward_ad._current_level >= 0 or torch._C._are_functorch_transforms_active()
                or v.stride(1) != 1):
            return False
        tensors = (p, v, self.wind, self.gravity)
        if duration is not None:
            if duration.shape != p.shape[:1]:
                return False
            tensors += (duration,)
        return (self.wind.is_contiguous() and self.gravity.is_contiguous()
                and all(t.dtype == torch.float32 and t.device == p.device for t in tensors)
                and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors)))

    def _integrate(self, p, v, duration=None):
        path = duration is None
        shape = (len(p), self.steps + 1, 3) if path else p.shape
        out_p, out_v = p.new_empty(shape), v.new_empty(shape)
        with torch.cuda.device(p.device):
            _flight[(triton.cdiv(len(p), 32),)](
                p, v, p if path else duration, self.gravity, self.wind, out_p, out_v,
                len(p), *p.stride(), *v.stride(), 0 if path else duration.stride(0),
                self.steps, self.drag, self.sub_dt, path, 32, num_warps=1, enable_fp_fusion=False)
        return out_p, out_v

    def advance(self, p, v, duration):
        if self._can_fuse(p, v, duration):
            return self._integrate(p, v, duration)
        return super().advance(p, v, duration)

    def advance_trajectory(self, p, v):
        if self._can_fuse(p, v):
            return self._integrate(p, v)
        return super().advance_trajectory(p, v)


def enable_research_flight(action_model, expected_backend=None):
    prior = getattr(action_model, 'impulse_prior', None)
    if type(prior) is not BallImpulsePriorV1:
        raise ValueError('Fused research flight requires the loaded single-impact prior')
    backend = dict(schema='research-fp32-flight-v1', source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch=str(torch.__version__), triton=triton.__version__)
    if expected_backend is not None and backend != expected_backend:
        raise ValueError('Resumed research flight backend differs from checkpoint')
    fast = ResearchCudaImpulsePrior(**prior.calibration_manifest()).to(prior.gravity)
    fast.gravity.copy_(prior.gravity)
    fast.wind.copy_(prior.wind)
    fast.requires_grad_(False).train(prior.training)
    action_model.impulse_prior = fast
    return backend
