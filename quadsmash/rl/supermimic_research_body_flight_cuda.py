"""Fused forward of the Body WM's native-force free-flight prior for research rollouts."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from . import supermimic_research_flight_cuda as impulse_backend
from .supermimic_research_flight_cuda import _load_xyz, _store_xyz
from .supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1


@triton.jit
def _body_flight(P, V, G, W, OUT_P, OUT_V, count, ps0, ps1, vs0, vs1,
                 STEPS: tl.constexpr, DT: tl.constexpr, FORCE_FACTOR: tl.constexpr,
                 INV_MASS: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < count
    px, py, pz = _load_xyz(P, lane * ps0, ps1, mask)
    vx, vy, vz = _load_xyz(V, lane * vs0, vs1, mask)
    gx, gy, gz = _load_xyz(G, 0, 1, True)
    wx, wy, wz = _load_xyz(W, 0, 1, True)
    for _ in range(STEPS):
        rx, ry, rz = vx - wx, vy - wy, vz - wz
        # Match native.drag_force followed by Torch FP32 scalar division.
        force = FORCE_FACTOR * tl.sqrt_rn((rx * rx + rz * rz) + ry * ry)
        vx = vx + DT * (gx + (force * rx) * INV_MASS)
        vy = vy + DT * (gy + (force * ry) * INV_MASS)
        vz = vz + DT * (gz + (force * rz) * INV_MASS)
        px, py, pz = px + DT * vx, py + DT * vy, pz + DT * vz
    _store_xyz(OUT_P, lane * 3, mask, px, py, pz)
    _store_xyz(OUT_V, lane * 3, mask, vx, vy, vz)


class ResearchBodyFlight(ShuttleFreeFlightV1):
    def __init__(self, flight):
        torch.nn.Module.__init__(self)
        self.steps, self.dt = flight.steps, flight.dt
        self.mass, self.coefficient = flight.mass, flight.coefficient
        self.register_buffer('gravity', flight.gravity)
        self.register_buffer('wind', flight.wind)
        self.inverse_mass = float(np.float32(1.) / np.float32(self.mass))

    def forward(self, position, velocity):
        supported = (position.is_cuda and position.ndim == 2 and position.shape[1] == 3 and len(position) > 0
                     and velocity.shape == position.shape and velocity.stride(1) == 1
                     and self.gravity.is_contiguous() and self.wind.is_contiguous()
                     and torch.autograd.forward_ad._current_level < 0 and not torch._C._are_functorch_transforms_active())
        tensors = (position, velocity, self.wind, self.gravity)
        if (not supported or any(t.dtype != torch.float32 or t.device != position.device for t in tensors)
                or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
            return super().forward(position, velocity)
        p, v = position.new_empty(position.shape), velocity.new_empty(velocity.shape)
        with torch.cuda.device(position.device):
            _body_flight[(triton.cdiv(len(p), 32),)](
                position, velocity, self.gravity, self.wind, p, v, len(p),
                *position.stride(), *velocity.stride(), self.steps, self.dt,
                -self.mass * self.coefficient, self.inverse_mass, 32, num_warps=1, enable_fp_fusion=False)
        return p, v


def enable_body_flight(physical_model, expected_backend=None):
    source = physical_model.flight
    if type(source) is not ShuttleFreeFlightV1:
        raise ValueError('Fused Body flight requires the native-force shuttle prior')
    backend = dict(schema='research-body-force-fp32-v1', source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   helpers_sha256=sha256(Path(impulse_backend.__file__).read_bytes()).hexdigest(),
                   torch=str(torch.__version__), triton=triton.__version__)
    if expected_backend is not None and expected_backend != backend:
        raise ValueError('Resumed Body flight backend differs from checkpoint')
    physical_model.flight = ResearchBodyFlight(source).train(source.training)
    return backend
