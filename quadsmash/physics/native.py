"""Shared force, geometry and event accounting for native PhysX shuttle tasks.

This module never moves a ball or applies contact impulses. PhysX handles
collisions; contact flags require filtered engine forces, not proximity.
"""
from pathlib import Path
import json
import torch


def physics_contract():
    return json.loads((Path(__file__).resolve().parents[2] / "configs/shuttle_physics_v2.json").read_text())


def drag_force(velocity, mass, coefficient, wind):
    relative = velocity - torch.as_tensor(wind, device=velocity.device, dtype=velocity.dtype)
    return -mass * coefficient * relative.norm(dim=-1, keepdim=True) * relative


def quat_rotate(q, vector):
    xyz = q[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)


def sphere_box_gap(ball_position, box_position, box_quaternion, half_extents, radius):
    inverse = torch.cat((box_quaternion[..., :1], -box_quaternion[..., 1:]), dim=-1)
    local = quat_rotate(inverse, ball_position - box_position)
    delta = local.abs() - torch.as_tensor(half_extents, device=local.device, dtype=local.dtype)
    return delta.clamp_min(0).norm(dim=-1) + delta.max(dim=-1).values.clamp_max(0) - radius


def swept_sphere_box(start, end, center_start, center_end, quat_start, quat_end,
                     half_extents, radius, iterations=256, tolerance=1e-6):
    """Batched conservative advancement for translating and rotating blades.

    Returns hit, first-contact fraction, unresolved. Unresolved grazing cases
    are explicitly reported; they must never silently become confident misses.
    This is a detection/audit query, not a second collision-response engine.
    """
    q0 = quat_start / quat_start.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = quat_end / quat_end.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (q0*q1).sum(-1)
    q1 = torch.where((dot<0)[:, None], -q1, q1)
    theta = torch.acos(dot.abs().clamp(0, 1))
    half = torch.as_tensor(half_extents, device=start.device, dtype=start.dtype)
    bound = ((end-start)-(center_end-center_start)).norm(dim=-1) + 2*theta*half.norm()
    fraction = torch.zeros_like(bound)
    hit = torch.zeros_like(bound, dtype=torch.bool)
    # Cheap bounding-sphere rejection keeps far-away balls out of the loop.
    a, direction = start-center_start, (end-start)-(center_end-center_start)
    t = (-(a*direction).sum(-1)/direction.square().sum(-1).clamp_min(1e-20)).clamp(0,1)
    active = (a+t[:,None]*direction).norm(dim=-1) <= half.norm()+radius+tolerance
    sine = theta.sin().clamp_min(1e-8)
    for _ in range(iterations):
        if not bool(active.any()):
            break
        linear = (1-fraction[:,None])*q0+fraction[:,None]*q1
        spherical = (((1-fraction)*theta).sin()[:,None]*q0+(fraction*theta).sin()[:,None]*q1)/sine[:,None]
        q = torch.where((theta<1e-5)[:,None],linear,spherical)
        gap = sphere_box_gap(start+fraction[:,None]*(end-start),
            center_start+fraction[:,None]*(center_end-center_start),q,half,radius)
        new = active & (gap<=tolerance)
        hit |= new
        proposed = fraction+(.9*gap/bound.clamp_min(1e-15)).clamp_min(1e-9)
        active &= (~new) & (proposed<=1) & (bound>1e-15)
        fraction = torch.where(active,proposed,fraction)
    return hit, fraction, active


def segment_plane_crossing(previous, current, axis, coordinate, direction=1):
    """Interpolate a directed center crossing; returns mask, fraction, point."""
    a, b = previous[..., axis] - coordinate, current[..., axis] - coordinate
    mask = (a * direction < 0) & (b * direction >= 0)
    denominator = b - a
    fraction = torch.where(mask, -a / torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(b)), torch.zeros_like(b)).clamp(0, 1)
    point = previous + fraction[..., None] * (current - previous)
    return mask, fraction, point


class ContactEpisodes:
    """Count force-confirmed contacts, rearming after measured separation."""

    def __init__(self, count, device, threshold=0.001, separation=0.002):
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.count = torch.zeros(count, dtype=torch.long, device=device)
        self.clear_frames = torch.zeros(count, dtype=torch.long, device=device)
        self.threshold = threshold
        self.separation = separation

    def reset(self, ids):
        self.active[ids] = False
        self.count[ids] = 0
        self.clear_frames[ids] = 0

    def update(self, filtered_force, gap):
        touching = filtered_force.norm(dim=-1) > self.threshold
        separated = (~touching) & (gap > self.separation)
        self.clear_frames.copy_(torch.where(separated, self.clear_frames + 1, torch.zeros_like(self.clear_frames)))
        self.active &= self.clear_frames < 2
        new = touching & (~self.active)
        self.count += new.long()
        self.active |= touching
        return new, touching


def configure_ccd(stage, body_paths, collider_paths, material_path=None):
    """Set engine CCD and small contact offsets before starting simulation."""
    from pxr import PhysxSchema, UsdShade
    cfg = physics_contract()
    for path in body_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"Missing collision body: {path}")
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateEnableCCDAttr(cfg["sweep_ccd"])
        api.CreateEnableSpeculativeCCDAttr(cfg["speculative_ccd"])
    for path in collider_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"Missing collider: {path}")
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(cfg["contact_offset_m"])
        api.CreateRestOffsetAttr(cfg["rest_offset_m"])
        if material_path:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
