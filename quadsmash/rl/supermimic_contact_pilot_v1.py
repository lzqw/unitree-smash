"""Build a measured Racket->Ball contact pilot from native substep traces.

The pilot is deliberately separate from the control-boundary event dataset. It
only accepts traces that contain the consumed PhysX substeps and keeps missing
sensor fields explicit.  A row is one contiguous racket-contact episode, with
the first touching substep used as the TOI witness.  Near-miss rows are chosen
from windows with no measured racket force and retain the closest gap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1


SCHEMA = "supermimic-racket-ball-contact-pilot-v1"
PILOT_FIELDS = (
    "toi_frame", "toi_time_s", "contact_object", "contact_point_w", "normal_w",
    "gap_m", "relative_velocity_w", "impulse_ns", "normal_impulse_ns",
    "tangential_impulse_ns", "penetration_m", "contact_order", "effective_mass_kg",
    "racket_part", "ball_state_before", "ball_state_after", "racket_state_before",
    "racket_state_after", "toi_pre_substep", "toi_post_substep",
    "outgoing_direction_w", "outgoing_speed_mps", "normal_source",
    "impulse_source", "effective_mass_source", "interference_objects",
    "spec_id", "episode_id", "capture_id", "lane_family", "lane_hash",
    "split_group", "split", "gap_bucket", "approach_speed_mps", "relative_speed_mps",
    "contact_slot_count", "contact_slot_indices", "slot_impulse_ns", "gap_source",
    "contact_region", "contact_grid", "toi_phase", "contact_count_substeps",
    "physics_dt_s", "quality_flags", "bucket", "interference_object",
    "contact_filter_paths", "racket_part_source",
    "probe_branch", "event_id",
)


_GEOMETRY_PREVIEW: BallGeometryPreviewV1 | None = None


def _geometry_preview() -> BallGeometryPreviewV1:
    global _GEOMETRY_PREVIEW
    if _GEOMETRY_PREVIEW is None:
        _GEOMETRY_PREVIEW = BallGeometryPreviewV1().eval()
    return _GEOMETRY_PREVIEW


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _first_trace(value: Any) -> dict | None:
    """Find a native-one-control-tick trace in a nested archive."""
    if isinstance(value, dict):
        if value.get("schema") == "native-one-control-tick-substeps-v1":
            return value
        for item in value.values():
            found = _first_trace(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_trace(item)
            if found is not None:
                return found
    return None


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    qv = torch.stack((x, y, z), -1)
    return vector + 2.0 * torch.linalg.cross(qv, torch.linalg.cross(qv, vector) - w[..., None] * vector)


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    qv = torch.stack((x, y, z), -1)
    return vector + 2.0 * torch.linalg.cross(qv, w[..., None] * vector + torch.linalg.cross(qv, vector))


def _frame_state(tensors: dict, index: int, lane: int) -> dict:
    ball = _as_tensor(tensors["shuttle_state_w"])[index, lane].tolist()
    racket_pos = _as_tensor(tensors["racket_position_w"])[index, lane].tolist()
    racket_quat = _as_tensor(tensors["racket_quaternion_w"])[index, lane].tolist()
    racket_lin = _as_tensor(tensors["racket_linear_velocity_w"])[index, lane].tolist()
    racket_ang = _as_tensor(tensors["racket_angular_velocity_w"])[index, lane].tolist()
    return {
        "ball_state_w": ball,
        "racket_position_w": racket_pos,
        "racket_quaternion_wxyz": racket_quat,
        "racket_linear_velocity_w": racket_lin,
        "racket_angular_velocity_w": racket_ang,
    }


def _force_tensor(tensors: dict, name: str, frames: int, lanes: int) -> torch.Tensor:
    if name not in tensors:
        return torch.zeros(frames, lanes, 3, dtype=torch.float32)
    value = _as_tensor(tensors[name]).float()
    while value.ndim > 3:
        value = value[..., 0, :]
    if value.ndim == 2:
        value = value[:, None, :]
    if value.shape[-1] != 3:
        raise ValueError(f"{name} must end in xyz, got {tuple(value.shape)}")
    if value.shape[:2] != (frames, lanes):
        raise ValueError(f"{name} shape does not align with substep state")
    return value


def _force_slots(tensors: dict, matrix_name: str, legacy_name: str,
                 frames: int, lanes: int) -> torch.Tensor:
    """Return [frames, lanes, slots, xyz] without dropping manifold slots."""
    name = matrix_name if matrix_name in tensors else legacy_name
    if name not in tensors:
        return torch.zeros(frames, lanes, 1, 3, dtype=torch.float32)
    value = _as_tensor(tensors[name]).float()
    # Native contact matrices are [frame, lane, body, slot, xyz].  The
    # racket body is body zero.  A reduced archive may already be
    # [frame, lane, slot, xyz] or [frame, lane, xyz].
    if value.ndim == 5:
        value = value[:, :, 0]
    elif value.ndim == 4 and value.shape[-1] != 3:
        value = value[:, :, 0]
    if value.ndim == 3:
        value = value.unsqueeze(2)
    if value.ndim != 4 or value.shape[:2] != (frames, lanes) or value.shape[-1] != 3:
        raise ValueError(f"{name} must align as [frames, lanes, slots, 3], got {tuple(value.shape)}")
    return value


def _position_slots(tensors: dict, frames: int, lanes: int) -> torch.Tensor:
    name = "contact_position_matrix_w" if "contact_position_matrix_w" in tensors else "racket_contact_position_w"
    if name not in tensors:
        return torch.full((frames, lanes, 1, 3), float("nan"))
    value = _as_tensor(tensors[name]).float()
    if value.ndim == 5:
        value = value[:, :, 0]
    elif value.ndim == 4 and value.shape[-1] != 3:
        value = value[:, :, 0]
    if value.ndim == 3:
        value = value.unsqueeze(2)
    if value.ndim != 4 or value.shape[:2] != (frames, lanes) or value.shape[-1] != 3:
        raise ValueError(f"{name} must align as [frames, lanes, slots, 3], got {tuple(value.shape)}")
    return value


def _first_finite_slot(points: torch.Tensor, start: int, lane: int) -> torch.Tensor:
    values = points[start, lane]
    finite = torch.isfinite(values).all(-1)
    if bool(finite.any()):
        return values[finite.nonzero().flatten()[0]]
    return torch.full((3,), float("nan"), dtype=values.dtype)


def _source_metadata(path: Path, source_hash: str, lane: int,
                     probe_branch: str | None = None) -> dict[str, str]:
    text = str(path.resolve())
    names = path.resolve().parts
    spec_match = next((part for part in reversed(names)
                       if re.search(r"(?:^|[-_])spec[-_]?\d+", part, re.I)), None)
    episode_match = re.search(r"(?:episode|ep)[-_]?(\d+)", text, re.I)
    spec_id = spec_match or "spec_unknown"
    if probe_branch is None:
        probe_branch = next((part for part in reversed(names)
                             if part in {"nominal_a", "nominal_b", "minus", "plus"}), "unknown")
    episode_id = f"episode_{episode_match.group(1)}" if episode_match else f"episode_{source_hash[:12]}"
    capture_id = f"capture_{source_hash[:12]}"
    lane_family = f"lane_family_{lane // 8}"
    lane_hash = hashlib.sha256(f"{source_hash}:{lane}".encode()).hexdigest()[:16]
    split_group = "|".join((spec_id, episode_id, lane_family, capture_id))
    return dict(spec_id=spec_id, episode_id=episode_id, capture_id=capture_id,
                lane_family=lane_family, lane_hash=lane_hash, split_group=split_group,
                probe_branch=str(probe_branch), event_id=f"{spec_id}:{probe_branch}")


def _analytic_gap(ball_pos: torch.Tensor, racket_pos: torch.Tensor,
                  quaternion: torch.Tensor) -> float:
    local = _quat_rotate_inverse(quaternion[None], (ball_pos - racket_pos)[None])[0]
    return float(_geometry_preview().racket_gap(local[None])[0])


def _band(value: float, edges: tuple[float, ...]) -> int:
    if not math.isfinite(value):
        return -1
    return sum(value >= edge for edge in edges)


def _gap_bucket(gap: float) -> str:
    return f"gap{_band(gap, (0.001, 0.005, 0.02, 0.05))}"


def _approach_bucket(speed: float) -> str:
    return f"approach{_band(speed, (0.5, 2., 5., 10.))}"


def _contact_intervals(mask: torch.Tensor) -> list[tuple[int, int]]:
    padded = torch.cat((torch.zeros(1, dtype=torch.bool), mask,
                        torch.zeros(1, dtype=torch.bool)))
    starts = (padded[1:] & ~padded[:-1]).nonzero().flatten().tolist()
    ends = ((~padded[1:]) & padded[:-1]).nonzero().flatten().tolist()
    return list(zip(starts, ends))


def _contact_region(local: torch.Tensor) -> tuple[str, tuple[int, int] | None]:
    yz = local[1:3]
    normalized = yz / 0.11
    if not torch.isfinite(normalized).all():
        return "unknown", None
    if float(normalized.abs().max()) > 0.9:
        return "edge", None
    if float(normalized.norm()) > 0.75:
        return "rim", None
    grid = torch.floor((normalized.clamp(-.999, .999) + 1.) * 2.5).to(torch.int64)
    ij = (int(grid[0]), int(grid[1]))
    return ("center" if ij == (2, 2) else "grid_5x5"), ij


def _bucket(row: dict) -> str:
    rel = torch.tensor(row["relative_velocity_w"], dtype=torch.float32)
    normal = torch.tensor(row["normal_w"], dtype=torch.float32)
    vn = abs(float((rel * normal).sum()))
    vt = float((rel - (rel * normal).sum() * normal).norm())
    racket = row["racket_state_before"]
    lin = torch.tensor(racket["racket_linear_velocity_w"], dtype=torch.float32).norm().item()
    ang = torch.tensor(racket["racket_angular_velocity_w"], dtype=torch.float32).norm().item()
    def band(x: float, edges: tuple[float, ...]) -> int:
        return sum(x >= edge for edge in edges)
    return ":".join((
        row.get("contact_region", "unknown"),
        str(row.get("contact_grid", "-")),
        row.get("racket_part", "unknown"),
        f"vn{band(vn, (1., 5., 10.))}",
        f"vt{band(vt, (1., 5., 10.))}",
        f"lin{band(lin, (.5, 2., 5.))}",
        f"ang{band(ang, (1., 5., 15.))}",
        f"phase{int(row.get('toi_phase', 0))}",
        f"order{min(int(row.get('contact_order', 1)), 2)}",
        row.get("gap_bucket", "gap-1"),
        _approach_bucket(float(row.get("approach_speed_mps", float("nan")))),
    ))


def _trace_candidates(path: Path, force_threshold: float, near_miss_radius: int,
                      sensitivity_thresholds: tuple[float, ...] = (),
                      max_near_miss_per_lane: int = 128) -> tuple[list[dict], dict]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    trace = _first_trace(package)
    if trace is None:
        return [], {}
    tensors = trace.get("tensors", {})
    required = ("shuttle_state_w", "racket_position_w", "racket_quaternion_w",
                "racket_linear_velocity_w", "racket_angular_velocity_w")
    if any(key not in tensors for key in required):
        return [], {}
    ball = _as_tensor(tensors["shuttle_state_w"]).float()
    frames, lanes = ball.shape[:2]
    dt = float(trace.get("physics_dt_s", 0.0))
    if dt <= 0:
        raise ValueError(f"{path}: trace physics_dt_s must be positive")
    racket_force_slots = _force_slots(tensors, "contact_force_matrix_w", "racket_force_w", frames, lanes)
    racket_force = racket_force_slots.sum(dim=2)
    ground_force = _force_tensor(tensors, "ground_force_w", frames, lanes)
    net_force_slots = _force_slots(tensors, "net_force_matrix_w", "net_force_w", frames, lanes)
    net_force = net_force_slots.sum(dim=2)
    position_all = _as_tensor(tensors["racket_position_w"]).float()
    quaternion_all = _as_tensor(tensors["racket_quaternion_w"]).float()
    local_all = _quat_rotate_inverse(quaternion_all, ball[..., :3] - position_all)
    analytic_gaps = _geometry_preview().racket_gap(local_all)
    point_slots = _position_slots(tensors, frames, lanes)
    touching = racket_force.norm(dim=-1) > force_threshold
    ground_touching = ground_force.norm(dim=-1) > force_threshold
    net_touching = net_force.norm(dim=-1) > force_threshold
    records: list[dict] = []
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    probe_branch = package.get("branch")
    contact_metadata = trace.get("contact_metadata", {})
    filter_paths = contact_metadata.get("filter_prim_paths")
    racket_body_name = contact_metadata.get("racket_body_name")
    racket_part = racket_body_name or "unknown"
    racket_part_source = ("native_contact_metadata" if racket_body_name else "unavailable_in_archive")
    source_meta = [_source_metadata(path, source_hash, lane, probe_branch) for lane in range(lanes)]
    sensitivity: dict[str, dict[str, int]] = {}
    for threshold in sensitivity_thresholds:
        threshold_touch = racket_force.norm(dim=-1) > threshold
        sensitivity[str(threshold)] = {
            "active_substeps": int(threshold_touch.sum()),
            "lanes_with_contact": int(threshold_touch.any(0).sum()),
            "contact_intervals": sum(len(_contact_intervals(threshold_touch[:, lane])) for lane in range(lanes)),
        }
    for lane in range(lanes):
        mask = touching[:, lane]
        starts = torch.where(mask & ~torch.cat((torch.zeros(1, dtype=torch.bool), mask[:-1])))[0].tolist()
        ends = torch.where(mask & ~torch.cat((mask[1:], torch.zeros(1, dtype=torch.bool))))[0].tolist()
        for order, (start, end) in enumerate(zip(starts, ends), 1):
            before = max(start - 1, 0)
            after = min(end + 1, frames - 1)
            force = racket_force[start:end + 1, lane]
            impulse = force.sum(0) * dt
            normal = force.mean(0)
            position = position_all[start, lane]
            quaternion = quaternion_all[start, lane]
            if float(normal.norm()) < 1e-12:
                normal = ball[start, lane, :3] - position
            normal = normal / normal.norm().clamp_min(1e-12)
            contact_point = _first_finite_slot(point_slots, start, lane)
            arm = contact_point - position
            racket_lin = _as_tensor(tensors["racket_linear_velocity_w"])[start, lane].float()
            racket_ang = _as_tensor(tensors["racket_angular_velocity_w"])[start, lane].float()
            ball_vel = ball[start, lane, 7:10]
            if not torch.isfinite(arm).all():
                arm = torch.zeros(3, dtype=position.dtype)
            surface = racket_lin + torch.linalg.cross(racket_ang, arm)
            relative = ball_vel - surface
            vn = float((relative * normal).sum())
            normal_impulse = float((impulse * normal).sum())
            tangent_impulse = impulse - normal_impulse * normal
            mass = normal_impulse / max(-(1.72) * vn, 1e-9) if vn < 0 else float("nan")
            gap = (float((ball[start, lane, :3] - contact_point).norm() - .025)
                   if torch.isfinite(contact_point).all()
                   else float(analytic_gaps[start, lane]))
            local = _quat_rotate_inverse(quaternion[None], ((contact_point - position) if torch.isfinite(contact_point).all()
                                                            else (ball[start, lane, :3] - position))[None])[0]
            region, grid = _contact_region(local)
            active_slots = racket_force_slots[start:end + 1, lane].norm(dim=-1).max(0).values > force_threshold
            slot_impulse = racket_force_slots[start:end + 1, lane].sum(0) * dt
            metadata = source_meta[lane]
            row = dict(
                source=str(path), source_sha256=source_hash, lane=lane,
                toi_frame=int(start), toi_time_s=float(start * dt),
                contact_object="racket", contact_point_w=contact_point.tolist(),
                normal_w=normal.tolist(), gap_m=gap,
                relative_velocity_w=relative.tolist(), impulse_ns=impulse.tolist(),
                normal_impulse_ns=normal_impulse, tangential_impulse_ns=tangent_impulse.tolist(),
                penetration_m=max(0., -gap), contact_order=order,
                effective_mass_kg=mass, racket_part=racket_part,
                contact_region=region, contact_grid=grid, toi_phase=int((start % 160) / 40),
                ball_state_before=_frame_state(tensors, before, lane),
                ball_state_after=_frame_state(tensors, after, lane),
                racket_state_before=_frame_state(tensors, before, lane),
                racket_state_after=_frame_state(tensors, after, lane),
                toi_pre_substep=_frame_state(tensors, before, lane),
                toi_post_substep=_frame_state(tensors, after, lane),
                outgoing_direction_w=(ball[after, lane, 7:10] / ball[after, lane, 7:10].norm().clamp_min(1e-12)).tolist(),
                outgoing_speed_mps=float(ball[after, lane, 7:10].norm()),
                quality_flags=["native_substep_force_integrated"] + (["racket_part_unavailable"] if not racket_body_name else []),
                normal_source="racket_force_direction_proxy",
                impulse_source="integrated_contact_force_matrix",
                effective_mass_source="inferred_from_normal_impulse_and_relative_speed",
                interference_objects=[name for name, active in (
                    ("ground", bool(ground_touching[start:end + 1, lane].any())),
                    ("net", bool(net_touching[start:end + 1, lane].any())),
                ) if active],
                contact_count_substeps=int(end - start + 1),
                contact_slot_count=int(active_slots.sum()),
                contact_slot_indices=[int(index) for index in active_slots.nonzero().flatten().tolist()],
                slot_impulse_ns=slot_impulse.tolist(),
                gap_source="measured_contact_point" if torch.isfinite(contact_point).all() else "analytic_racket_geometry",
                approach_speed_mps=max(0., -vn), relative_speed_mps=float(relative.norm()),
                gap_bucket=_gap_bucket(gap),
                physics_dt_s=dt,
                contact_filter_paths=filter_paths,
                racket_part_source=racket_part_source,
                **metadata,
            )
            row["bucket"] = _bucket(row)
            records.append(row)
        # Sample hard negatives even in lanes that contain another contact.
        # Only a local neighborhood around a positive interval is excluded.
        lane_near_miss: list[dict] = []
        for center in range(0, frames, max(1, near_miss_radius)):
            if touching[max(0, center - near_miss_radius):min(frames, center + near_miss_radius + 1), lane].any():
                continue
            ball_pos = ball[center, lane, :3]
            racket_pos = position_all[center, lane]
            quaternion = quaternion_all[center, lane]
            racket_lin = _as_tensor(tensors["racket_linear_velocity_w"])[center, lane].float()
            racket_ang = _as_tensor(tensors["racket_angular_velocity_w"])[center, lane].float()
            normal = ball_pos - racket_pos
            normal = normal / normal.norm().clamp_min(1e-12)
            arm = ball_pos - racket_pos
            surface = racket_lin + torch.linalg.cross(racket_ang, arm)
            relative = ball[center, lane, 7:10] - surface
            vn = float((relative * normal).sum())
            gap = float(analytic_gaps[center, lane])
            interference = "ground" if float(ground_force[max(0, center - 1):min(frames, center + 2), lane].norm(dim=-1).max()) > force_threshold else (
                "net" if float(net_force[max(0, center - 1):min(frames, center + 2), lane].norm(dim=-1).max()) > force_threshold else "none")
            metadata = source_meta[lane]
            row = dict(source=str(path), source_sha256=source_hash, lane=lane,
                       toi_frame=int(center), toi_time_s=float(center * dt),
                       contact_object="no_contact", contact_point_w=[float("nan")] * 3,
                       normal_w=normal.tolist(), gap_m=gap,
                       relative_velocity_w=relative.tolist(), impulse_ns=[0.] * 3,
                       normal_impulse_ns=0., tangential_impulse_ns=[0.] * 3,
                       penetration_m=0., contact_order=0, effective_mass_kg=float("nan"),
                           racket_part=racket_part, contact_region="near_miss",
                       contact_grid=None, toi_phase=int((center % 160) / 40),
                       ball_state_before=_frame_state(tensors, max(0, center - 1), lane),
                       ball_state_after=_frame_state(tensors, min(frames - 1, center + 1), lane),
                       racket_state_before=_frame_state(tensors, max(0, center - 1), lane),
                       racket_state_after=_frame_state(tensors, min(frames - 1, center + 1), lane),
                       toi_pre_substep=_frame_state(tensors, max(0, center - 1), lane),
                       toi_post_substep=_frame_state(tensors, min(frames - 1, center + 1), lane),
                       outgoing_direction_w=[float("nan")] * 3, outgoing_speed_mps=float("nan"),
                       quality_flags=["native_substep_no_racket_force", "near_miss_candidate"],
                       normal_source="unavailable_without_manifold_normal",
                       impulse_source="zero_by_no_racket_force",
                       effective_mass_source="unavailable_without_contact",
                       interference_objects=[name for name, active in (
                           ("ground", bool(ground_touching[:, lane].any())),
                           ("net", bool(net_touching[:, lane].any())),
                       ) if active],
                       interference_object=interference,
                       contact_count_substeps=0, contact_slot_count=0,
                       contact_slot_indices=[], slot_impulse_ns=[[0., 0., 0.]],
                       gap_source="analytic_racket_geometry",
                       approach_speed_mps=max(0., -vn), relative_speed_mps=float(relative.norm()),
                       gap_bucket=_gap_bucket(gap), physics_dt_s=dt,
                           contact_filter_paths=filter_paths,
                           racket_part_source=racket_part_source,
                       **metadata)
            row["bucket"] = ":".join(("near_miss", interference, str(row["toi_phase"]),
                                        row["gap_bucket"], _approach_bucket(max(0., -vn))))
            lane_near_miss.append(row)
        if len(lane_near_miss) > max_near_miss_per_lane > 0:
            indices = torch.linspace(0, len(lane_near_miss) - 1, max_near_miss_per_lane).round().long().tolist()
            records.extend(lane_near_miss[index] for index in indices)
        else:
            records.extend(lane_near_miss)
    return records, sensitivity


def _select(rows: list[dict], target: int, seed: int) -> list[dict]:
    if target <= 0 or not rows:
        return []
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    grouped: dict[str, list[dict]] = {}
    for index in order:
        grouped.setdefault(rows[index].get("bucket", "unknown"), []).append(rows[index])
    chosen: list[dict] = []
    while len(chosen) < min(target, len(rows)) and grouped:
        for key in list(grouped):
            values = grouped[key]
            if values:
                chosen.append(values.pop())
                if len(chosen) == min(target, len(rows)):
                    break
            if not values:
                grouped.pop(key)
    return chosen


def _deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    unique = []
    seen = set()
    for row in rows:
        event_id = row.get("event_id")
        key = ((event_id, row.get("contact_object"), row.get("lane"), row.get("toi_frame"),
                row.get("contact_order")) if event_id else
               (row.get("source_sha256"), row.get("contact_object"), row.get("lane"),
                row.get("toi_frame"), row.get("contact_order")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique, len(rows) - len(unique)


def _coverage(rows: list[dict]) -> dict:
    positives = [row for row in rows if row.get("contact_object") == "racket"]
    negatives = [row for row in rows if row.get("contact_object") == "no_contact"]
    regions = {name: sum(row.get("contact_region") == name for row in positives)
               for name in ("center", "grid_5x5", "rim", "edge")}
    phases = {str(index): sum(int(row.get("toi_phase", -1)) == index for row in positives)
              for index in range(4)}
    orders = {str(index): sum(min(int(row.get("contact_order", 0)), 2) == index for row in positives)
              for index in (1, 2)}
    interference = {name: sum(name in row.get("interference_objects", []) for row in positives)
                    for name in ("ground", "net")}
    near_gap = {str(index): sum(row.get("gap_bucket") == f"gap{index}" for row in negatives)
                for index in range(5)}
    near_approach = {str(index): sum(row.get("bucket", "").endswith(f"approach{index}") for row in negatives)
                     for index in range(5)}
    # These are intentionally hard minima for the 2k+1k pilot.  A manifest
    # can therefore be INSUFFICIENT_COVERAGE even when raw candidate counts
    # are large enough.
    required = {
        "regions": {"center": 100, "grid_5x5": 400, "rim": 100, "edge": 200},
        "toi_phase": {str(index): 50 for index in range(4)},
        "contact_order": {"1": 1, "2": 100},
        "interference": {"ground": 25, "net": 25},
        "near_miss_gap": {str(index): 50 for index in range(5)},
        "near_miss_approach": {str(index): 50 for index in range(5)},
    }
    observed = {"regions": regions, "toi_phase": phases, "contact_order": orders,
                "interference": interference, "near_miss_gap": near_gap,
                "near_miss_approach": near_approach}
    missing = []
    for category, values in required.items():
        for key, minimum in values.items():
            if observed[category].get(key, 0) < minimum:
                missing.append(f"{category}:{key}<{minimum}")
    return {"observed": observed, "required": required, "missing": missing,
            "passed": not missing}


def _assign_splits(rows: list[dict], seed: int) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        # A source archive is atomic.  This prevents lane-family groups from
        # leaking the same capture into train and validation/test.
        groups.setdefault(row.get("source_sha256", "unknown"), []).append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    split_names = ("train", "validation", "test")
    assignment: dict[str, str] = {}
    for index, key in enumerate(ordered):
        bucket = index % 10
        assignment[key] = "train" if bucket < 7 else ("validation" if bucket < 9 else "test")
    # With at least three independent groups, guarantee all split names exist.
    for index, name in enumerate(split_names):
        if ordered and name not in assignment.values() and index < len(ordered):
            assignment[ordered[index]] = name
    known_specs = {row.get("spec_id") for row in rows if row.get("spec_id") != "spec_unknown"}
    if len(known_specs) >= 2:
        # Hold out one complete spec group.  All source archives belonging to
        # that spec move together, preserving source atomicity and making the
        # test spec genuinely unseen during fitting.
        holdout_spec = sorted(known_specs)[-1]
        for key in ordered:
            if any(row.get("spec_id") == holdout_spec for row in groups[key]):
                assignment[key] = "test"
    for row in rows:
        row["split"] = assignment.get(row.get("source_sha256"), "train")
    split_sources = {name: {row.get("source_sha256") for row in rows if row["split"] == name}
                     for name in split_names}
    overlap = {f"{left}:{right}": sorted(split_sources[left] & split_sources[right])
               for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    specs = {name: {row.get("spec_id") for row in rows if row["split"] == name}
             for name in split_names}
    unseen_test_specs = sorted((specs["test"] & known_specs) - specs["train"])
    return {
        "row_counts": {name: sum(row["split"] == name for row in rows) for name in split_names},
        "group_counts": {name: sum(value == name for value in assignment.values()) for name in split_names},
        "source_overlap": overlap,
        "source_overlap_free": not any(overlap.values()),
        "specs": {name: sorted(value) for name, value in specs.items()},
        "unseen_test_specs": unseen_test_specs,
        "unseen_spec_holdout": len(known_specs) < 2 or bool(unseen_test_specs),
    }


def build_pilot(inputs: str | Path | list[str | Path], output: str | Path, *, contacts: int = 2000,
                negatives: int = 1000, workers: int = 1, seed: int = 20261120,
                force_threshold: float = .001, near_miss_radius: int = 4) -> dict:
    """Build the pilot archive and return its manifest."""
    if isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(item) for item in inputs]
    files = sorted({p for root in paths for p in (root.rglob("*.pt") if root.is_dir() else [root])})
    if not files:
        raise FileNotFoundError("No .pt native substep traces found")
    sensitivity_thresholds = (0.0001, 0.001, 0.005, 0.01)
    fn = lambda path: _trace_candidates(path, force_threshold, near_miss_radius, sensitivity_thresholds)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(fn, files))
    else:
        groups = [fn(path) for path in files]
    rows, duplicate_rows_removed = _deduplicate([row for group, _ in groups for row in group])
    positives = [row for row in rows if row["contact_object"] == "racket"]
    negatives_rows = [row for row in rows if row["contact_object"] == "no_contact"]
    selected = _select(positives, contacts, seed) + _select(negatives_rows, negatives, seed + 1)
    selected.sort(key=lambda row: (row["contact_object"], row["source"], row["lane"], row["toi_frame"]))
    split = _assign_splits(selected, seed)
    coverage = _coverage(selected)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(schema=SCHEMA, rows=selected,
                   field_schema=list(PILOT_FIELDS), source_files=[str(p.resolve()) for p in files],
                   source_count=len(files), substep_truth=True, split=split,
                   policy_training_admitted=False, contact_bptt="detached",
                   v13_residual_gradient="detached")
    torch.save(payload, output / "pilot.pt")
    buckets: dict[str, int] = {}
    for row in selected:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
    sensitivity: dict[str, dict[str, int]] = {}
    for _, stats in groups:
        for threshold, values in stats.items():
            target = sensitivity.setdefault(threshold, {"active_substeps": 0, "lanes_with_contact": 0,
                                                         "contact_intervals": 0})
            for key in target:
                target[key] += int(values.get(key, 0))
    status = ("INSUFFICIENT_CANDIDATES" if len(positives) < contacts or len(negatives_rows) < negatives
              else ("INSUFFICIENT_COVERAGE" if not coverage["passed"] or not split["source_overlap_free"]
                    or not split["unseen_spec_holdout"] else "READY"))
    manifest = dict(schema=SCHEMA, requested_contacts=contacts, requested_negatives=negatives,
                    selected_contacts=sum(r["contact_object"] == "racket" for r in selected),
                    selected_negatives=sum(r["contact_object"] == "no_contact" for r in selected),
                    candidate_contacts=len(positives), candidate_negatives=len(negatives_rows),
                    duplicate_rows_removed=duplicate_rows_removed,
                    workers=int(workers), seed=int(seed), force_threshold_n=float(force_threshold),
                    bucket_counts=buckets, substep_truth=True, threshold_sensitivity=sensitivity,
                    coverage=coverage, split=split,
                    coverage_requirements={
                        "contact_region": ["center", "grid_5x5", "rim", "edge"],
                        "normal_incidence": "vn bands in bucket",
                        "tangential_relative_speed": "vt bands in bucket",
                        "racket_linear_speed": "lin bands in bucket",
                        "racket_angular_speed": "ang bands in bucket",
                        "toi_phase": "phase bands in bucket",
                        "racket_part": "recorded when source exposes part, otherwise explicit unknown",
                        "multiple_contact": "contact_order and contact_count_substeps",
                        "interference": ["none", "ground", "net"],
                    },
                    fields=list(PILOT_FIELDS), quality_policy="missing fields are explicit NaN and quality_flags",
                    admission={"policy_training_admitted": False, "contact_bptt": "detached",
                               "v13_residual_gradient": "detached"},
                    pilot_status=status)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
