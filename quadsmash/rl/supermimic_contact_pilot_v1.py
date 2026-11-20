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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch


SCHEMA = "supermimic-racket-ball-contact-pilot-v1"
PILOT_FIELDS = (
    "toi_frame", "toi_time_s", "contact_object", "contact_point_w", "normal_w",
    "gap_m", "relative_velocity_w", "impulse_ns", "normal_impulse_ns",
    "tangential_impulse_ns", "penetration_m", "contact_order", "effective_mass_kg",
    "racket_part", "ball_state_before", "ball_state_after", "racket_state_before",
    "racket_state_after", "toi_pre_substep", "toi_post_substep",
    "outgoing_direction_w", "outgoing_speed_mps", "normal_source",
    "impulse_source", "effective_mass_source", "interference_objects",
)


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
    ))


def _trace_candidates(path: Path, force_threshold: float, near_miss_radius: int) -> list[dict]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    trace = _first_trace(package)
    if trace is None:
        return []
    tensors = trace.get("tensors", {})
    required = ("shuttle_state_w", "racket_position_w", "racket_quaternion_w",
                "racket_linear_velocity_w", "racket_angular_velocity_w")
    if any(key not in tensors for key in required):
        return []
    ball = _as_tensor(tensors["shuttle_state_w"]).float()
    frames, lanes = ball.shape[:2]
    dt = float(trace.get("physics_dt_s", 0.0))
    if dt <= 0:
        raise ValueError(f"{path}: trace physics_dt_s must be positive")
    racket_force = _force_tensor(tensors, "racket_force_w", frames, lanes)
    ground_force = _force_tensor(tensors, "ground_force_w", frames, lanes)
    net_force = _force_tensor(tensors, "net_force_matrix_w", frames, lanes)
    point = _as_tensor(tensors.get("racket_contact_position_w", torch.full((frames, lanes, 3), float("nan")))).float()
    while point.ndim > 3:
        point = point[..., 0, :]
    valid = torch.isfinite(point).all(-1)
    touching = (racket_force.norm(dim=-1) > force_threshold) & valid
    ground_touching = ground_force.norm(dim=-1) > force_threshold
    net_touching = net_force.norm(dim=-1) > force_threshold
    records: list[dict] = []
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
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
            normal = normal / normal.norm().clamp_min(1e-12)
            position = _as_tensor(tensors["racket_position_w"])[start, lane].float()
            quaternion = _as_tensor(tensors["racket_quaternion_w"])[start, lane].float()
            contact_point = point[start, lane]
            arm = contact_point - position
            racket_lin = _as_tensor(tensors["racket_linear_velocity_w"])[start, lane].float()
            racket_ang = _as_tensor(tensors["racket_angular_velocity_w"])[start, lane].float()
            surface = racket_lin + torch.linalg.cross(racket_ang, arm)
            ball_vel = ball[start, lane, 7:10]
            relative = ball_vel - surface
            vn = float((relative * normal).sum())
            normal_impulse = float((impulse * normal).sum())
            tangent_impulse = impulse - normal_impulse * normal
            mass = normal_impulse / max(-(1.72) * vn, 1e-9) if vn < 0 else float("nan")
            gap = float((ball[start, lane, :3] - contact_point).norm() - .025)
            local = _quat_rotate_inverse(quaternion[None], (contact_point - position)[None])[0]
            region, grid = _contact_region(local)
            row = dict(
                source=str(path), source_sha256=source_hash, lane=lane,
                toi_frame=int(start), toi_time_s=float(start * dt),
                contact_object="racket", contact_point_w=contact_point.tolist(),
                normal_w=normal.tolist(), gap_m=gap,
                relative_velocity_w=relative.tolist(), impulse_ns=impulse.tolist(),
                normal_impulse_ns=normal_impulse, tangential_impulse_ns=tangent_impulse.tolist(),
                penetration_m=max(0., -gap), contact_order=order,
                effective_mass_kg=mass, racket_part="unknown",
                contact_region=region, contact_grid=grid, toi_phase=int((start % 160) / 40),
                ball_state_before=_frame_state(tensors, before, lane),
                ball_state_after=_frame_state(tensors, after, lane),
                racket_state_before=_frame_state(tensors, before, lane),
                racket_state_after=_frame_state(tensors, after, lane),
                toi_pre_substep=_frame_state(tensors, before, lane),
                toi_post_substep=_frame_state(tensors, after, lane),
                outgoing_direction_w=(ball[after, lane, 7:10] / ball[after, lane, 7:10].norm().clamp_min(1e-12)).tolist(),
                outgoing_speed_mps=float(ball[after, lane, 7:10].norm()),
                quality_flags=["native_substep_force_integrated", "racket_part_unavailable"],
                normal_source="racket_force_direction_proxy",
                impulse_source="integrated_contact_force_matrix",
                effective_mass_source="inferred_from_normal_impulse_and_relative_speed",
                interference_objects=[name for name, active in (
                    ("ground", bool(ground_touching[start:end + 1, lane].any())),
                    ("net", bool(net_touching[start:end + 1, lane].any())),
                ) if active],
                contact_count_substeps=int(end - start + 1),
                physics_dt_s=dt,
            )
            row["bucket"] = _bucket(row)
            records.append(row)
        if not mask.any():
            for center in range(0, frames, max(1, near_miss_radius)):
                if touching[max(0, center - near_miss_radius):min(frames, center + near_miss_radius + 1), lane].any():
                    continue
                point_row = point[center, lane]
                ball_pos = ball[center, lane, :3]
                gap = float((ball_pos - point_row).norm() - .025) if torch.isfinite(point_row).all() else float("nan")
                interference = "ground" if float(ground_force[center, lane].norm()) > force_threshold else (
                    "net" if float(net_force[center, lane].norm()) > force_threshold else "none")
                row = dict(source=str(path), source_sha256=source_hash, lane=lane,
                           toi_frame=int(center), toi_time_s=float(center * dt),
                           contact_object="no_contact", contact_point_w=point_row.tolist(),
                           normal_w=[float("nan")] * 3, gap_m=gap,
                           relative_velocity_w=[float("nan")] * 3, impulse_ns=[0.] * 3,
                           normal_impulse_ns=0., tangential_impulse_ns=[0.] * 3,
                           penetration_m=0., contact_order=0, effective_mass_kg=float("nan"),
                           racket_part="unknown", contact_region="near_miss",
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
                           contact_count_substeps=0, physics_dt_s=dt)
                row["bucket"] = "near_miss:" + interference + ":" + str(row["toi_phase"])
                records.append(row)
    return records


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
    fn = lambda path: _trace_candidates(path, force_threshold, near_miss_radius)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(fn, files))
    else:
        groups = [fn(path) for path in files]
    rows = [row for group in groups for row in group]
    positives = [row for row in rows if row["contact_object"] == "racket"]
    negatives_rows = [row for row in rows if row["contact_object"] == "no_contact"]
    selected = _select(positives, contacts, seed) + _select(negatives_rows, negatives, seed + 1)
    selected.sort(key=lambda row: (row["contact_object"], row["source"], row["lane"], row["toi_frame"]))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(schema=SCHEMA, rows=selected,
                   field_schema=list(PILOT_FIELDS), source_files=[str(p.resolve()) for p in files],
                   source_count=len(files), substep_truth=True)
    torch.save(payload, output / "pilot.pt")
    buckets: dict[str, int] = {}
    for row in selected:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
    manifest = dict(schema=SCHEMA, requested_contacts=contacts, requested_negatives=negatives,
                    selected_contacts=sum(r["contact_object"] == "racket" for r in selected),
                    selected_negatives=sum(r["contact_object"] == "no_contact" for r in selected),
                    candidate_contacts=len(positives), candidate_negatives=len(negatives_rows),
                    workers=int(workers), seed=int(seed), force_threshold_n=float(force_threshold),
                    bucket_counts=buckets, substep_truth=True,
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
                    pilot_status="READY" if len(positives) >= contacts and len(negatives_rows) >= negatives else "INSUFFICIENT_CANDIDATES")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
