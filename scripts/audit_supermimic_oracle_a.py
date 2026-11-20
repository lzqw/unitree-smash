#!/usr/bin/env python3
"""Compare the analytic racket--ball impulse prior with native PhysX substeps.

This is a read-only B0 diagnostic.  It consumes an existing
``native_probe.pt`` trace and never runs Isaac Sim, changes a policy, or writes
training labels.  The report keeps the measured substep response separate from
the analytic prediction so a failed check can be attributed to timing,
geometry, restitution, friction, or the missing finite effective mass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation


def _float(value: torch.Tensor | float) -> float:
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _vec(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().tolist()]


def _json_value(value):
    if torch.is_tensor(value):
        return _vec(value) if value.ndim else _float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _load_trace(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    trace = payload.get("raw", {}).get("substep_trace")
    if not isinstance(trace, dict) or trace.get("schema") != "native-one-control-tick-substeps-v1":
        raise ValueError(f"{path} does not contain a native substep trace")
    tensors = trace.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError(f"{path} is missing substep tensors")
    required = (
        "physics_frame", "ball_clock", "shuttle_state_w", "racket_position_w",
        "racket_quaternion_w", "racket_linear_velocity_w",
        "racket_angular_velocity_w", "racket_force_w",
        "racket_contact_position_w", "racket_contact_position_valid",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise ValueError(f"{path} missing trace fields: {','.join(missing)}")
    frames = tensors["physics_frame"]
    if frames.ndim != 1 or len(frames) != int(trace.get("count", -1)):
        raise ValueError("Invalid native substep frame count")
    expected = torch.arange(int(frames[0]), int(frames[0]) + len(frames))
    if not torch.equal(frames, expected):
        raise ValueError("Native substep frame sequence has a gap")
    nframes, lanes = tensors["shuttle_state_w"].shape[:2]
    for key, value in tensors.items():
        if key == "physics_frame":
            continue
        if value.shape[:2] != (nframes, lanes):
            raise ValueError(f"Trace tensor {key} has inconsistent clock or lane dimensions")
    if not torch.equal(
        torch.isfinite(tensors["racket_contact_position_w"]).all(-1),
        tensors["racket_contact_position_valid"],
    ):
        raise ValueError("Contact-position validity mask disagrees with recorded points")
    return payload


def _contact_intervals(touch: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open active intervals in substep indices."""
    padded = torch.cat((torch.zeros(1, dtype=torch.bool), touch,
                        torch.zeros(1, dtype=torch.bool)))
    starts = (padded[1:] & ~padded[:-1]).nonzero().flatten().tolist()
    ends = ((~padded[1:]) & padded[:-1]).nonzero().flatten().tolist()
    return list(zip(starts, ends))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(math.floor(fraction * (len(values) - 1)))))
    return values[index]


@torch.no_grad()
def audit(probe_dir: Path) -> dict:
    nominal_path = probe_dir / "nominal_a" / "native_probe.pt"
    nominal = _load_trace(nominal_path)
    tensors = nominal["raw"]["substep_trace"]["tensors"]
    dt = float(nominal["raw"]["substep_trace"]["physics_dt_s"])
    physics_path = ROOT / "configs" / "shuttle_physics_v2.json"
    physics = json.loads(physics_path.read_text())
    threshold = float(physics["contact_force_threshold_n"])
    mass = float(physics["mass_kg"])
    restitution = float(physics["restitution"])
    friction = float(physics["friction"])
    drag = float(physics["quadratic_drag_k_per_m"])
    gravity = torch.tensor([0.0, 0.0, -9.81])
    prior = BallImpulsePriorV1().eval()

    # Nominal A/B must be a byte-for-byte deterministic repeat when available.
    repeat_max = None
    nominal_b_path = probe_dir / "nominal_b" / "native_probe.pt"
    if nominal_b_path.exists():
        other = _load_trace(nominal_b_path)["raw"]["substep_trace"]["tensors"]
        repeat_max = 0.0
        for key in tensors:
            if key == "physics_frame":
                continue
            a, b = tensors[key], other[key]
            if a.dtype == torch.bool:
                same = (a == b)
                error = 0.0 if bool(same.all()) else 1.0
            else:
                error = _float((a - b).abs().nan_to_num().max())
            repeat_max = max(repeat_max, error)

    state = tensors["shuttle_state_w"]
    racket_position = tensors["racket_position_w"]
    racket_rotation = rotation(tensors["racket_quaternion_w"])
    racket_linear = tensors["racket_linear_velocity_w"]
    racket_angular = tensors["racket_angular_velocity_w"]
    force = tensors["racket_force_w"]
    lanes = state.shape[1]
    rows = []
    for lane in range(lanes):
        touch = force[:, lane].norm(dim=-1) > threshold
        intervals = _contact_intervals(touch)
        for order, (start, end) in enumerate(intervals):
            # A transition edge identifies the first active substep.  The
            # interval length is retained to expose persistent/multiple contact.
            i = int(start)
            if i <= 0:
                i = 1
            pre_i = i - 1
            ball_pre = state[pre_i, lane]
            ball_post = state[i, lane]
            v0 = ball_pre[7:10]
            acceleration = gravity - drag * v0.norm() * v0
            incoming = v0 + dt * acceleration
            free_position = ball_pre[:3] + dt * incoming
            center0 = racket_position[pre_i, lane]
            center1 = racket_position[i, lane]
            r0 = racket_rotation[pre_i, lane]
            r1 = racket_rotation[i, lane]
            local0 = r0.transpose(-1, -2) @ (ball_pre[:3] - center0)
            local1 = r1.transpose(-1, -2) @ (free_position - center1)
            gap0 = _float(prior.preview.racket_gap(local0[None])[0])
            gap1 = _float(prior.preview.racket_gap(local1[None])[0])
            denominator = gap0 - gap1
            fraction = 0.5 if abs(denominator) < 1e-12 else gap0 / denominator
            fraction = max(0.0, min(1.0, fraction))
            impact_time = fraction * dt
            delta0 = ball_pre[:3] - center0
            surface0 = racket_linear[pre_i, lane] + torch.linalg.cross(
                racket_angular[pre_i, lane], delta0)
            relative0 = r0.transpose(-1, -2) @ (v0 - surface0)
            endpoint_ball, endpoint_velocity = prior.advance(
                ball_pre[:3][None], v0[None], torch.full((1,), dt))
            endpoint_ball = endpoint_ball[0]
            endpoint_velocity = endpoint_velocity[0]
            delta1 = endpoint_ball - center1
            surface1 = racket_linear[i, lane] + torch.linalg.cross(
                racket_angular[i, lane], delta1)
            relative1 = r1.transpose(-1, -2) @ (endpoint_velocity - surface1)
            contact = prior.contact_state_at_time(
                ball_pre[:3][None], v0[None], r0[None], center0[None],
                racket_linear[pre_i, lane][None], racket_angular[pre_i, lane][None],
                torch.tensor([impact_time]), endpoint_rotation=r1[None],
                endpoint_center=center1[None], endpoint_velocity=racket_linear[i, lane][None],
                endpoint_angular_velocity=racket_angular[i, lane][None],
                local_start=local0[None], local_end=local1[None],
                relative_start=relative0[None], relative_end=relative1[None])
            normal = contact["normal"][0]
            surface_velocity = contact["surface_velocity"][0]
            toi_velocity = contact["velocity"][0]
            normal_speed = _float(((toi_velocity - surface_velocity) * normal).sum())
            normal_dv = -(1.0 + restitution) * min(normal_speed, 0.0) * normal
            tangent_relative = toi_velocity - surface_velocity - normal_speed * normal
            tangent_speed = _float(tangent_relative.norm())
            tangent_dv_norm = min((2.0 / 7.0) * tangent_speed,
                                  friction * _float(normal_dv.norm()))
            tangent_dv = tangent_dv_norm * tangent_relative / max(tangent_speed, 1e-12)
            predicted_dv = normal_dv - tangent_dv
            native_dv = ball_post[7:10] - incoming
            native_impulse = mass * native_dv
            native_normal_impulse = _float((native_impulse * normal).sum())
            native_tangent_impulse = native_impulse - native_normal_impulse * normal
            predicted_impulse = mass * predicted_dv
            effective_mass = None
            if normal_speed < -1e-8 and native_normal_impulse > 0:
                effective_mass = native_normal_impulse / ((1.0 + restitution) * -normal_speed)
            predicted_speed = _float((incoming + predicted_dv).norm())
            native_speed = _float(ball_post[7:10].norm())
            cos = _float(torch.nn.functional.cosine_similarity(
                (incoming + predicted_dv)[None], ball_post[7:10][None]))
            # A gap-based object label is useful for ground/net interference,
            # while the force edge remains the authoritative racket event.
            ground_gap = _float(ball_pre[2] - float(physics["collision_radius_m"]))
            net_gap = _float(prior.preview.net_gap(ball_pre[:3][None])[0])
            object_candidates = {"racket": gap0, "ground": ground_gap, "net": net_gap}
            nearest_object = min(object_candidates, key=object_candidates.get)
            rows.append({
                "lane": lane,
                "substep_index": int(i),
                "contact_order": int(order),
                "contact_interval_end": int(end),
                "contact_substep_count": int(end - start),
                "contact_object": "racket",
                "nearest_gap_object": nearest_object,
                "toi_fraction": fraction,
                "toi_s": impact_time,
                "absolute_substep_time_s": (pre_i * dt) + impact_time,
                "racket_part": int(contact["part"][0]),
                "contact_point_m": _vec((contact["center"][0] + contact["rotation"][0] @ contact["local"][0])),
                "normal": _vec(normal),
                "normal_relative_speed_m_s": normal_speed,
                "incoming_speed_m_s": _float(incoming.norm()),
                "native_outgoing_speed_m_s": native_speed,
                "analytic_outgoing_speed_m_s": predicted_speed,
                "outgoing_direction_cosine": cos,
                "native_dv_m_s": _vec(native_dv),
                "analytic_dv_m_s": _vec(predicted_dv),
                "native_impulse_ns": _vec(native_impulse),
                "analytic_impulse_ns": _vec(predicted_impulse),
                "native_normal_impulse_ns": native_normal_impulse,
                "analytic_normal_impulse_ns": _float((predicted_impulse * normal).sum()),
                "native_tangent_impulse_ns": _vec(native_tangent_impulse),
                "native_tangent_impulse_norm_ns": _float(native_tangent_impulse.norm()),
                "analytic_tangent_impulse_ns": _vec(-mass * tangent_dv),
                "analytic_tangent_impulse_norm_ns": _float((mass * tangent_dv).norm()),
                "effective_mass_kg": effective_mass,
                "analytic_effective_mass_kg": mass,
                "racket_force_norm_n": _float(force[i, lane].norm()),
                "ground_gap_m": ground_gap,
                "net_gap_m": net_gap,
                "contact_position_valid": bool(tensors["racket_contact_position_valid"][i, lane]),
                "native_contact_position_m": _vec(tensors["racket_contact_position_w"][i, lane])
                if bool(tensors["racket_contact_position_valid"][i, lane]) else None,
            })

    def errors(key):
        return [abs(float(row[key])) for row in rows if row.get(key) is not None]

    normal_errors = [
        abs(row["native_normal_impulse_ns"] - row["analytic_normal_impulse_ns"])
        for row in rows
    ]
    tangent_errors = [
        abs(row["native_tangent_impulse_norm_ns"] - row["analytic_tangent_impulse_norm_ns"])
        for row in rows
    ]
    direction = [row["outgoing_direction_cosine"] for row in rows]
    amplitude_ratio = [row["analytic_outgoing_speed_m_s"] / max(row["native_outgoing_speed_m_s"], 1e-8)
                       for row in rows]
    amplitude_error = [abs(value - 1.0) for value in amplitude_ratio]
    normal_relative_error = [abs(row["native_normal_impulse_ns"] - row["analytic_normal_impulse_ns"])
                             / max(abs(row["native_normal_impulse_ns"]), 1e-8) for row in rows]
    tangent_relative_error = [abs(row["native_tangent_impulse_norm_ns"] - row["analytic_tangent_impulse_norm_ns"])
                              / max(abs(row["native_tangent_impulse_norm_ns"]), 1e-8) for row in rows]
    bucket_rows: dict[str, list[dict]] = {}
    for row in rows:
        speed = abs(row["normal_relative_speed_m_s"])
        bucket = f"vn{sum(speed >= edge for edge in (1., 5., 10.))}"
        bucket_rows.setdefault(bucket, []).append(row)
    bucket_summary = {}
    for bucket, values in sorted(bucket_rows.items()):
        bucket_summary[bucket] = {
            "count": len(values),
            "direction_cosine_mean": _mean([item["outgoing_direction_cosine"] for item in values]),
            "normal_impulse_mae_ns": _mean([
                abs(item["native_normal_impulse_ns"] - item["analytic_normal_impulse_ns"])
                for item in values
            ]),
            "amplitude_ratio_mean": _mean([
                item["analytic_outgoing_speed_m_s"] / max(item["native_outgoing_speed_m_s"], 1e-8)
                for item in values
            ]),
        }
    object_counts = {name: sum(row["nearest_gap_object"] == name for row in rows)
                     for name in ("racket", "ground", "net")}
    summary = {
        "contact_event_count": len(rows),
        "lane_count": lanes,
        "lane_with_contact_count": len({row["lane"] for row in rows}),
        "multi_contact_event_count": sum(row["contact_substep_count"] > 1 for row in rows),
        "max_contact_substep_count": max((row["contact_substep_count"] for row in rows), default=0),
        "nearest_gap_object_counts": object_counts,
        "normal_impulse_mae_ns": _mean(normal_errors),
        "tangent_impulse_mae_ns": _mean(tangent_errors),
        "normal_impulse_relative_error_mean": _mean(normal_relative_error),
        "tangent_impulse_relative_error_mean": _mean(tangent_relative_error),
        "direction_cosine_mean": _mean(direction),
        "direction_cosine_p10": _percentile(direction, .10),
        "amplitude_ratio_mean": _mean(amplitude_ratio),
        "amplitude_ratio_p10": _percentile(amplitude_ratio, .10),
        "amplitude_ratio_p90": _percentile(amplitude_ratio, .90),
        "amplitude_abs_error_mean": _mean(amplitude_error),
        "bucket_summary": bucket_summary,
        "native_effective_mass_mean_kg": _mean([
            row["effective_mass_kg"] for row in rows if row["effective_mass_kg"] is not None
        ]),
        "native_effective_mass_min_kg": min((row["effective_mass_kg"] for row in rows
                                              if row["effective_mass_kg"] is not None), default=None),
        "native_effective_mass_max_kg": max((row["effective_mass_kg"] for row in rows
                                              if row["effective_mass_kg"] is not None), default=None),
        "toi_fraction_mean": _mean([row["toi_fraction"] for row in rows]),
        "repeat_max_abs_error": repeat_max,
        "analytic_model": {
            "restitution": restitution,
            "friction": friction,
            "mass_kg": mass,
            "drag_k_per_m": drag,
            "native_dt_s": dt,
            "limitations": [
                "finite racket effective mass is inferred from native impulse",
                "PhysX manifold force is sampled at substep resolution",
                "racket force edge is authoritative for racket contact",
                "ground/net labels are gap-based interference diagnostics",
            ],
        },
    }
    return {
        "schema": "supermimic-oracle-a-v1",
        "scope": "B0 native PhysX substep versus analytic TOI/impulse diagnostic",
        "probe_dir": str(probe_dir.resolve()),
        "source_branch": nominal.get("branch"),
        "source_schema": nominal.get("schema"),
        "summary": summary,
        "events": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = _json_value(audit(args.probe_dir.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
