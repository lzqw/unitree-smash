"""Episode-safe event windows for contact-model supervision.

The packed native replay contains control-boundary transitions, while PhysX
substep traces are optional.  This module keeps that distinction explicit. A
record built from packed replay is useful for event-aligned WM fitting, but it
never fabricates substep impulse labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


SCHEMA = "supermimic-contact-event-v1"
EVENT_COLUMNS = {
    "qualified_contact": 1,
    "crossed_net": 2,
    "landed_target": 3,
    "recovery": 4,
    "body_ground": 5,
    "frame_hit": 6,
    "fault": 7,
}


def recommended_window(*, control_dt_s: float = 0.01) -> dict:
    """Return the event window used by the first packed-replay build.

    Four ticks cover the 40 ms pre-impact approach.  Sixty-four ticks cover
    the complete 640 ms imagination horizon and retain post-impact flight and
    immediate recovery context.  The values remain explicit in the manifest
    so later datasets can be compared without relying on defaults.
    """
    pre_ticks = 4
    post_ticks = 64
    return dict(
        pre_control_ticks=pre_ticks,
        post_control_ticks=post_ticks,
        control_dt_s=float(control_dt_s),
        pre_window_s=pre_ticks * float(control_dt_s),
        post_window_s=post_ticks * float(control_dt_s),
        rationale="pre-impact approach plus full 64-step post-impact WM horizon",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_safe_indices(episode: torch.Tensor, lane: int, center: int, pre: int, post: int):
    ticks = torch.arange(center - pre, center + post + 1, dtype=torch.long)
    valid = (ticks >= 0) & (ticks < episode.shape[0])
    valid_ticks = ticks.clamp(0, episode.shape[0] - 1)
    valid &= episode[valid_ticks, lane] == episode[center, lane]
    return valid_ticks, valid


def _event_edges(event_state: torch.Tensor, column: int) -> list[tuple[int, int]]:
    edge = event_state[1:, :, column] > event_state[:-1, :, column]
    # The event is caused by transition t -> t+1.  Keep t as the center so
    # state[t], executed_action[t] and next_state[t] describe the same control
    # interval that created the latch.
    return [(int(t), int(lane)) for t, lane in edge.nonzero(as_tuple=False).tolist()]


def _window_tensor(tensor: torch.Tensor, ticks: torch.Tensor, lane: int) -> torch.Tensor:
    return tensor[ticks, lane].clone()


def _build_record(tensors: dict, *, tick: int, lane: int, kind: str, pre: int, post: int, split: str):
    episode = tensors["episode"]
    ticks, valid = _episode_safe_indices(episode, lane, tick, pre, post)
    record = {
        "event_id": f"{kind}:ep{int(episode[tick, lane])}:lane{lane}:tick{tick}",
        "event_kind": kind,
        "lane": lane,
        "episode_id": int(episode[tick, lane]),
        "contact_tick": tick,
        "window_ticks": ticks,
        "window_valid": valid,
        "split": split,
        "substep_trace_available": False,
        "quality_flags": ["control_boundary_only", "no_physx_substep_impulse_label"],
    }
    for key in (
        "state", "next_state", "targets", "final_action", "previous_action",
        "requested_action", "reward", "terminated", "timeout", "event_state",
        "next_event_state", "present", "elapsed",
    ):
        if key in tensors:
            record[key] = _window_tensor(tensors[key], ticks, lane)
    # Ball occupies the last six dimensions in the 55D physical state.
    if "state" in record and "next_state" in record:
        delta = record["next_state"][:, 52:55] - record["state"][:, 52:55]
        at_center = int(pre)
        record["event_ball_velocity_delta"] = delta[at_center].clone()
        record["event_ball_velocity_delta_norm"] = float(delta[at_center].norm())
    return record


def _split(event_id: str) -> str:
    value = int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "holdout"


def _load_lane_split(path: str | Path | None, lane_count: int):
    if path is None:
        return None
    path = Path(path).resolve()
    split = json.loads(path.read_text())
    train = sorted(set(int(x) for x in split.get("train_lanes", [])))
    validation = sorted(set(int(x) for x in split.get("validation_lanes", [])))
    holdout = sorted(set(int(x) for x in split.get("holdout_lanes", [])))
    groups = {"train": train, "validation": validation, "holdout": holdout}
    assigned = [lane for values in groups.values() for lane in values]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Lane split groups overlap")
    if any(lane < 0 or lane >= lane_count for lane in assigned):
        raise ValueError("Lane split contains an out-of-range lane")
    if not train or not validation or not holdout:
        raise ValueError("Explicit lane split requires train, validation and holdout lanes")
    if set(assigned) != set(range(lane_count)):
        raise ValueError("Explicit lane split must cover every original lane exactly once")
    return groups, path


def build_from_packed(packed_path: str | Path, output_dir: str | Path, *, negative_ratio: int = 2,
                      max_negative: int | None = None,
                      lane_split_manifest: str | Path | None = None) -> dict:
    """Build a compact event dataset from a packed native replay."""
    packed_path = Path(packed_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    package = torch.load(packed_path, map_location="cpu", weights_only=True)
    tensors = package["tensors"]
    lane_split = _load_lane_split(lane_split_manifest, int(tensors["episode"].shape[1]))
    lane_to_split = None
    if lane_split:
        lane_to_split = {
            lane: name
            for name, lanes in lane_split[0].items()
            for lane in lanes
        }
    control_dt_s = float(package.get("source_manifest", {}).get("control_dt_s", 0.01))
    window = recommended_window(control_dt_s=control_dt_s)
    pre, post = window["pre_control_ticks"], window["post_control_ticks"]
    events: list[dict] = []
    for kind, column in EVENT_COLUMNS.items():
        for tick, lane in _event_edges(tensors["event_state"], column):
            event_id = f"{kind}:ep{int(tensors['episode'][tick, lane])}:lane{lane}:tick{tick}"
            split = lane_to_split[lane] if lane_to_split else _split(event_id)
            events.append(_build_record(tensors, tick=tick, lane=lane, kind=kind,
                                        pre=pre, post=post, split=split))
    # Use event-free windows as explicit negatives.  They are sampling labels,
    # not a claim that no PhysX substep contact happened within the tick.
    positive_keys = {(r["episode_id"], r["lane"], r["contact_tick"]) for r in events}
    negative_candidates = []
    state = tensors["event_state"]
    next_state = tensors["next_event_state"]
    event_delta = (next_state > state).any(-1)
    for tick in range(state.shape[0]):
        for lane in range(state.shape[1]):
            key = (int(tensors["episode"][tick, lane]), lane, tick)
            if key not in positive_keys and not bool(event_delta[tick, lane]):
                negative_candidates.append((tick, lane))
    target_negatives = min(len(negative_candidates), len(events) * max(0, int(negative_ratio)))
    if max_negative is not None:
        target_negatives = min(target_negatives, int(max_negative))
    generator = torch.Generator().manual_seed(882072)
    order = torch.randperm(len(negative_candidates), generator=generator)[:target_negatives].tolist()
    for index in order:
        tick, lane = negative_candidates[index]
        event_id = f"no_contact:ep{int(tensors['episode'][tick, lane])}:lane{lane}:tick{tick}"
        split = lane_to_split[lane] if lane_to_split else _split(event_id)
        events.append(_build_record(tensors, tick=tick, lane=lane, kind="no_contact",
                                    pre=pre, post=post, split=split))
    payload = {
        "schema": SCHEMA,
        "source_packed": str(packed_path),
        "source_sha256": _sha256(packed_path),
        "window": window,
        "event_columns": EVENT_COLUMNS,
        "events": events,
        "substep_truth": False,
    }
    torch.save(payload, output_dir / "events.pt")
    counts = {}
    split_counts = {}
    for record in events:
        counts[record["event_kind"]] = counts.get(record["event_kind"], 0) + 1
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    manifest = {
        "schema": SCHEMA,
        "source_packed": str(packed_path),
        "source_sha256": payload["source_sha256"],
        "window": window,
        "substep_truth": False,
        "event_count": len(events),
        "event_counts": counts,
        "split_counts": split_counts,
        "negative_sampling": {
            "requested_ratio": int(negative_ratio),
            "selected": target_negatives,
            "semantics": "control-boundary negative, not PhysX substep-certified no-contact",
        },
    }
    if lane_split:
        groups, split_path = lane_split
        manifest["lane_split"] = groups
        manifest["lane_split_manifest"] = str(split_path)
        manifest["lane_split_sha256"] = _sha256(split_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

