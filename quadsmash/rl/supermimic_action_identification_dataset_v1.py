"""Pack immutable paired PhysX probes for action-to-racket identification.

The resulting dataset is deliberately separate from policy-distribution Replay.
It contains executable plus/minus action windows and native next states, but it
is never used as a policy observation or as a fabricated counterfactual.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .supermimic_native_pair_capture_v1 import BRANCHES, tree_digest, validate_spec
from .supermimic_physical_dataset_v1 import encode_physical


SCHEMA = "supermimic-action-identification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_branch(path: Path, branch: str) -> dict[str, Any]:
    probe = path / branch / "native_probe.pt"
    if not probe.is_file():
        raise ValueError(f"missing native probe: {probe}")
    value = torch.load(probe, map_location="cpu", weights_only=True)
    if value.get("schema") != "native-paired-prefix-run-v1":
        raise ValueError(f"wrong native probe schema: {probe}")
    if value.get("branch") != branch:
        raise ValueError(f"native probe branch mismatch: {probe}")
    if value.get("actor_updates", 0) != 0 or value.get("world_updates", 0) != 0:
        raise ValueError("identification probes must not update actor or world model")
    return value


def _same_contract(runs: dict[str, dict[str, Any]]) -> None:
    reference = runs[BRANCHES[0]]
    for branch, run in runs.items():
        for key in (
            "spec", "seed", "task_contract", "policy_execution", "physics",
            "prefix_sha256", "executor_before_sha256", "execution_code_sha256",
            "restore_mode",
        ):
            if run.get(key) != reference.get(key):
                raise ValueError(f"probe contract mismatch for {branch}: {key}")
        if tree_digest(run["raw"]["start"]) != tree_digest(reference["raw"]["start"]):
            raise ValueError(f"probe start mismatch for {branch}")
        if not bool(run["raw"]["command_valid_lanes"].all()):
            raise ValueError(f"probe crosses reset for {branch}")
    if reference.get("restore_mode") != "from-ground-identical-prefix":
        raise ValueError("identification probes require identical ground restoration")


def pack_collection(collection: str | Path, output: str | Path) -> dict[str, Any]:
    collection = Path(collection)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    status_path = collection / "status.json"
    if not status_path.is_file():
        raise ValueError(f"missing collection status: {status_path}")
    status = json.loads(status_path.read_text())
    if status.get("status") != "COLLECTION_FINISHED":
        raise ValueError("collection is not complete")
    spec_dirs = sorted(path for path in collection.glob("spec_*") if path.is_dir())
    if not spec_dirs:
        raise ValueError("collection has no completed spec directories")

    rows: list[dict[str, Any]] = []
    source_specs = []
    for spec_dir in spec_dirs:
        spec_path = Path(status.get("spec_paths", {}).get(spec_dir.name, ""))
        if not spec_path.is_file():
            startup = json.loads((collection / "startup.json").read_text())
            entries = {Path(item["path"]).stem: item for item in startup.get("specs", [])}
            item = entries.get(spec_dir.name)
            spec_path = Path(item["path"]) if item else Path()
        if not spec_path.is_file():
            raise ValueError(f"cannot resolve source spec for {spec_dir.name}")
        spec = json.loads(spec_path.read_text())
        validate_spec(spec)
        runs = {branch: _load_branch(spec_dir, branch) for branch in BRANCHES}
        _same_contract(runs)
        reference = runs[BRANCHES[0]]
        pairs = {branch: runs[branch]["raw"] for branch in BRANCHES}
        start_physical = reference["raw"]["start"]["before"]["physical"]
        n = int(start_physical["joint_position"].shape[0])
        def step_tensor(branch: str, key: str) -> torch.Tensor:
            sequence = pairs[branch].get("step_sequence", [])
            if not sequence:
                raise ValueError(f"missing step sequence for {branch}")
            return torch.stack([item[key] for item in sequence], dim=0)

        step_targets_minus = step_tensor("minus", "final_joint_targets")
        step_targets_plus = step_tensor("plus", "final_joint_targets")
        step_actions_minus = step_tensor("minus", "final_action")
        step_actions_plus = step_tensor("plus", "final_action")
        for lane in range(n):
            row: dict[str, Any] = {
                "sample_id": f"{spec_dir.name}:lane{lane}",
                "spec_name": spec_dir.name,
                "lane_index": lane,
                "head": spec.get("head"),
                "event_tick": int(spec["tick"]),
                "action_tick": int(spec.get("action_tick", spec["tick"])),
                "action_hold_ticks": int(spec.get("action_hold_ticks", 1)),
                "epsilon_request": float(spec["epsilon_request"]),
                "request_direction": torch.tensor(spec["request_directions"][lane], dtype=torch.float32),
                "state": encode_physical(start_physical)[lane].clone(),
                "previous_action": start_physical["final_action"][lane].clone(),
                "foot_contact": start_physical["foot_contact"][lane].clone(),
                "present": start_physical["ball_present"][lane].clone(),
                "elapsed": start_physical["elapsed_s"][lane].clone(),
                "next_nominal": pairs["nominal_a"]["next_state"][lane].clone(),
                "next_minus": pairs["minus"]["next_state"][lane].clone(),
                "next_plus": pairs["plus"]["next_state"][lane].clone(),
                "target_nominal": pairs["nominal_a"]["final_joint_targets"][lane].clone(),
                "target_minus": pairs["minus"]["final_joint_targets"][lane].clone(),
                "target_plus": pairs["plus"]["final_joint_targets"][lane].clone(),
                "step_targets_minus": step_targets_minus[:, lane].clone(),
                "step_targets_plus": step_targets_plus[:, lane].clone(),
                "step_final_actions_minus": step_actions_minus[:, lane].clone(),
                "step_final_actions_plus": step_actions_plus[:, lane].clone(),
            }
            rows.append(row)
        source_specs.append({"name": spec_dir.name, "path": str(spec_path.resolve()), "sha256": _sha256(spec_path)})

    if not rows:
        raise ValueError("identification collection produced no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": SCHEMA,
        "source_collection": str(collection.resolve()),
        "source_status_sha256": _sha256(status_path),
        "source_specs": source_specs,
        "probe_count": len(rows),
        "independent_spec_count": len(source_specs),
        "rows": rows,
        "scope": "paired native action-to-racket identification; never a policy rollout input",
    }
    torch.save(result, output)
    receipt = {
        "schema": SCHEMA,
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "probe_count": len(rows),
        "independent_spec_count": len(source_specs),
        "action_hold_ticks": sorted({int(row["action_hold_ticks"]) for row in rows}),
        "event_ticks": sorted({int(row["event_tick"]) for row in rows}),
        "admitted_to_policy_training": False,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
