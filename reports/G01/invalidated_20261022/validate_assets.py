#!/usr/bin/env python3
"""Asset and actuator audit for G01.

The audit is conservative: a Go2 USD plus an unconfirmed placeholder arm is
not reported as a coupled Go2-PiPER system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="reports/G01")
    parser.add_argument("--piper", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    go2 = Path("/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd")
    piper = args.piper or next(iter((root / "assets").rglob("*piper*.usd")), None)
    arm_manifest = root / "assets/arms/placeholder_6dof.json"
    racket_manifest = root / "assets/rackets/standard_racket_v1.json"
    go2_ok = go2.is_file()
    piper_ok = piper is not None and piper.is_file()
    arm_data = json.loads(arm_manifest.read_text(encoding="utf-8"))
    racket_data = json.loads(racket_manifest.read_text(encoding="utf-8"))
    names = [f"leg_joint_{i}" for i in range(12)] + list(arm_data["joint_names"])
    mapping_ok = go2_ok and piper_ok and len(names) == 18 and len(set(names)) == 18
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "go2": {"path": str(go2), "sha256": digest(go2), "exists": go2_ok, "source": "local authorized path; joint names require USD/Isaac inspection"},
        "piper": {"path": str(piper) if piper else None, "sha256": digest(piper) if piper else None, "exists": piper_ok, "status": "verified" if piper_ok else "missing"},
        "arm": {"manifest": str(arm_manifest), "sha256": digest(arm_manifest), "status": arm_data.get("status"), "parameters_are_not_real_robot_data": arm_data.get("parameters_are_not_real_robot_data", True), "dof": arm_data.get("dof")},
        "racket": {"manifest": str(racket_manifest), "sha256": digest(racket_manifest), "parameters": racket_data},
        "joint_contract": {"expected_active_dof": 18, "joint_names": names, "mapping_verified": bool(mapping_ok)},
        "assumptions": ["PiPER installation plate and calibrated inertial properties are not verified in this workspace.", "The placeholder arm is not evidence of a standard PiPER asset.", "No standing dynamics run is reported without the coupled articulated scene."],
    }
    (output / "assets_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = {"joint_mapping_ok": bool(mapping_ok), "coupled_dynamics_ok": False, "asset_manifest_ok": bool(go2_ok and piper_ok and not arm_data.get("parameters_are_not_real_robot_data", True)), "stand_success_rate": 0.0, "sample_count": 0, "stand_episode_seconds": 0.0, "asset_backend": "not_run_without_verified_coupled_scene"}
    (output / "asset_tests.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
