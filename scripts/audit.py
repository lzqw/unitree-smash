#!/usr/bin/env python3
"""Record the local QuadSmash/Isaac environment without installing anything."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/G00/environment.lock.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    isaac_root = Path("/home/autodrive/Isaac/IsaacLab")
    go2 = Path("/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd")
    piper = root / "assets/vendor/piper_isaac_sim/USD/piper_v1.usd"
    coupled_usd = root / "assets/quadsmash_go2_piper.usd"
    racket_manifest = root / "assets/rackets/standard_racket_v1.json"
    assumptions = root / "assets/assumptions.md"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"])
    modules = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "isaaclab", "isaacsim", "rsl_rl", "gymnasium", "numpy", "jax")}
    versions = {}
    for distribution in ("torch", "isaacsim", "isaaclab", "rsl-rl-lib", "gymnasium", "numpy"):
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    isaac_version = None
    version_file = isaac_root / "VERSION"
    if version_file.is_file():
        isaac_version = version_file.read_text(encoding="utf-8").strip()
    allocation_ok = bool(cuda_visible and cuda_visible not in ("", "NoDevFiles", "-1"))
    torch_cuda_ok = False
    torch_cuda_version = None
    torch_device = None
    if modules["torch"]:
        try:
            import torch

            torch_cuda_ok = bool(torch.cuda.is_available())
            torch_cuda_version = torch.version.cuda
            torch_device = torch.cuda.get_device_name(0) if torch_cuda_ok else None
        except Exception:
            torch_cuda_ok = False
    dependency_lock_ok = bool(
        isaac_root.is_dir()
        and isaac_version == "2.3.0"
        and modules["torch"]
        and modules["isaaclab"]
        and modules["isaacsim"]
        and modules["rsl_rl"]
        and versions.get("isaacsim", "").startswith("5.1.")
        and torch_cuda_ok
    )
    native_smoke = root / "reports/G00/native_smoke/result.json"
    native_eval = root / "reports/G00/native_eval/result.json"
    gpu_smoke_ok = False
    checkpoint_reload_ok = False
    if native_smoke.is_file():
        try:
            smoke = json.loads(native_smoke.read_text(encoding="utf-8"))
            gpu_smoke_ok = bool(
                smoke.get("backend") == "Isaac Lab / PhysX"
                and smoke.get("all_tensors_finite") is True
                and smoke.get("observation_shape") == [smoke.get("num_envs"), 45]
            )
        except (OSError, ValueError):
            gpu_smoke_ok = False
    if native_eval.is_file():
        try:
            evaluation = json.loads(native_eval.read_text(encoding="utf-8"))
            checkpoint_reload_ok = bool(
                evaluation.get("backend") == "Isaac Lab / PhysX"
                and evaluation.get("policy_parameters_finite") is True
                and isinstance(evaluation.get("checkpoint_sha256"), str)
                and bool(evaluation.get("checkpoint_sha256"))
            )
        except (OSError, ValueError):
            checkpoint_reload_ok = False
    racket_manifest_ok = False
    if racket_manifest.is_file() and assumptions.is_file():
        try:
            racket_cfg = json.loads(racket_manifest.read_text(encoding="utf-8"))
            racket_manifest_ok = bool(
                racket_cfg.get("asset_id") == "standard_racket_v1"
                and float(racket_cfg.get("mass_kg", 0.0)) > 0.0
                and racket_cfg.get("status")
            )
        except (OSError, ValueError, TypeError):
            racket_manifest_ok = False
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": {"system": platform.platform(), "machine": platform.machine()},
        "gpu": {"query": gpu, "cuda_visible_devices": cuda_visible, "allocation_declared": allocation_ok},
        "dependencies": {"modules": modules, "versions": versions, "torch_cuda_available": torch_cuda_ok, "torch_cuda_version": torch_cuda_version, "torch_device": torch_device},
        "isaaclab": {"path": str(isaac_root), "version": isaac_version, "source_version_sha256": sha256(version_file)},
        "assets": {
            "go2_usd": {"path": str(go2), "exists": go2.is_file(), "sha256": sha256(go2)},
            "piper_usd": {"path": str(piper), "exists": piper.is_file(), "sha256": sha256(piper)},
            "coupled_usd": {"path": str(coupled_usd), "exists": coupled_usd.is_file(), "sha256": sha256(coupled_usd)},
            "racket_manifest": {"path": str(racket_manifest), "exists": racket_manifest.is_file(), "sha256": sha256(racket_manifest)},
            "assumptions": {"path": str(assumptions), "exists": assumptions.is_file(), "sha256": sha256(assumptions)},
        },
        "checks": {"gpu_smoke_ok": gpu_smoke_ok, "dependency_lock_ok": dependency_lock_ok, "checkpoint_reload_ok": checkpoint_reload_ok, "racket_manifest_ok": racket_manifest_ok},
        "interpretation": "Checks are backed by the project-local Isaac Lab/PhysX smoke and checkpoint reload artifacts when those artifacts exist.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
