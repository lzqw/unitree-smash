"""Strict adapter for the measured G07 shared-target residual policy.

The historical 27/128 policy is not a ``full-racket-residual-ppo-v4``
checkpoint. Its public base is a measured G07 controller and its residual
actor is exported separately. This adapter keeps that execution contract
explicit instead of relabeling the checkpoint or partially loading it into
the analytic controller.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path

import torch

from .full_racket_residual_v4 import ResidualActionMixerV4
from .persistent_recovery_assistance_v1 import PersistentRecoveryAssistanceV1


SCHEMA = "g07-shared-target-residual-ppo-v1"
BASE_SCHEMA = "g07-public-shared-target-bias-v5"
DEFAULT_DIAGNOSTIC_DIR = (
    Path("/home/autodrive/LZQW/unitree_smash")
    / "reports/G07_full_racket_v5/current_intercept_diagnosis_20261027"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load G07 execution source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic_dir(execution_root: Path | None) -> Path:
    if execution_root is not None:
        root = Path(execution_root).expanduser().resolve()
        return root / "reports/G07_full_racket_v5/current_intercept_diagnosis_20261027"
    value = os.environ.get("QUADSMASH_G07_EXECUTION_ROOT")
    if value:
        return Path(value).expanduser().resolve() / "reports/G07_full_racket_v5/current_intercept_diagnosis_20261027"
    return DEFAULT_DIAGNOSTIC_DIR


def _validate_source(checkpoint: dict, diag: Path) -> dict:
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError(f"Require {SCHEMA}, got {checkpoint.get('schema')!r}")
    if not checkpoint.get("native_execution"):
        raise ValueError("G07 checkpoint is not marked as native executed")
    base = checkpoint.get("base_controller")
    if not isinstance(base, dict) or base.get("schema") != BASE_SCHEMA:
        raise ValueError("G07 checkpoint lacks the measured shared-target base")
    if checkpoint.get("asset_sha256") != base.get("asset_sha256"):
        raise ValueError("G07 residual/base asset hashes differ")
    if checkpoint.get("residual_base_mode") != "shared_target_bias":
        raise ValueError("Unexpected G07 residual base mode")
    bias = base.get("shared_target_bias", {}).get("bias_world_m")
    if not isinstance(bias, list) or len(bias) != 3:
        raise ValueError("G07 base has no declared shared target bias")
    factory = diag / "native_factory.py"
    core = diag / "shared_target_controller.py"
    cpu_check = diag / "CPU_SHARED_TARGET_CHECK.json"
    for path in (factory, core, cpu_check):
        if not path.is_file():
            raise FileNotFoundError(f"Missing measured G07 execution source: {path}")
    expected_factory = base.get("factory_sha256")
    expected_core = base.get("target_core_sha256")
    if expected_factory and _sha256(factory) != expected_factory:
        raise ValueError("G07 native_factory.py hash does not match checkpoint")
    if expected_core and _sha256(core) != expected_core:
        raise ValueError("G07 shared_target_controller.py hash does not match checkpoint")
    blob = checkpoint.get("residual_actor_jit")
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError("G07 checkpoint has no residual_actor_jit")
    return base


class G07SharedTargetResidualControllerV1:
    """Run the historical G07 controller through the current env interface."""

    actor_kind = "native_residual_jit_794_256_128_64"
    schema = SCHEMA
    uses_external_execution_contract = True

    def __init__(self, checkpoint: dict, count: int, device: torch.device,
                 execution_root: Path | None = None):
        diag = _diagnostic_dir(execution_root)
        base = _validate_source(checkpoint, diag)
        factory = _load_module(diag / "native_factory.py", "quadsmash_g07_native_factory")
        self.source_root = diag
        self.base = factory.make_controller(
            base["source_public_policy"], count, device,
            bias_world_m=base["shared_target_bias"]["bias_world_m"],
        )
        recovery = checkpoint.get("residual_training_metadata", {}).get("recovery_assistance")
        if not isinstance(recovery, dict):
            raise ValueError("G07 checkpoint lacks measured recovery assistance metadata")
        self.base = PersistentRecoveryAssistanceV1(self.base, recovery)
        self.actor = torch.jit.load(
            io.BytesIO(checkpoint["residual_actor_jit"]), map_location=device
        ).eval()
        scales = checkpoint.get("residual_joint_scale_rad")
        if not isinstance(scales, list) or len(scales) != 18:
            raise ValueError("G07 residual scale must contain exactly 18 joints")
        arm_rate = float(checkpoint.get("residual_arm_target_rate_rad_s", 5.0))
        max_scale = float(checkpoint.get("residual_scale_limit_rad", max(scales)))
        self.mixer = ResidualActionMixerV4(count, device, scales, arm_rate, max_scale)
        if checkpoint.get("residual_training_metadata", {}).get("memory_conditioned", False):
            raise ValueError("Memory-conditioned G07 actor needs an explicit observation adapter")

    @property
    def reference_controller(self):
        return self.base.reference_controller

    @property
    def logs(self):
        return self.base.logs

    @logs.setter
    def logs(self, value):
        self.base.logs = value

    def reset(self, ids=None):
        self.base.reset(ids)
        self.mixer.reset(ids)

    @torch.inference_mode()
    def apply_residual(self, observation, residual, default, limits):
        if observation["policy"].shape[-1] != 794:
            raise ValueError("G07 actor requires the native 794D public observation")
        base = self.base.action(observation, default, limits)
        return self.mixer.action(base, residual, observation["policy"], default, limits)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.apply_residual(observation, self.actor(observation["policy"]), default, limits)


def make_g07_residual_controller(checkpoint, count, device, execution_root=None):
    return G07SharedTargetResidualControllerV1(
        checkpoint, count, device,
        None if execution_root is None else Path(execution_root),
    )

