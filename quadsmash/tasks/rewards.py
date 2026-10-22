"""Stage-aware rewards for event-conditioned whole-body control."""

from __future__ import annotations

import numpy as np

from .goals import CurriculumConfig, CurriculumStage


def _as_batch(value: np.ndarray | list[float] | float, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def event_tracking_reward(time_error: np.ndarray, position_error: np.ndarray, normal_alignment: np.ndarray, velocity_error: np.ndarray, *, time_tolerance: float = 0.12, position_tolerance: float = 0.18, normal_tolerance: float = 0.65, velocity_tolerance: float = 3.0) -> np.ndarray:
    """Reward the racket at the requested time window, not throughout the episode."""
    if min(time_tolerance, position_tolerance, normal_tolerance, velocity_tolerance) <= 0:
        raise ValueError("tracking tolerances must be positive")
    t = _as_batch(time_error, "time_error")
    p = _as_batch(position_error, "position_error")
    n = _as_batch(normal_alignment, "normal_alignment")
    v = _as_batch(velocity_error, "velocity_error")
    return (np.exp(-0.5 * (t / time_tolerance) ** 2) * np.exp(-p / position_tolerance**2 - np.maximum(0.0, 1.0 - n) / normal_tolerance - v / velocity_tolerance**2)).astype(np.float32)


def outcome_success(airborne_contact: np.ndarray, valid_return: np.ndarray, attack_requirement: np.ndarray, stable_recovery: np.ndarray, config: CurriculumConfig | None = None, event_target_reached: np.ndarray | None = None) -> np.ndarray:
    """Return the terminal success indicator with stage-specific requirements."""
    cfg = config or CurriculumConfig.for_stage("smash_and_recover")
    airborne_contact = np.asarray(airborne_contact, dtype=bool)
    # Early jump/recovery stages have no hit requirement; their event is the
    # stable recovery itself.  Later stages require an actual airborne contact
    # (or the virtual target event in the virtual-event stage).
    if cfg.stage is CurriculumStage.VIRTUAL_EVENT:
        if event_target_reached is None:
            raise ValueError("virtual-event success requires event_target_reached")
        terms = [np.asarray(event_target_reached, dtype=bool)]
    else:
        terms = [airborne_contact if cfg.require_real_contact else np.ones_like(airborne_contact)]
    if cfg.require_real_contact:
        terms.append(airborne_contact)
    if cfg.require_valid_return:
        terms.append(np.asarray(valid_return, dtype=bool))
    if cfg.require_attack:
        terms.append(np.asarray(attack_requirement, dtype=bool))
    if cfg.require_recovery:
        terms.append(np.asarray(stable_recovery, dtype=bool))
    result = terms[0].copy()
    for term in terms[1:]:
        result &= term
    return result


def compute_reward(contact: np.ndarray, attack_quality: np.ndarray, target_error: np.ndarray, energy: np.ndarray, risk: np.ndarray, *, tracking: np.ndarray | None = None, recovery: np.ndarray | None = None, success: np.ndarray | None = None, contact_weight: float = 2.0, attack_weight: float = 1.0, landing_weight: float = 0.5, energy_weight: float = 0.002, risk_weight: float = 0.2, recovery_weight: float = 0.35, success_weight: float = 8.0) -> np.ndarray:
    """Evaluate training, event, outcome, and safety terms element-wise.

    Contact is an event pulse and does not terminate the episode by itself;
    ``success`` is reserved for the joint hit-and-recovery outcome.
    """
    contact = _as_batch(contact, "contact")
    attack_quality = _as_batch(attack_quality, "attack_quality")
    target_error = _as_batch(target_error, "target_error")
    energy = _as_batch(energy, "energy")
    risk = _as_batch(risk, "risk")
    result = contact_weight * contact + attack_weight * attack_quality + landing_weight * np.exp(-target_error)
    if tracking is not None:
        result += _as_batch(tracking, "tracking")
    if recovery is not None:
        result += recovery_weight * _as_batch(recovery, "recovery")
    if success is not None:
        result += success_weight * _as_batch(success, "success")
    result -= energy_weight * energy + risk_weight * risk
    return result.astype(np.float32)

