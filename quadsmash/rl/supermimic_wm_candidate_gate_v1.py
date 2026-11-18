"""Conservative promotion checks for recorded-command WM candidates.

The check compares two frozen world-model rollouts on the same held-out
windows.  It is intentionally separate from native contact validation.  A
candidate that improves training loss while degrading contact-edge prediction
must remain a diagnostic artifact and cannot silently become the WM used for
actor training.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class MetricBound:
    """A lower-is-better metric and its permitted candidate degradation."""

    name: str
    absolute_tolerance: float = 0.02
    relative_tolerance: float = 0.05


DEFAULT_PROTECTED_METRICS = (
    MetricBound("event_aligned/0/1/ball_velocity_rmse"),
    MetricBound("event_aligned/0/1/ball_forward_sign_error", .01, .05),
    # The event edge is a one-step timing contract.  A contact-edge gain is
    # not admissible when it introduces any measurable pre/post impulse, so
    # adjacent offsets use a strict non-degradation bound.  The absolute
    # contact and longer-horizon metrics retain their calibrated tolerances.
    MetricBound("event_aligned/-1/1/ball_velocity_rmse", 0., 0.),
    MetricBound("event_aligned/1/1/ball_velocity_rmse", 0., 0.),
    # Protect the pre-impact autoregressive neighborhood as well. A model
    # that improves the contact-edge frame by moving the impulse earlier is
    # unsafe for BPTT even when its one-step pre-impact error improves.
    MetricBound("event_aligned/-1/4/ball_velocity_rmse", 0., 0.),
    MetricBound("event_aligned/0/4/ball_velocity_rmse", .03, .10),
)


def _lookup(result: dict, path: str):
    """Read a metric from a validation result using slash-separated keys."""
    value = result
    for part in path.split('/'):
        # Validation reports keep metrics under an ``errors`` node, while
        # small unit fixtures may use the compact path directly.  Supporting
        # both forms keeps the gate tied to the report schema without making
        # missing data look like zero.
        if part not in value and isinstance(value, dict) and isinstance(value.get('errors'), dict):
            value = value['errors']
        if part not in value:
            return None
        value = value[part]
    return value


def compare_candidate_to_anchor(
    anchor: dict,
    candidate: dict,
    *,
    metrics=DEFAULT_PROTECTED_METRICS,
    require_contact_improvement=True,
):
    """Return an auditable decision for one frozen candidate.

    Both inputs are the per-candidate records under ``results`` in the JSON
    emitted by ``validate_supermimic_physical_wm.py``.  No metric is silently
    treated as zero when the validation record is incomplete.
    """
    checks = []
    failures = []
    improvements = []
    for spec in metrics:
        before = _lookup(anchor, spec.name)
        after = _lookup(candidate, spec.name)
        row = dict(metric=spec.name, anchor=before, candidate=after,
                   absolute_tolerance=spec.absolute_tolerance,
                   relative_tolerance=spec.relative_tolerance)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            row.update(status="missing")
            failures.append(row)
            checks.append(row)
            continue
        if not math.isfinite(float(before)) or not math.isfinite(float(after)):
            row.update(status="nonfinite")
            failures.append(row)
            checks.append(row)
            continue
        delta = float(after) - float(before)
        allowance = spec.absolute_tolerance + spec.relative_tolerance * max(abs(float(before)), 1e-6)
        row.update(delta=delta, allowance=allowance,
                   status="improved" if delta < -1e-9 else "unchanged" if abs(delta) <= allowance else "degraded")
        if delta < -1e-9:
            improvements.append(row)
        if delta > allowance:
            failures.append(row)
        checks.append(row)

    contact_metric = _lookup(anchor, "event_aligned/0/1/ball_velocity_rmse")
    contact_candidate = _lookup(candidate, "event_aligned/0/1/ball_velocity_rmse")
    contact_improved = (
        isinstance(contact_metric, (int, float)) and isinstance(contact_candidate, (int, float))
        and math.isfinite(float(contact_metric)) and math.isfinite(float(contact_candidate))
        and float(contact_candidate) < float(contact_metric) - 1e-6
    )
    if require_contact_improvement and not contact_improved:
        failures.append(dict(metric="event_aligned/0/1/ball_velocity_rmse",
                             status="no_contact_edge_improvement"))

    return dict(
        schema="supermimic-wm-candidate-gate-v1",
        passed=not failures,
        checks=checks,
        improvements=improvements,
        failures=failures,
        contact_edge_improved=contact_improved,
        policy="candidate must preserve every protected metric and improve the held-out contact-edge velocity metric",
        metric_specs=[asdict(spec) for spec in metrics],
    )
