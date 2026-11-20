"""Pure helpers for auditable WM sampling and actor rollback records."""


def identification_window_capacity(collections, requested):
    """Upper bound before joint source/event quotas and episode caps are applied.

    Each signed lane covers exactly 16 control ticks, yielding one distinct
    reset-safe 16-step training window. The two nominal repeats are controls.
    """
    available = sum(item["measured_windows"] for item in collections)
    return dict(requested=requested, distinct_signed_windows=available,
                unavoidable_shortfall=max(0, requested-available))


def collect_until_quota(ready, collect, budget, *, interval=256):
    """Test the actual joint sampler, collecting only until it succeeds or the tick budget expires."""
    if budget < 0 or interval < 1:
        raise ValueError('Invalid native bootstrap tick budget')
    used = 0
    while not ready():
        if used == budget:
            raise RuntimeError(f'Native WM bootstrap exhausted {used} ticks with unsatisfied joint quotas')
        count = min(interval, budget-used)
        collect(count)
        used += count
    return used


def sampling_promotion_gate(report, mode="diagnostic"):
    """Return the explicit WM sampling contract for one optimizer batch.

    Diagnostic runs may continue with fallbacks so that missing data is
    measurable. Promotion runs require every requested source and event quota
    to be met without fallback.
    """
    if mode not in ("diagnostic", "promotion"):
        raise ValueError(f"Unknown WM sampling mode: {mode}")
    shortfall = {}
    for field in ("source_shortfall", "stage_shortfall"):
        for name, value in dict(report.get(field, {})).items():
            value = int(value)
            if value > 0:
                shortfall[f"{field}.{name}"] = value
    fallback_count = int(report.get("fallback_count", 0))
    feasible = bool(report.get("joint_quota_feasible", True))
    passed = not shortfall and fallback_count == 0 and feasible
    return {
        "mode": mode,
        "passed": bool(passed),
        "shortfall": shortfall,
        "fallback_count": fallback_count,
        "joint_quota_feasible": feasible,
        "promotion_allowed": mode == "promotion" and bool(passed),
        "diagnostic_continuation_allowed": mode == "diagnostic",
    }


def rollback_audit(*, phase, update_index, proposal_checkpoint,
                   stats, pre_metrics, post_metrics, replay_stream):
    """Build a complete record for a proposal that was backtracked/rejected.

    The learner uses an in-memory parameter snapshot and alpha backtracking.
    ``proposal_checkpoint`` is therefore an auditable reference, not a claim
    that a checkpoint file was restored into the process.
    """
    trials = list(stats.get("backtrack_trials", ()))
    accepted = bool(stats.get("update_accepted", False))
    alpha = float(stats.get("backtrack_alpha", 1.0))
    backtracked = accepted and alpha < 1.0
    rejected = not accepted
    reasons = []
    for trial in (trials[:-1] if accepted else trials):
        if not trial.get("toi_coverage_preserved", True):
            reasons.append("lost_toi_lane")
        if not bool(trial.get("budget_ok", True)):
            reasons.append("command_or_target_budget")
        if float(trial.get("loss", stats.get("objective_before", 0.0))) > float(stats.get("objective_before", 0.0)):
            reasons.append("objective_increase")
    if backtracked:
        reasons.append("accepted_with_reduced_step")
    elif rejected:
        reasons.append(str(stats.get("rejection_reason") or "proposal_rejected"))
    return {
        "phase": phase,
        "update_index": int(update_index),
        "proposal_checkpoint": str(proposal_checkpoint),
        "rollback_reason": ",".join(dict.fromkeys(reasons)) if reasons else None,
        "rollback_mode": "alpha_backtrack" if backtracked else ("full_snapshot_restore" if rejected else "none"),
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "native_metric_scope": "mixed-policy collection through this proposal; updated actor evaluated at next frozen refresh",
        "pre_actor_loss": stats.get("objective_before"),
        "post_actor_loss": stats.get("objective_after"),
        "kl": stats.get("kl"),
        "kl_available": bool(stats.get("kl_available", False)),
        "restored_checkpoint": None,
        "checkpoint_restored": False,
        "parameter_snapshot_restored": bool(rejected),
        "rollback_reference_checkpoint": str(proposal_checkpoint),
        "replay_stream": int(replay_stream),
        "replay_reused_after_rollback": True,
    }
