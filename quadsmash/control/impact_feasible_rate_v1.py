"""Offline bounded joint-rate search driven by the *actual* analytic return.

This is NOT a new physical velocity limit, an alternative scoring threshold,
or a PhysX certificate. It only retargets an unreachable IDEAL racket speed
when a different rate in the original joint-rate box predicts a valid return.

The requested racket velocity and the target generator's legal outgoing
velocity are retained as separate immutable audit fields. All physical
position/normal, whole-body and final 3-mm impact checks remain downstream.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares


TASK_FLAGS = ("contact_feasible", "legal_return_predicted",
              "attack5_predicted", "target_landing_predicted")


def optimize_impact_feasible_rate(
        jacobian, base_velocity, initial_rate, rate_limits,
        future_to_planning_rotation, metadata, impact_evaluator,
        desired_outgoing_world, *, max_nfev=28):
    """Find an analytic TASK-valid racket rate within original per-joint bounds.

    jacobian, base_velocity: future-base frame.
    rotation: future-base -> fixed current planning frame (R).
    impact_evaluator: authoritative *analytic* outcome callback for this event.
    The callback transforms fixed-planning-frame fields into world before its
    physical string-contact/outgoing integration.
    """
    started = perf_counter()
    J = np.asarray(jacobian, dtype=float)
    base = np.asarray(base_velocity, dtype=float)
    qd0 = np.asarray(initial_rate, dtype=float)
    cap = np.asarray(rate_limits, dtype=float)
    R = np.asarray(future_to_planning_rotation, dtype=float)
    outgoing_target = np.asarray(desired_outgoing_world, dtype=float)
    if (J.shape != (3, 6) or R.shape != (3, 3)
            or any(x.shape != (3,) for x in (base, outgoing_target))
            or any(x.shape != (6,) for x in (qd0, cap))
            or not all(np.isfinite(x).all()
                       for x in (J, R, base, qd0, cap, outgoing_target))
            or np.any(cap <= 0) or np.any(np.abs(qd0) > cap + 1e-8)
            or not np.allclose(R.T @ R, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(R),1.,atol=2e-5)
            or type(max_nfev) is not int or not 1 <= max_nfev <= 100
            or not callable(impact_evaluator)):
        raise ValueError("Finite, originally bounded impact-rate inputs required")
    required_meta = (
        "contact_time_s", "contact_point_local",
        "achieved_position_current_frame_m",
        "achieved_normal_current_frame",
    )
    if any(k not in metadata for k in required_meta):
        raise ValueError("Missing impact event metadata")

    normal=np.asarray(metadata['achieved_normal_current_frame'],dtype=float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-12:
        raise ValueError("Finite planning-frame contact normal required")
    normal=normal/np.linalg.norm(normal)
    best = None
    best_quality = (float("inf"), float("inf"))
    best_diagnostic = None
    n_queries = 0
    failure_reasons = {}
    task_failures = {key: 0 for key in TASK_FLAGS}
    cache = {}

    def evaluate_rate(rate):
        nonlocal best, best_quality, best_diagnostic, n_queries
        rate = np.asarray(rate, dtype=float)
        if rate.shape != (6,) or not np.isfinite(rate).all():
            raise ValueError("Nonfinite impact rate trial")
        # SciPy bounds may produce a few ulps of numerical overshoot. We
        # evaluate only inside original physical speed bounds, and never use
        # this projection as an executed velocity or physical certificate.
        rate = np.clip(rate, -cap, cap)
        key = rate.astype("<f8").tobytes()
        if key in cache:
            return cache[key]
        achieved = R @ (base + J @ rate)
        trial = dict(metadata, achieved_velocity_current_frame_m_s=achieved.copy())
        outcome = impact_evaluator(trial)
        n_queries += 1
        if not isinstance(outcome, dict):
            raise ValueError("Impact evaluator must return a mapping")
        for flag in TASK_FLAGS:
            if outcome.get(flag) is not True:
                task_failures[flag] += 1
        reason = outcome.get("rejection_reason")
        if reason:
            failure_reasons[str(reason)] = failure_reasons.get(str(reason), 0) + 1
        raw = outcome.get("outgoing_velocity")
        if raw is None:
            # In a nonapproaching or non-string case this objective has no
            # usable derivative; reject rather than inventing an outgoing.
            residual = np.full(3, 10., dtype=float)
            output_error = float("inf")
        else:
            v_out = np.asarray(raw, dtype=float)
            if v_out.shape != (3,) or not np.isfinite(v_out).all():
                raise ValueError("Invalid finite predicted outgoing velocity")
            residual = (v_out - outgoing_target) / 5.
            output_error = float(np.linalg.norm(v_out - outgoing_target))
            if best_diagnostic is None or output_error < best_diagnostic[0]:
                best_diagnostic = (output_error, rate.copy(), achieved.copy(), dict(outcome))
        valid = (raw is not None and
                 all(outcome.get(k) is True for k in TASK_FLAGS))
        if valid:
            cost = outcome.get("ranking_cost", float("inf"))
            cost = float(cost) if cost is not None else float("inf")
            if not np.isfinite(cost):
                raise ValueError("Invalid analytic return quality score")
            quality = (cost, output_error)
            if quality < best_quality:
                best_quality = quality
                best = dict(rate=rate.copy(), achieved=achieved.copy(),
                            outcome=dict(outcome), outgoing_error=output_error)
        cache[key] = residual
        return residual

    # The original constrained speed solution is tested first. A second
    # deterministic seed maximizes racket speed along the CONTACT NORMAL.
    # R maps future body to planning frame, not to world: using world outgoing
    # velocity here would mix frames. No case-specific action bank is used.
    seeds = [qd0.copy()]
    direction = R.T @ normal
    saturated = cap * np.sign(J.T @ direction)
    if not np.allclose(saturated, qd0, atol=1e-8):
        seeds.append(saturated)
    diagnostics = []
    for seed in seeds:
        evaluate_rate(seed)
        # Skip nonlinear solve if even the seeded result already predicts
        # the original full ball-side task; valid seed remains in 'best'.
        if best is not None:
            break
        fit = least_squares(evaluate_rate, np.clip(seed, -cap, cap),
                            bounds=(-cap, cap), max_nfev=max_nfev,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        evaluate_rate(fit.x)
        diagnostics.append(dict(success=bool(fit.success), status=int(fit.status),
                                nfev=int(fit.nfev)))
        if best is not None:
            break
    return dict(
        accepted=best is not None,
        qd_rad_s=None if best is None else best["rate"],
        achieved_velocity_current_frame_m_s=(
            None if best is None else best["achieved"]),
        analytic_outcome=None if best is None else best["outcome"],
        target_outgoing_velocity_world_m_s=outgoing_target.copy(),
        original_rate_rad_s=qd0.copy(),
        seeds_rad_s=[seed.copy() for seed in seeds],
        seed_direction_future_base=direction.copy(),
        original_requested_velocity_error_preserved=True,
        best_predicted_outgoing_error_m_s=(
            None if best_diagnostic is None else best_diagnostic[0]),
        best_trial_diagnostic=(None if best_diagnostic is None else dict(
            rate_rad_s=best_diagnostic[1],
            achieved_velocity_current_frame_m_s=best_diagnostic[2],
            analytic_outcome=best_diagnostic[3],
            execution_authorized=False)),
        task_flag_failures=task_failures,
        trials=n_queries, failures=failure_reasons, solvers=diagnostics,
        prediction_kind="analytic_impact_rate_retarget_not_physx",
        physical_feasibility_verified=False,
        seconds=perf_counter()-started,
    )
