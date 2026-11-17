"""Optional task-constrained normal refinement; never a contact certificate.

The original position-only anchor is immutable. A candidate with a better
face normal is used only if exact FK confirms the original physical position
budget and the unchanged joint bounds. Solver success is not sufficient.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


def refine_contact_normal(kinematics, anchor_q, candidate_q, lower, upper,
                          position_target, normal_target, position_budget_m,
                          *, max_iterations=32):
    started = perf_counter()
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    anchor = np.asarray(anchor_q, dtype=float)
    current = np.asarray(candidate_q, dtype=float)
    target = np.asarray(position_target, dtype=float)
    normal = np.asarray(normal_target, dtype=float)
    budget = float(position_budget_m)
    if (any(x.shape != (6,) for x in (low, high, anchor, current))
            or target.shape != (3,) or normal.shape != (3,)
            or not all(np.isfinite(x).all() for x in (low, high, anchor, current, target, normal))
            or not 0. < budget <= .003 or not np.all(low < high)
            or np.linalg.norm(normal) < 1e-12):
        raise ValueError("Finite constrained-normal task and strict position budget required")
    normal = normal / np.linalg.norm(normal)

    def sample(q):
        point, face, jp, jn = kinematics(np.asarray(q, dtype=float))
        point = np.asarray(point, dtype=float)
        face = np.asarray(face, dtype=float)
        jp, jn = np.asarray(jp, dtype=float), np.asarray(jn, dtype=float)
        if (point.shape != (3,) or face.shape != (3,) or
                jp.shape != (3, 6) or jn.shape != (3, 6) or
                not all(np.isfinite(v).all() for v in (point, face, jp, jn))):
            raise ValueError("Invalid exact-FK normal refinement")
        return point, face, jp, jn

    best = [anchor.copy(), float("inf"), float("inf")]
    def consider(q):
        q = np.asarray(q, dtype=float)
        if np.any(q < low-1e-9) or np.any(q > high+1e-9) or not np.isfinite(q).all():
            return
        point, face, _, _ = sample(q)
        ep = float(np.linalg.norm(point-target))
        en = float(np.linalg.norm(face-normal))
        if ep <= budget+1e-10 and en < best[1]-1e-10:
            best[:] = [q.copy(), en, ep]

    consider(anchor)
    consider(current)
    # The nonlinear position constraint is scaled to O(1) at the boundary.
    # Its Jacobian is exact for the same tool/contact-point FK as the objective.
    def objective(q):
        _, face, _, _ = sample(q)
        d = face-normal
        return .5 * float(d @ d)

    def objective_jac(q):
        _, face, _, jn = sample(q)
        return jn.T @ (face-normal)

    def position_slack(q):
        point, _, _, _ = sample(q)
        d = point-target
        return 1.-float(d @ d)/(budget*budget)

    def position_slack_jac(q):
        point, _, jp, _ = sample(q)
        return -2.*(point-target) @ jp / (budget*budget)

    # Never claim infeasibility from one local optimum; this is a deterministic
    # refinement of the existing anchor, not a global pose certificate.
    seeds = [best[0].copy()]
    if np.linalg.norm(current-anchor) > 1e-8 and np.linalg.norm(current-seeds[0]) > 1e-8:
        seeds.append(current.copy())
    attempts = []
    for seed in seeds:
        if position_slack(seed) < -1e-10:
            continue
        outcome = minimize(objective, seed, jac=objective_jac,
                           bounds=list(zip(low, high)), method="SLSQP",
                           constraints=[{"type":"ineq","fun":position_slack,
                                         "jac":position_slack_jac}],
                           options={"maxiter":max_iterations, "ftol":1e-9, "disp":False})
        consider(outcome.x)
        attempts.append(dict(success=bool(outcome.success), status=int(outcome.status),
                             iterations=int(outcome.nit), message=str(outcome.message)))
    # Recheck on fresh FK; no optimizer tolerance can override the position cap.
    q = best[0].copy()
    point, face, _, _ = sample(q)
    ep = float(np.linalg.norm(point-target))
    en = float(np.linalg.norm(face-normal))
    if ep > budget+1e-10 or np.any(q < low-1e-9) or np.any(q > high+1e-9):
        raise RuntimeError("Constrained normal solver attempted unsafe output")
    return dict(q_rad=q, position_error_m=ep, normal_vector_error=en,
                position_budget_m=budget, attempts=attempts,
                seconds=perf_counter()-started,
                physical_feasibility_verified=False)
