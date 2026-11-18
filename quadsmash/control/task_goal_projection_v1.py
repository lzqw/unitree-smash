"""Project a legal analytic goal toward measured model response, not a control certificate.

Contact is CONDITIONAL at the causal ball position. The actual racket may be
elsewhere. This module never changes commands, contact timing, or physical limits.
"""
import numpy as np
from scipy.optimize import minimize
from .task_constrained_impact_goal_v2 import task_margins


def project_task_goal(model, position, incoming, local, seed_normal, seed_velocity,
                      reference_normal, reference_velocity, *, max_iterations=32):
    vectors = [np.asarray(x, dtype=float).copy() for x in
               (position, incoming, local, seed_normal, seed_velocity,
                reference_normal, reference_velocity)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in vectors):
        raise ValueError('Finite world-frame vectors required')
    p, incoming, local, seed_n, seed_v, ref_n, ref_v = vectors
    if any(abs(np.linalg.norm(n)-1) > 1e-8 for n in (seed_n, ref_n)):
        raise ValueError('Unit normals required')
    if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 100:
        raise ValueError('Bounded integer iteration budget required')
    if np.max(np.abs(seed_v)) > 40:
        raise ValueError('Seed outside analytic target domain')
    # A local chart moves the target normal too; this is not fixed-face rate search.
    axis = np.eye(3)[np.argmin(np.abs(seed_n))]
    tangent = np.cross(seed_n, axis); tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(seed_n, tangent)))
    original_errors = np.array([np.linalg.norm(seed_n-ref_n), np.linalg.norm(seed_v-ref_v)])
    scales = np.array([.12, .75])
    cache = {}; best = None

    def evaluate(x):
        nonlocal best
        key = np.asarray(x, dtype=float).tobytes()
        if key not in cache:
            n = seed_n + basis @ x[:2]; n /= np.linalg.norm(n)
            v = np.asarray(x[2:]).copy()
            forward = model.evaluate(p, incoming, p-model.physics['collision_radius_m']*n,
                                     v, n, local)
            margins = task_margins(forward, model.court)
            errors = np.array([np.linalg.norm(n-ref_n), np.linalg.norm(v-ref_v)])
            cost = float(np.sum((errors/scales)**2))
            valid = bool(margins['valid'] and np.all(errors <= original_errors + 1e-10))
            record = dict(normal_world=n, racket_velocity_world_m_s=v,
                          reference_errors=errors, cost=cost, forward_outcome=forward,
                          task_margins=margins, accepted=valid)
            cache[key] = record
            if valid and (best is None or cost < best['cost']):
                best = record
        return cache[key]

    start = np.r_[0., 0., seed_v]
    original = evaluate(start)
    if not original['task_margins']['valid']:
        raise ValueError('Seed fails actual forward ball task: '+original['task_margins']['limiting_constraint'])

    def constraints(x):
        r = evaluate(x)
        return np.r_[r['task_margins']['normalized'],
                     (original_errors**2-r['reference_errors']**2)/scales**2]

    fit = minimize(lambda x: evaluate(x)['cost'], start, method='SLSQP',
                   bounds=[(-2., 2.)]*2 + [(-40., 40.)]*3,
                   constraints=[dict(type='ineq', fun=constraints)],
                   options=dict(maxiter=max_iterations, ftol=1e-8, eps=1e-5))
    evaluate(fit.x)
    # Fresh forward check: cached flags or solver success never authorize a goal.
    chosen = best
    n, v = chosen['normal_world'], chosen['racket_velocity_world_m_s']
    forward = model.evaluate(p, incoming, p-model.physics['collision_radius_m']*n, v, n, local)
    margins = task_margins(forward, model.court)
    if not margins['valid']:
        raise ValueError('Final forward task changed')
    return dict(original=original, selected={**chosen, 'forward_outcome': forward, 'task_margins': margins},
                improved=bool(chosen['cost'] < original['cost']-1e-8),
                solver_success=bool(fit.success), solver_message=str(fit.message),
                iterations=int(fit.nit), queries=len(cache),
                conditional_contact_only=True, robot_feasibility_verified=False,
                actual_commands_changed=False, valid_teacher_reference=False,
                native_execution=False)
