"""Full original recovery remainder with an explicit foot-pair clearance.

Uses the unchanged coupled position/speed-stop dynamics, native command map,
reference clock, terminal conditions and continuous stable-window requirement.
Only the planning path constraints gain a 5 mm foot-pair separation margin.
"""
import numpy as np

from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_constrained_recovery_suffix_v1 import state_from_row
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.position_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.timed_whole_body_recovery_v1 import build_timed_plan


def predict_plan(plan, query, tick, stable_ticks, audit, substeps, deadline):
    w = _WORKER; state = state_from_row(query)
    geometry = FootSeparationPathConstraintV1(
        HipMountPathConstraintV1(w['geometry'], audit, margin_rad=.02),
        w['spheres'], margin_m=.005)
    result = evaluate_suffix(w['controller'], state, build_timed_plan(w, plan),
        w['spheres'], geometry, w['protocol'], control_tick=tick,
        substeps=substeps, deadline=deadline, preceding_stable_ticks=stable_ticks, save=substeps==160)
    for key, field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if not np.array_equal(result[key], query[field]):
            raise ValueError('Predictor changed the supplied measured state/history')
    failure = result['failure']
    if failure and (failure.get('collision') or {}).get('constraint_type') == 'foot_pair_clearance':
        failure['reason'] = 'foot_pair_clearance'
        micro = (geometry.last_violation['prediction_geometry_call']-1) % substeps
        failure['micro'] = micro
        result['rank'][1] = result['required_model_steps']-result['completed_model_steps']-micro/substeps
    result.update(planning_foot_pair_margin_m=.005,
        minimum_checked_foot_pair_gap_m=geometry.minimum_gap_m if np.isfinite(geometry.minimum_gap_m) else None,
        physical_foot_geometry_changed=False, foot_pair_impulse_model_used=False)
    return dict(plan=plan, result=result)
