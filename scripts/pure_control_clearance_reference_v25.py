"""Experimental surface-margin objective for existing full-body candidates.

Keeps V15's actuator, contact, command authority, path checks and final stance.
The 10 mm / 50 ms soft planning preference is not a success threshold, an
identified error bound, a safety certificate or a substitute for native contact.
"""
import numpy as np

from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.event_recovery_manager_v1 import authored_clearance_state


def surface_margin_penalty(surfaces, margin_m=.01, approach_s=.05):
    if margin_m <= 0 or approach_s < 0:
        raise ValueError('Positive planning margin and nonnegative approach time required')
    rows = []
    for name, item in surfaces.items():
        if name.endswith('_foot') or '::' in name:
            continue
        gap, rate = float(item['gap_m']), float(item['rate_m_s'])
        if not np.isfinite([gap, rate]).all():
            raise ValueError('Finite current surface geometry required')
        approach_gap = gap + approach_s * min(0., rate)
        violation = max(0., (margin_m - approach_gap) / margin_m)
        rows.append(dict(body=name, gap_m=gap, rate_m_s=rate,
                         linear_approach_gap_m=approach_gap, soft_cost=violation**2))
    if not rows:
        raise ValueError('No nonfoot ground surfaces')
    return sum(row['soft_cost'] for row in rows), rows


class PureControlClearanceReferenceV25(PureControlStanceFootReferenceV15):
    def __init__(self, *args, clearance_weight=10., **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(clearance_weight) or clearance_weight < 0:
            raise ValueError('Finite nonnegative objective weight required')
        self.clearance_weight = float(clearance_weight)

    def _tracking_cost(self, state, desired):
        cost = super()._tracking_cost(state, desired)
        if self.clearance_weight:
            penalty, _ = surface_margin_penalty(authored_clearance_state(self.geometry, state))
            cost += self.clearance_weight * penalty
        return cost

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        penalty, surfaces = surface_margin_penalty(authored_clearance_state(self.geometry, state))
        diagnostic.update(controller='pure_control_clearance_reference_v25',
                          surface_margin_soft_cost=penalty,
                          surface_margin_weight=self.clearance_weight,
                          surface_margin_m=.01, surface_approach_time_s=.05,
                          nearest_approach_surface=min(surfaces, key=lambda row: row['linear_approach_gap_m']),
                          physical_and_success_thresholds_unchanged=True,
                          margin_is_not_a_safety_certificate=True)
        return target, diagnostic
