"""Avoid foot-foot impact in a predictor that only models foot-ground response.

This adds an explicit planning clearance around unchanged authored foot
spheres. It does not change native collision radii, manufacture pair impulses,
or prove that all other self-collisions have been excluded.
"""
import itertools
import json

import numpy as np


class FootSeparationPathConstraintV1:
    def __init__(self, base, spheres, *, margin_m=.005):
        self.base = base; self.model = base.model; self.shapes = base.shapes
        if not np.isfinite(margin_m) or not 0 <= margin_m <= .01:
            raise ValueError('Explicit planning margin between 0 and 10 mm required')
        if [s['body'] for s in spheres] != self.model.feet:
            raise ValueError('Original four authored foot spheres required')
        self.spheres = spheres; self.margin_m = float(margin_m)
        self.calls = 0; self.last_violation = None; self.minimum_gap_m = float('inf')

    def evaluate(self, q, *, jacobian=False):
        self.calls += 1
        try:
            # The base updates all frame placements for this exact q. Keep
            # its ground outputs/indices and hip-mount constraints unchanged.
            distances, rows = self.base.evaluate(q, jacobian=jacobian)
        except ValueError:
            self.last_violation = getattr(self.base, 'last_violation', None)
            raise
        centers = [self.model.data.oMf[self.model.body_frames[s['body']]].act(
            np.asarray(s['center_body_m'])) for s in self.spheres]
        for a, b in itertools.combinations(range(4), 2):
            gap = float(np.linalg.norm(centers[a]-centers[b])
                        - self.spheres[a]['radius_m'] - self.spheres[b]['radius_m'])
            self.minimum_gap_m = min(self.minimum_gap_m, gap)
            if gap < self.margin_m-1e-6:
                self.last_violation = dict(constraint_type='foot_pair_clearance',
                    bodies=[self.spheres[a]['body'], self.spheres[b]['body']],
                    gap_m=gap, planning_margin_m=self.margin_m,
                    violation_m=self.margin_m-gap, prediction_geometry_call=self.calls)
                raise ValueError('foot_pair_clearance:' + json.dumps(self.last_violation, sort_keys=True))
        return distances, rows
