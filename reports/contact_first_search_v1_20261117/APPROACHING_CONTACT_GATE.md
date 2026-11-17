# Causal relative-normal impact admission

The prior contact gate only required nonzero absolute racket speed. A fast
racket and a ball moving away from its selected face could therefore pass.
The rally callback now supplies the predicted incoming shuttle velocity at
each candidate time, in the same fixed planning frame as the other targets.

For selected outward normal n, require (v_racket - v_ball) dot n > 0.
The normal and achieved point velocity are rotated from future body frame to
the fixed current frame before this check. Separating and tangential candidates
are rejected as nonapproaching_contact. Selected velocity, closing speed and
whether the check ran are recorded. Callers not supplying incoming velocity
retain the prior path and explicitly report approaching_contact_checked=false.

Test command:
```
bash scripts/python.sh -m pytest -q tests/test_contact_first_trajectory_search_v1.py tests/test_full_racket_residual_v4.py tests/test_contact_first_base_transport_v1.py
```
Actual result: 31 passed in 1.38s; exit 0. Regressions include a fast but
separating racket/ball pair and a candidate-time-dependent incoming velocity.

This is only a necessary approach condition, not proof of geometrical contact,
correct ball-side placement, finite-inertia collision response or outgoing
quality. Native collision response is PhysX-only according to the locked
shuttle physics config; the repository's analytic collision helper must not
be represented as an exact PhysX predictor. No native evaluation was started.
