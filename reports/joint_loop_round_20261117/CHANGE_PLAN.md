# Change-to-test mapping

1. Separate legal return and attack5/ball-side conditions: impact evaluator and tests/test_impact_outcome_evaluator_v1.py.
2. Normalize and prioritize task results: contact_first_trajectory_search_v1.py and its production search tests.
3. Causal frozen-base legs, isolated candidate state: causal_frozen_leg_prefix_v1.py and CAUSAL_LEG_REPLAY.json (isolation passes; physical replay rejects).
4. Exact joint prefix with Recovery priority: controller and tests/test_joint_prefix_execution_v1.py.
5. Native initial state, fixed-frame terminal measurement, mass/Jacobian: attach_joint_contact_prefix_v1.py; small native attempt exercises request formation, but prefilter rejection prevents full-body adoption.
6. Geometry witnesses: authored_ground_clearance_v1.py, fullbody evaluator, GEOMETRY_REPLAY.json and native pose comparison.
7. Native integration: probe_full_racket_rally_v4.py and SmallPanelGroundObserver; separate live/predictive actor audits, no optimizer.

All rejected native candidates mean stop and report; do not expand candidates, change tolerances, run 128 or launch another probe in this round.
