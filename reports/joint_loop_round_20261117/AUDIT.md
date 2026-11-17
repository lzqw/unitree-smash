# Joint-loop development audit

Baseline: 782375ac5ede91b9aade8ed3cd1400a4a08dc7a2. Workspace: /home/autodrive/LZQW/unitree_smash_teacher_eval. Controller type: frozen_base_assisted_programmatic_teacher. No training, optimizer creation or SuperMimic changes are part of this patch. Recovery implementation, scoring, assets and physical constraints are unchanged.

## Changes and evidence

- Impact output separates geometric contact, legal return, attack5 (speed >=5 m/s, forward and downward), target landing, and conditional predicted ball-side completion. Missing event/context requirements remain unverified. Native complete5 remains the original scorer's decision. Regression explicitly keeps the 2.88 m/s example legal but not attack5.
- Search uses normalized normal/velocity errors and task eligibility before scalar cost. Predicted full-body mass/Jacobian are supplied for second-stage impact evaluation. Sticking-foot constrained effective inertia remains an approximation, not PhysX calibration.
- Online prefix uses isolated copies of the existing frozen base controller, regenerated causal observations from each predicted state, and per-candidate histories. It never reads saved future native actions. Prediction inference is counted separately.
- Exact accepted 18D prefix is installed and decoded/checked at 2e-6 rad before execution. Recovery owns the final write after its original trigger; no post-terminal unvalidated prefix extrapolation. Partial reset and history ownership tests cover this path.
- Native planning state comes from state_from_native. Terminal measurements transform into the fixed planning frame. Initial and predicted geometry rejections include actual authored shape witnesses.
- Probe explicitly enables impact evaluation, joint-prefix requirement and unilateral_release. It uses original origin0 physics, seed 882072, panel indices [0,64], 800 ticks, original Recovery and ground-contact observer semantics. No 128 evaluation is authorized by this probe result.

## Geometry attribution

GEOMETRY_REPLAY.json and NATIVE_POSE_COMPARISON.json identify Head_lower, /World/Go2/Head_lower/collisions/mesh_0/sphere. Initial gaps are near zero, then predicted gaps become -1.187834 mm and -1.228827 mm. Reconstructed root translation differs from saved native poses by less than 0.4 micrometres; rotation errors are below 2e-8 rad. This evidence does not support millimetre reconstruction error. Authored sphere support is not a native force/contact measurement. These historical foot-release samples are not certified successful takeoffs. No ground height or tolerance was changed.

## Validation and limitations

CPU_TEST_OUTPUT.txt: 107 passed, exit 0, prior to final runner/inference instrumentation. FINAL_CPU_TEST_OUTPUT.txt: final focused production-function regression, 53 passed, exit 0. CAUSAL_LEG_REPLAY.json confirms deterministic independent candidate history and unchanged live history, but its multi-step replay is rejected at node 2 by active-set nonconvergence. Do not call that replay dynamically feasible.

The sole frozen native attempt is attempts/probe01. Source/config hashes and command are recorded in its manifest. Assets were checked during the run in ASSET_AUDIT_DURING_RUN.json; this is not represented as a prelaunch full USD audit.

## Newly identified blocker; no in-run correction

IMPACT_REJECTION_REPLAY.json reproduces lane 0's impact rejection using saved causal observations and the frozen candidate. The contact/ball gap is 0.1002279445 m. target_state_fn adds the legacy nominal-target displacement to an already propagated world ball position. This is duplicate target compensation. Offline instrumentation only inspected the prefilter, not a second native candidate or full-body certification. The frozen source is preserved. Fix this coordinate bug with regression before another candidate, after this round stops.

The online interfaces are wired, but actual accepted-prefix execution and effective return are NOT demonstrated. Fixed foot-body contact anchors, true moving-surface first intersection and measured finite-mass collision calibration remain limitations.
