# World-model accuracy repair

## Code changes

- `fit_supermimic_action_ball.py` now optimizes every trainable parameter in the selected action-ball model. This includes the material/inertia `mechanical_head` used by V4/V5/V6 candidates.
- Candidate fitting now adds reset-safe four-step autoregressive ball supervision using the same `ball_transition` path used by validation and imagination.
- The transition immediately before a recorded contact edge receives an explicit hard-negative weight. This targets the observed one-tick timing error without using future labels in model forward.
- Candidate metadata records rollout settings, per-stage losses, and the actual trainable parameter names.

## Offline evidence

The real packed frozen archive was used with a one-update CPU fit. The run was diagnostic only and did not modify any checkpoint.

- contact transitions in the training split: 82
- pre-contact hard negatives: 82
- reset-safe rollout starts: 55,641
- rollout horizon: 4 control steps
- CPU optimizer update: 1
- event-aligned candidate selection: rejected because pre-contact velocity RMSE degraded
- final candidate: restored to the initial model because no event-safe checkpoint existed

This confirms the new loss path runs on real data and that the event-safety rule prevents an unsafe candidate from being admitted. It does not establish an improved model.

## Tests

- `tests/test_supermimic_action_ball_fit.py` and `tests/test_supermimic_action_ball_v1.py`: 13 passed.
- Python syntax check: passed.

## Remaining validation

The fitted candidate still lacks same-prefix PhysX action perturbation evidence. CUDA is unavailable on this host, so no native event-aligned probe, GPU fit, actor update, or R→A→R canary was run.

Large-scale training remains stopped until the driver is restored and the repaired candidate passes independent event-aligned holdout plus native response-direction validation.
