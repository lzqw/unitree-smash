# Event-coupled self-verification

The current branch was self-checked after commit `fec393c`.

Passed checks include strict finite-mass admission, contact reachability atlas,
event planner gates, rollout adapter, full-body 18D rollout contract, impact
prediction, terminal refinement, reference continuation and coupled prefix
constraints. Result was 48 tests passed, exit code 0.

The full-body rollout now exposes actual leg and arm command changes through the
existing nonlinear predictor. The strict finite-mass status remains fail-closed
until a candidate supplies mass matrix, contact Jacobian, support mode and
support Jacobian from the same actual predicted strike state.

Small native PhysX validation was **NOT_RUN**. There is no qualified candidate
with `FULLBODY_PREDICTION_VALID` and `FINITE_MASS_VERIFIED`, so starting native
verification would bypass the current admission contract. No native hit,
attack5, Recovery or complete5 result is claimed.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
