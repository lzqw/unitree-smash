# Strict finite-mass impact admission P0

Added `StrictImpactAdmissionV1`. A candidate must provide a full floating-base
mass matrix, racket contact Jacobian, explicit `support_mode`, and a support
Jacobian while supported. Flight candidates carrying support constraints are
rejected. Missing robot state fails closed. The adapter distinguishes
`FINITE_MASS_VERIFIED` from `EFFECTIVE_MASS_UNVERIFIED` and never grants teacher
eligibility by itself.

The existing analytic evaluator remains available for coarse ranking. No
physical limit, collision asset, scorer, Recovery controller or SuperMimic was
changed. Native evaluation was not run.

9 related tests passed, exit 0.
