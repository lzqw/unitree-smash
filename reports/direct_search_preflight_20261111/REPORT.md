# Direct full-sequence rescue CPU gate

Date: 2026-11-11 (Asia/Shanghai)

This gate precedes the single authorized native evaluation for the direct command-sequence rescue candidate. It is read-only with respect to the simulator and does not execute any candidate command.

## Result

`CPU_PREFLIGHT_PASS`.

- Fixed evaluation set: 74 recovery cases from the saved 128-case run.
- Candidate generation: two state-derived full-authority constant-target sequences, leg extension factors 2.25 and 3.0.
- Sequence length: 32 nodes over the existing 0.32 s prediction window.
- Full sequence acceptance: 7/74 cases (TRAIN 7, 17, 23, 32, 37, 60, 68).
- Partial candidates are never executed.
- Per-node execution preview uses the existing limiter/PD command mapping and branch-local final-command history.
- No case IDs, failure labels, or trajectory-library matching are used to select a candidate.

## Failure diagnostics

The first persistent-fallback replay contains explicit geometry and solver records. Geometry rejection is attributed to concrete object pairs (`ground / Head_lower` and four `authored_net / Racket::net` events); no foot-pair geometry violation was observed. Solver records distinguish solved and primal-infeasible nodes and retain residual/constraint fields. Inactive feet receive zero contact force in the QP contact model. No concrete checker implementation error was found in this CPU replay, so the native gate is allowed to proceed.

## Native gate

The only permitted native run is the launch command in `../direct_search_native_preparation_20261111/launch_command.sh`. It keeps the original 128 environments, seed, 8 s episode, physics, actuator limits, contact criteria, and evaluation interface. If the native run is started, no second seed or automatic follow-up run is permitted.
