# Same-event history-preserving joint refinement continuation

Base commit c89a6a50b5adc27355b8886503a296a20a7bf589. Offline teacher development only; no native/128 evaluation, training, SuperMimic operation, scorer/asset/physical-limit change, or Recovery modification.

## Implementation

`--previous-refinement` continues from a complete selected same-event model prefix instead of silently reverting to the initial hold-tail seed. It checks the artifact SHA, original core inputs, ancestry hashes, exact target metadata, event time, ordered33-node state history, signed actual tracking, finite18D commands and exact per-node arm references. The original initial reference is preserved. Missing/partial/mismatched data is rejected; returned mutable data is copied.

The baseline is still freshly integrated from the original q/v and actual previous final command. Its entire q/v sequence must be exactly equal to the selected predecessor prediction. No intermediate ideal state, old solver workspace or native future state is used as an initial condition. The same24-parameter bounded correction,198 tracking constraints, command bounds, terminal-task protection and original1/half/quarter schedule remain in force. This is one local optimization continuation, not a parameter scan or a new native candidate.

## Same-input, same-target CPU result

| Metric | Previous selected sequence | New half-step |
|---|---:|---:|
| Position error | 173.850565mm | 149.904946mm |
| Normal-vector error | 0.330989113 | 0.329378624 |
| Requested velocity error | 2.990948934m/s | 2.983944701m/s |

All three residuals decrease against the unchanged0.33s target.33/33 path intervals pass for the half-step. Maximum arm tracking is0.249265693082rad. The full-step proposal fails at zero-based node16 with0.250076202848rad, above the unchanged0.25rad gate. Its partial prefix is not executed or accepted. The local linear tracking model's largest error for the accepted half-step is4.00549100103e-05rad at node30, arm index0; this measures local CPU approximation accuracy, not agreement with PhysX.

The final3mm position,0.12 normal and0.75m/s velocity-error thresholds all remain failed. No actual forward collision task is certified and no qualified teacher reference is installed/exported. PhysX reference, native hit, Recovery and complete5 remain NOT_RUN. Do not confuse complete path validity with complete task acceptance.

## Verification and timing

60 tests passed, exit0. New tests verify exact same-event/target continuation, preserved initial and per-node references, isolation, and rejection of partial or drifted histories. Existing path, command, contact and bundle gates pass. Source/input/protected hashes verified after completion.

27 fresh full-prefix model calls took224.129767s; CPU exit0. Maximum CPU command mapping round-trip error is4.4408920985006262e-16rad. This is not a native actuator audit. Full raw traces, reference histories, rejected full step, command hashes and exact reproduction command are preserved.

## Remaining work

Continuation now genuinely builds on prior verified model paths. Remaining position/normal/velocity gaps are still substantial. Full steps continue to touch the tracking boundary and fail slightly because the local response is approximate; only independently verified backtracking is accepted. Future work should quantify local model headroom and terminal reach/task constraints, preserving all original bounds. No widening of the physical tracking gate, blind promotion, or128-case evaluation is justified. Independent native model calibration and actual ball contact/Recovery closure remain outstanding.
