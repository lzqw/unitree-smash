# Frozen-base-assisted impact outcome development

Status: IN_PROGRESS. Native probe: NOT_RUN. No full128 evaluation or training.
Controller type: frozen_base_assisted_programmatic_teacher. The user's new
scope explicitly permits frozen walk/standing/PWA strike inference; their
weights are not replaced or optimized. Recovery controller, task scorer and
assets were not changed. SuperMimic was not touched.

## Implemented

ImpactOutcomeEvaluator reuses the authored string strip geometry, the existing
finite scalar mass/Coulomb collision function, locked drag propagation and
court/net dimensions. Incoming shuttle velocity and racket velocity are
separate. It reports string admission, incoming normal velocity, predicted
outgoing velocity and reaction impulse, net clearance, landing, return/target
flags and rejection reasons. Normal effective mass can be supplied via full
M and contact J; tangential response remains a scalar-mass approximation.
No PhysX impulse or shuttle state is overwritten.

The optional contact-first flag evaluate_impact_outcome enables actual per-time
causal world-frame impact evaluation. Invalid predicted returns cannot win on
IK residual alone. Every adjusted face/velocity is reevaluated; ranking retains
normal deviation and requested-velocity error. When a full-body evaluator
supplies measured terminal state, outcome is recomputed from that state.
This coarse prediction does NOT replace complete5, safety or native scoring.
A conservative full-sphere net-clearance gate differs intentionally from the
native tape-touch rule; it is an uncalibrated planning filter, not a new scorer.

Fixed another actual coordinate bug: the controller previously built the
body-from-world basis but applied it as world-from-body. World offsets also
were not transformed. The production helper now has a translated, pitched,
rotated, nonzero angular-velocity regression test.

Frozen base inference instrumentation counts each walk/standing/strike call,
freezes parameters, and checks state-dict SHA before/after. The development
probe explicitly requires --frozen-base-assisted. No approach/recovery learned
heads or policy optimizers are instantiated by that probe. Full Recovery MPC
integration remains a separate required gate; the probe itself is not evidence
that the entire historical joint-teacher chain is running.

## Contact solver evidence

Separating contacts now initialize released rather than forcibly sticking.
Original dynamics, friction, unilateral and motor tolerances are unchanged.
An optional iteration trace records solved modes/forces/contact velocity,
motor saturation transitions, matrix condition and linear residual.
The four-foot 5m/s synthetic failure is fixed with this warm start; disabling
it reproduces 80 iterations and failure. An iteration-exhaustion test still
rejects a genuinely unfinished solve. Four internal substeps with separating
feet match force-free dynamics and never introduce support force.

REAL_TAKEOFF_REPLAY.json uses first chronological measured contact release in
both preserved native lanes, including one tick before/at/after. All12 single
interval cold/warm queries solved without recorded physical-limit violations.
These are historical state CPU replays, not new native successes.

Two recorded full18D command prefixes were also replayed. BOTH were rejected
at node0 by authored nonfoot clearance: 0.0011878341m and 0.0012288274m
penetration respectively. Dynamics residuals were approximately 5e-13 and
4e-13; motor, velocity and position violations were zero. The rejection is
retained. Next: compare reconstructed root/foot/body geometry with the original
native pose trace before deciding whether model mismatch or actual nonfoot
contact caused it. Do not relax the hard clearance gate.

## Tests and remaining work

CPU_TEST_OUTPUT.txt: combined103-test suite passed, exit0, before the final
world-pose regression. Final focused53 tests including that regression passed
in1.30s, exit0. All successes are CPU tests, not measured hit rates.

Not yet complete: live causal frozen-leg reference generation and identical
execution, full-body effective-mass wiring, final joint-controller/native
probe integration and one actual small native probe. No candidate was frozen
for execution while real prefix admission remains unresolved. No full128
candidate, native success rate or completion marker is claimed.
