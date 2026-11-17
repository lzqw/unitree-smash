# Joint leg/arm prediction progress

Base commit: d598e36. Native evaluation: NOT_RUN for this change.

The previous evaluator held every leg at its initial measured joint position.
It now accepts an explicit absolute 12-leg target sequence, copies it into
read-only storage and replays each interval alongside the candidate six-axis
arm target through the existing full 18D implicit PD dynamics. Original joint,
velocity, torque, friction and authored ground-clearance checks remain active.
Missing leg nodes and physical position-limit violations are rejected without
clipping. The default remains the historical measured-leg hold.

Replay can retain the exact commands, full q/v state and measured contacts per
interval. Its SHA256 identifies the actual little-endian float64 commands sent
to the predictor. This is not a native actuator command audit or a replacement
for Recovery's 2e-6 audit.

A separate timing defect was corrected: contact node k now advances exactly k
control intervals. Previously the evaluator advanced k+1 intervals and compared
the resulting state against a target for time k*dt. A test compares the terminal
state against an independent three-interval replay at 0.03 seconds. Tracking
error is checked against the next state sample after each held command.

Validation: 80 tests passed, exit 0; actual output is in
JOINT_REPLAY_TEST_OUTPUT.txt. New tests verify full-inertia base response to a
nonconstant leg sequence, input-copy and repeated-candidate isolation, original
limit rejection, complete sequence coverage and exact terminal timing.

Remaining work: connect a causal leg/WBC trajectory producer and execution
consumer to the unified event planner, and validate them natively. The current
implicit model freezes the measured contact mode; it cannot yet prove a
support-to-flight transition. Do not describe this patch as completed takeoff
optimization. No ground forces or new contact modes are introduced here.
Recovery, SuperMimic, native assets and scoring were not changed.
