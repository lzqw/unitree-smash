# Contact Reachability Atlas P1

Added a read-only six-axis contact reachability atlas. It evaluates position,
normal, velocity, joint-limit margin, speed margin and Jacobian singular values
for explicitly supplied causal candidate states. Results are marked
`KINEMATIC_ONLY`, `fullbody_verified=false`, `native_verified=false` and
`teacher_eligible=false`. The atlas cannot inject base states or commands.

Three CPU tests passed. No native evaluation, training or SuperMimic operation
was run.
