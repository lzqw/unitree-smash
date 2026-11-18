# Expanded support-reference position refinement

The previous position-priority result had several support-leg correction
parameters on their local search bounds while still leaving an ~100 mm contact
error. This run increases only the offline search radius for twelve leg
reference corrections from 0.03 to 0.06 rad. It does not alter actuator limits,
PD, contact physics, joint velocity/torque/position bounds, or command clipping.
Every proposed 18D sequence is still independently replayed through the full
original path gates; an invalid path is rejected.

The run starts from the complete selected position-priority prefix. Its q/v,
18D command history and arm references are checked exactly. The same explicit
projected analytic goal and event are used.

| physical residual | previous selected | expanded-leg selected |
|---|---:|---:|
| contact position | 99.744665 mm | **89.735171 mm** |
| normal vector | 0.1227874 | **0.1226542** |
| racket velocity | 1.8680523 m/s | **1.8679742 m/s** |

The selected candidate passed all 33 prediction intervals. Runtime 232.3996 s,
27 fresh rollouts, replay exit 0. The matched predecessor history audit passed
exit 0. CPU regression: 47 passed, exit 0.

The final gates remain false: position 89.735 mm > 3 mm, normal 0.122654 >
0.12, velocity 1.86797 m/s > 0.75 m/s. No teacher reference is qualified or
installed. This is CPU-model evidence only; it is not PhysX or native success.

No native evaluation, 128-case evaluation, training or SuperMimic operation
occurred. PHYSX_REFERENCE_VALID, NATIVE_HIT, RECOVERY_SUCCESS and COMPLETE5 are
NOT_RUN. The next meaningful step is to modify the upstream contact event/base
trajectory rather than continue enlarging local correction bounds; preserving
all original physical gates is required.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
