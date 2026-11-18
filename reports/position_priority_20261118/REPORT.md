# Position-priority continuation after velocity optimization

This run starts from the selected velocity-priority full prefix. Its complete
q/v state, final 18D commands and arm references are checked node by node before
optimization. The same explicit projected task goal and causal contact event
are retained. The objective minimizes contact-position residual; normal and
racket-velocity task groups are protected by terminal non-regression constraints.
All original path, tracking, joint, contact and mapping checks remain active.

## Matched results

| physical residual | velocity-priority predecessor | position-priority selected |
|---|---:|---:|
| contact position | 110.951216 mm | **99.744665 mm** |
| normal vector | 0.1230509 | **0.1227874** |
| racket velocity | 1.8683924 m/s | **1.8680523 m/s** |

The selected proposal is a real 18D command change. It passed all 33 prediction
intervals. Runtime was 236.9228 s with 27 fresh rollouts; replay exit 0. The
matched history audit passed exit 0. Full physical gates remain false: position
99.745 mm > 3 mm, normal 0.122787 > 0.12, and velocity 1.86805 m/s > 0.75.
No valid teacher reference is emitted.

47 CPU regression tests passed, exit 0. No native run, 128-case evaluation,
training or SuperMimic operation occurred. CPU future state remains model
output, not native ground truth. PhysX reference, native hit, Recovery success
and complete5 are NOT_RUN.

The position and speed gaps are now both measured on a progressively matched
sequence, rather than on independently optimized targets. The remaining large
position gap indicates that local terminal command correction is insufficient;
the next change must alter contact event/base state or upstream arm reference,
while preserving the same strict gates. No tolerance was relaxed.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
