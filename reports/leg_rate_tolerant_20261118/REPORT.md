# Leg endpoint-rate refinement with explicit local tolerance

The previous leg endpoint-rate proposal reduced position but was rejected by a
zero-tolerance terminal non-regression check because racket velocity changed by
about 0.0008 m/s. This diagnostic run enables an explicit normalized local
non-regression tolerance of 0.005. This is an optimizer acceptance tolerance,
not a change to the original 0.75 m/s task gate or any physical limit.

Starting from the complete 67.21 mm prefix, the selected 36-parameter leg-rate
proposal was independently replayed through all 33 intervals:

| residual | baseline | selected |
|---|---:|---:|
| contact position | 67.213736 mm | **60.228075 mm** |
| normal vector | 0.1203062 | **0.1199730** |
| racket velocity | 1.8553705 m/s | 1.8561646 m/s |

The selected path passed every original path check. The change in racket
velocity is 0.000794 m/s and is within the explicitly recorded local tolerance;
the original full task still fails because 1.856 m/s > 0.75 m/s. Runtime
374.0649 s, 38 fresh rollouts, replay exit 0. The predecessor history audit
passed exit 0. 48 CPU regression tests passed, exit 0.

This is not a qualified teacher trajectory: position remains 60.228 mm, and
there is no PhysX/native confirmation. No 128-case evaluation, training or
SuperMimic operation occurred. The tolerance is bounded to 0.01 and defaults
to 1e-9; hardware, command, contact, torque, speed and final scoring gates are
unchanged.

Next work must improve the upstream base/contact state and actual racket speed;
repeatedly increasing local leg freedom is unlikely to close the remaining
60 mm and 1.1 m/s gaps.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
