# Wider bounded leg-support continuation

The prior support-leg position refinement was limited by a 0.06 rad offline
correction radius. This run increases only the search radius to 0.10 rad for
twelve leg reference corrections. It does not change physical actuator limits,
PD gains, joint positions/velocities/torques, contacts, ground geometry or
command clipping. Every candidate is replayed through the original 33-interval
full-body path gates, and invalid candidates are rejected.

The exact selected prefix from `expanded_leg_position_20261118` is restored
first; q/v, 18D commands and references are checked. The same causal event and
explicit projected goal are used.

| residual | previous selected | wider-leg selected |
|---|---:|---:|
| contact position | 89.735171 mm | **67.213736 mm** |
| normal vector | 0.1226542 | **0.1203062** |
| racket velocity | 1.8679742 m/s | **1.8553705 m/s** |

The selected candidate passed all 33 prediction intervals. Runtime 224.2847 s,
27 fresh rollouts, replay exit 0. The matched predecessor audit passed exit 0;
47 CPU regression tests passed, exit 0.

The candidate is still not a valid teacher trajectory: position 67.214 mm >
3 mm, normal 0.120306 > 0.12, and velocity 1.85537 m/s > 0.75 m/s. No native
or PhysX evidence exists. `PHYSX_REFERENCE_VALID`, `NATIVE_HIT`,
`RECOVERY_SUCCESS` and `COMPLETE5` are NOT_RUN. No 128-case evaluation,
training or SuperMimic operation occurred.

This result confirms that support-reference changes materially affect the
predicted terminal contact, but the remaining gap is still large. Further
radius inflation without a new base/event formulation is not justified; next
work should model the upstream takeoff/contact state and validate it with the
same complete gates.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
