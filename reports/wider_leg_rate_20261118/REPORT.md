# Wider leg-rate support search

The optional leg-rate support search radius for twelve leg position corrections
was increased to 0.20 rad in the offline parameterization. Original actuator
position, velocity, torque, contact, arm tracking and command round-trip gates
remain authoritative. The changed radius only expands the search proposal
space. Invalid full-prefix sequences remain rejected.

The run starts from the complete `wide_leg_position_20261118` prefix. Baseline
history is reproduced within 1e-9 numerical reconstruction tolerance. The
selected proposal passes all 33 intervals.

| residual | baseline | selected |
|---|---:|---:|
| contact position | 67.213736 mm | **58.598094 mm** |
| normal vector | 0.1203062 | **0.1195293** |
| racket velocity | 1.8553705 m/s | 1.8563981 m/s |

Runtime was 925.3688 s with 38 complete rollouts and exit 0. The matched
history audit passed. 48 CPU regression tests passed. The original task gates
remain unsatisfied because position is 58.598 mm and racket velocity error is
1.856 m/s. No valid teacher reference is installed.

Native/PhysX evaluation, 128-case evaluation, training and SuperMimic were not
run. The result supports increasing available push-off motion only as a bounded
offline search. It does not establish that the required height can be produced
by the real robot. Further enlargement is deferred until the support trajectory
is calibrated against native dynamics.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
