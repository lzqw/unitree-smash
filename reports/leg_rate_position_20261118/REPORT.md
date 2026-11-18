# Optional leg endpoint-rate diagnostic

This optional mode adds twelve support-leg endpoint-rate parameters to the
existing twelve leg position corrections (36 parameters total). It is disabled
by default and does not change hardware limits, PD, contact dynamics, or the
original path gates. Commands and references are independently rechecked at
every node.

From the complete selected 67.21 mm prefix, the local position proposal reached
60.23 mm and a half-step reached 63.54 mm. Both were rejected because the
strict terminal task non-regression gate detected a small racket-velocity
regression. The optimizer therefore retained the unchanged baseline:

| residual | baseline retained | best rejected proposal |
|---|---:|---:|
| position | 67.213736 mm | 60.228075 mm |
| normal | 0.1203062 | 0.1199730 |
| racket velocity | 1.8553705 m/s | 1.8561646 m/s |

The rejected proposal is not an executable teacher reference. This result is
useful negative evidence: leg endpoint-rate freedom can reduce position in the
CPU model, but not while preserving all existing terminal task groups. No
threshold was relaxed and no command was installed.

The run used 36 bounded parameters, 40 full rollouts, 396.6076 s, and exit 0.
All 33-node paths in accepted trial evaluations passed their path checks;
selected index remains baseline because no verified terminal improvement exists.
47 CPU regression tests passed, exit 0. Native/PhysX, 128-case evaluation,
training and SuperMimic remain NOT_RUN.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
