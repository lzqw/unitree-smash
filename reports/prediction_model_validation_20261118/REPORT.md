# Recorded PhysX prediction validation — not teacher admission

Base: a20fe5c4f7491306f4191fe2c0ac9a90726a9ac0. Teacher worktree only.

This follow-up adds an independent comparison runner, native-state mapping regression, and evidence summarizer. It evaluates three **existing** CPU models, without altering controller behavior, physics, limits, Recovery, scoring, or SuperMimic. No new native run or training was launched. The source is the previously completed, from-reset 800-tick historical PhysX calibration, not new candidate performance.

## Results

All values below are maximum absolute errors over the stated completed predictions. A solve count is NOT a physical feasibility count.

| Window | Current body-anchor | Authored foot spheres | Spheres + experimental simulator speed constraint |
|---|---:|---:|---:|
| Independent 10ms transitions, 28 requested: solves | 27/28 | 28/28 | 28/28 |
| Maximum joint velocity error, rad/s | 28.8864 | 28.8864 | 1.71772 |
| Open-loop 140ms, 28 requested: solves | 10/28 | 28/28 | 28/28 |
| Maximum racket body-origin position error, m | 0.006660 (incomplete) | 0.003234 | 0.003373 |
| Open-loop 800ms, 160 requested: solves | 2/160 | 160/160 | 160/160 |
| Maximum racket body-origin position error, m | 0.000495 (incomplete) | 2.84934 | 2.88333 |

The experimental coupled speed constraint improves the historical FL-calf saturation transition: at lane0 tick259 the maximum joint velocity error falls from28.0454 to0.01506 rad/s. This is short-horizon model evidence, not a new control result.

Long prediction remains inaccurate. Starting at tick201, both sphere models exceed3mm body-origin error by tick203 in both lanes. Both already report joint-position violations at tick201 (lane0 0.0001587rad; lane1 0.0004074rad). They check positional limits but do not model their reaction response; neither models full nonfoot/self-collision response. These omissions are hypotheses to investigate, not proof of a unique cause. The current body-anchor model instead fails its contact/motor active set at tick202 on both lanes.

The sphere+speed model reaches0.80379m root-position error over800ms. Numerical convergence across160 intervals therefore does not justify promotion to teacher admission.

## Measurement boundaries

- `racket_position_error_m` is the Racket body-frame origin error. It is NOT the actual string contact-point or shuttle-center error;3mm final contact accuracy is unverified.
- Speed reaction is an existing coupled ideal simulator constraint, not post-step velocity clipping. Its impulses are not calibrated as actual finite PhysX drive/solver braking. Motor torque is bounded separately.
- `applied_drive_torque` in the historical source is IsaacLab ImplicitActuator's estimate, not measured PhysX solver drive torque.
- No policy sees future native state. Known recorded18D commands drive this offline prediction; future measured states are comparison labels only. This does not validate causal future base-policy generation.
- Missing/nonfinite physical diagnostics remain unknown. Summary thresholds are display thresholds, never substituted for teacher hard constraints.
- The short runs preceded addition of explicit start/steps metadata; raw results are preserved. Their summaries restore249/14 from executed command configuration and mark this provenance.

## Tests and failures

`FINAL_TESTS.txt`:21passed, exit0. State mapping/link twist, lane/history isolation, coupled impulse mechanics and evidence fail-closed semantics tested.

`comparison01`:exit1, missing `pxr` during live USD extraction. No dependencies or shared environment were changed. Subsequent runs use cached authored sphere geometry with matching original asset and complete dependency SHA256 checks; source export hash is recorded in AUTHORED_SPHERES.json.

`comparison02`, `rollout01`, `long_horizon01`:exit0. All raw per-step errors and diagnostics are retained. Early solver failures stop that model/lane and remain in denominators.

A first report-generation attempt failed with KeyError steps on the initial result schema; no simulation was rerun. Explicit recorded-window metadata resolves reporting only.

## Next work, ordered

1. Validate positional stop and velocity/contact reaction together on the first divergent historical interval. Preserve original limits and reaction/momentum evidence; do not tune to final score.
2. Add complete predicted-state and actual contact-point normal/velocity telemetry, then compare with measured native geometry and contact events. Do not relabel body-origin error as contact accuracy.
3. Investigate omitted nonfoot/self-collision and drive/contact models against the historical trace before trusting long horizons.
4. Retain from-reset isolated PhysX as the independent offline reference check. CPU model failure remains MODEL_UNCERTAIN, not evidence of native impossibility.

PHYSX_REFERENCE_VALID(new candidate)=NOT_RUN; NATIVE_HIT(new candidate)=NOT_RUN; Recovery/complete5(new candidate)=NOT_RUN. No128-case evaluation. Accurate full-horizon prediction and a valid new teacher reference remain unfinished.
