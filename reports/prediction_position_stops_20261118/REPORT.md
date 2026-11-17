# Coupled stops and authored nonfoot collision prediction

Base e096d4e; teacher worktree only. No training, native experiment, SuperMimic modification or128-case evaluation. This is historical CPU model validation, not teacher admission.

## Change and causal evidence

The previous long-horizon sphere model checked joint position limits without modeling stops. This work wires the repository's existing `position_limited_velocity_step` into an opt-in diagnostic branch, retaining unchanged speed, torque and position limits. A preliminary equivalent interval solver was tested first, then removed in favor of the existing canonical implementation. The preliminary patch and all attempts remain available; final production-default speed behavior is unchanged.

Geometric audit then showed predicted Head_lower penetration of2.7–3.0mm by state tick212, with no corresponding ground reaction in the feet-only model. Native recorded geometry approaches this same authored sphere's ground surface later. Added a diagnostic-only nonfoot sphere contact response, using the original0.047m Head_lower sphere and the existing coupled friction/PD/limit solver. It is an actual geometric constraint reaction, not an arbitrary base force. Nonfoot contact remains inadmissible for a teacher reference.

Full mesh/cylinder ground contacts, self-collision manifolds, PhysX material combination, finite solver compliance and restitution equivalence are NOT established. The same interface friction0.65 is retained; material calibration is still pending. Reproducing a historical head-contact trajectory does not turn it into clean recovery.

## Identical recorded commands, no parameter scan

Initial native state at tick201, both original lanes,80control intervals each, dt0.01s,160integration substeps. Future truth is only used for errors; recorded commands are prescribed open-loop inputs. No future native states enter prediction.

| Model | Completed intervals /160 | Max root position error | Max racket body-origin error |
|---|---:|---:|---:|
| Previous feet + speed constraint |160/160|0.80379m|2.88333m|
| Added ideal position stops, preliminary |130/160|0.80285m|2.85919m|
| Added authored Head_lower response, canonical |160/160|0.07344m|0.22388m|

Stops-only lane0 fails at tick251 on contact complementarity2.11146e-7 versus unchanged2e-7 solver threshold; failure is retained, not rounded into success. Lane1 completes. Comparison of maxima for incomplete runs is not a full-panel accuracy claim.

Canonical final maximum joint velocity error is22.46896rad/s, joint position error0.19714rad. Mean maximum joint velocity error per interval is1.43207rad/s. Numerical completion is not physical admissibility.

A separate preliminary one-step test resets to measured state each interval:160/160solves, max joint velocity error2.40973rad/s, max racket origin error1.119mm. This MUST NOT be represented as continuous800ms accuracy.

## Explicit contact proxy validation

Frozen historical checkpoint seed: face_sign1, string_uv[0,0.07], giving the teacher's ball-center contact proxy offset[0.0256,0,0.07]m. This is not the physical string plane itself. Evaluating the same FK/Jacobian at predicted and recorded native q/v yields maxima:

- proxy position error0.238564m;
- normal vector error0.132732;
- point velocity error0.877925m/s.

These are not independent PhysX contact measurements. They fail precise contact requirements and do not justify teacher promotion.

The first postprocessor output FINAL_CONTACT_PROXY.json contained a normal-vector alias to Pinocchio's mutable workspace, incorrectly reporting zero normal error. It is INVALID for normal error. The bug was fixed with an explicit copy and a nonzero-rotation regression. FINAL_CONTACT_PROXY_CORRECTED.json is authoritative. Prediction trajectories were unaffected; no dynamics rerun was required for this reporting fix.

## Tests and provenance

FINAL_TESTS_CORRECTED.txt:28passed, exit0. Tests include canonical mechanical momentum/reaction checks, inactive-path regression, invalid intervals, sphere geometry preservation, nonfoot/foot-mask separation, command history isolation and mutable FK-workspace isolation.

long01, one_step01, nonfoot01, canonical_nonfoot01 each exit0 as offline diagnostics; solver failure is a recorded row, not process failure. PRELIMINARY_IMPLEMENTATION.patch reproduces the provisional implementation; final canonical implementation has matching output statistics for nonfoot01 and canonical_nonfoot01. Geometry and source hashes are retained.

The legacy velocity_limit_impulse fields in preliminary runs with position stops combine both joint reactions. Do not interpret them as independently measured speed braking. Final canonical implementation separates position and speed reactions.

## Next step

1. Inspect the earliest pre-head-contact drift and existing authored hip/mount self-collision response against the same native states; do not retune physical limits.
2. Validate other active nonfoot contact geometry and material/drive semantics using actual native evidence, keeping uncertain collision models diagnostic-only.
3. Continue reporting contact point/normal/velocity errors, not only body-origin position. Use short causal prediction and independent from-reset PhysX validation for future candidates only after task and execution gates pass.

No complete new teacher reference, new native hit, Recovery orcomplete5 is claimed. Accurate long-horizon prediction remains unfinished.
