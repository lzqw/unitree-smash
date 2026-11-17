# Hip/mount response and prediction horizon validation

Base35a3c9b. Teacher worktree only; no training, native experiment, SuperMimic changes,128-case evaluation or physical/scoring changes.

## Evidence and implementation

The previous prediction omitted hip/mount self-contact. On the SAME recorded80-step trajectories, predicted FR_hip overlap reaches0.084665rad, beginning in the first interval; native overlap remains approximately6.57e-5rad at the authored boundary. This branch reuses the existing geometry-hash-checked HipMountCollisionResponseV1, coupled with existing motor, position, speed, foot and Head_lower reactions. Original motor bounds are unchanged. An authored collision boundary is not a new motor limit.

The initial400-iteration run fails at tick201 in both lanes. Contact residuals are1.57e-6/1.84e-6 against unchanged2e-7 tolerance. Failures remain in long01. The offline solver budget was explicitly increased to2000, without changing equations or tolerances. First interval needed718 iterations. Default remains400. This changes allowed numerical work, not physical acceptance; no native candidate was selected.

## Results

Frozen model/configuration throughout long02 and contact_window01; end hashes checked.

| Metric | Previous head + stops,800ms | Head + stops + hips,800ms | Same new model,320ms from tick249 |
|---|---:|---:|---:|
| Numerically solved / requested |160/160|160/160|64/64|
| Max root position error |73.44mm|11.73mm|0.368mm|
| Max racket body-origin error |223.88mm|78.15mm|2.959mm|
| Max joint velocity error |22.469rad/s|4.003rad/s|0.114rad/s|

Counts are numerical completion, NOT physical feasibility or native hits. At800ms lane0 final racket-origin error is27.42mm, lane1 is78.15mm. Future truth enters comparisons only. Prescribed historical commands are not proof of causal future base-policy rollout accuracy.

Using frozen historical contact proxy[0.0256,0,0.07]m, not the physical string plane:

| Error |800ms|320ms|
|---|---:|---:|
| Maximum point position |83.242mm|3.121mm|
| Maximum normal vector difference |0.053924|0.004278|
| Maximum point velocity |0.34532m/s|0.09259m/s|

Truth is FK/Jacobian evaluated at recorded native q/v, not an independent PhysX contact-point measurement. The320ms maximum still exceeds3mm; no rounding into success. At the previously fixed candidate event times (lane0 tick276; lane1 tick278), errors are1.937mm and1.883mm respectively. These historical-command samples do not certify newly optimized arm sequences.

## Limits and tests

- Partial collision model only: full mesh/cylinder response, manifold friction, compliance and PhysX solver equivalence remain unverified.
- Head contact remains inadmissible teacher behavior even when the diagnostic reproduces it.
- Existing hip response corrects initial tiny overlap on its original10ms time scale; this was not fitted to these samples.
- Short-window accuracy does not establish800ms precision. No controller is modified and no reference is installed/exported.
- FINAL_TESTS.txt:20passed, exit0, including mechanics, hip response, original-limit/input immutability, foot/head separation and FK workspace isolation.
- All three CPU processes exit0; numerical failures are separately preserved. TIMING.json contains wall-clock CPU timings, not GPU or real-time claims.
- The first report-generation command failed (exit1, system Python lacks NumPy); reporting was repaired with standard-library statistics, without installing dependencies or rerunning predictions.

## Next work

1. Diagnose remaining takeoff-state drift with native/predicted momentum and contact geometry, especially lane1.
2. Validate altered arm/leg trajectories against all original full-body constraints. Historical model accuracy cannot certify a new command sequence. Fresh measured states may support short causal prediction; future truth must not become online input.
3. After task, trajectory and execution gates pass, independently validate a frozen candidate from reset in PhysX. These CPU results do not authorize128-case evaluation.

The goal remains incomplete: no new valid teacher reference, native hit, Recovery success orcomplete5 is established.
