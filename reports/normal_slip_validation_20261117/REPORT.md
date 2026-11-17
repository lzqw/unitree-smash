# Normal/slip follow-up: actual server validation and fixes

**Result: code/test progress, full control closure NOT_COMPLETED. No native probe,128 evaluation or training.**

Started at `3308200`, fetched and merged review `fd0e9d911bf04295b3147961b430738769197d45` (merge `f4944f1`) into the existing independent teacher worktree. This preserves the user's newer **offline reference generation** policy. The old review report's100ms acceptance prerequisite is superseded: time is telemetry, not a reason to reject an otherwise valid offline reference. Original physical/command/task gates remain mandatory.

## What changed after reviewing the submitted branch

1. **Constrained normal refinement keeps feasible intermediate iterates.** The submitted solver considered seeds and final `minimize.x` only. A feasible good iterate could be discarded when SLSQP exhausted32 iterations or finished on an infeasible trial. We now evaluate every objective trial with exact FK/original bounds, preserve the best feasible result, and independently recheck the result. q_anchor remains immutable. A failed optimizer status is still logged, never called convergence/global optimality.
2. **Coupled sliding-direction closure.** Fixed-point direction damping still oscillated at node9 in all three recorded18D replays. Added bounded deterministic least-squares in at most one angle per sliding contact, repeatedly solving the same coupled mass/contact/PD/joint-stop system. This solves direction self-consistency, not extra ground forces. Every final candidate still passes the original dynamic, friction, unilateral and complementarity gates. Outer max80 iterations and inner max32 evaluations are fixed; no parameter scan.
3. **Preserve historical path.** The review patch unconditionally changed half-step slip updates to quarter-step. Restored historical `.5` default and restricted `.25` plus coupled slip closure to the explicitly enabled Approach joint-stop path. Recovery and default measured-fixed path are untouched.
4. **Better paired replay evidence.** Log worst joint velocity, native counterpart, original limit, root error, contact forces and full solver trace on the first physical violation. Earlier records only named the worst positional joint, which could misidentify a velocity failure.

No speculative coordinate fix: the production per-time FK is already body-local and matches its transformed target. No frame conversions were changed.

## Actual CPU tests

- Submitted branch's requested six files: **74 passed in1.49s**, exit0 (`INITIAL_TESTS.txt`). This is a new run, not the prior89-test count.
- After changes, same targeted set: **77 passed in1.45s**, exit0.
- Full relevant regression suite: **101 passed in1.96s**, exit0 (`ALL_TESTS.txt`, `TEST_EXIT.json`). Includes prior action-history, Recovery exclusion, partial reset,2e-6 mapping and fullbody tests.
- New anisotropic-contact test demonstrates the old fixed-point solve fails within80 iterations while coupled direction closure converges in3 outer iterations and satisfies momentum, Coulomb friction, unilateral constraints and original residual tolerances. Synthetic evidence is not native safety evidence.
- New normal test preserves a feasible intermediate result even when the optimizer's terminal trial is infeasible. It retains original failure status and immutable input anchor.

## Same frozen inputs: geometry and velocity

Same recorded two lanes, same checkpoint/physics/3 original event offsets, same bounded planning times2.01/2.17/2.33/2.49 seconds;8 planning calls and24 contact candidates. No new random seeds or case-specific action tables.

`SAME_INPUT.json` independently reproduces the old0.3762389mm and new position-only anchor0.1681726mm under exactly the old selected time/geometry/frame. That anchor is retained, not overwritten.

| Fixed input | Contact position after refinement | Normal-vector error | Original requested speed error |
|---|---:|---:|---:|
| Lane0 at2.01s, relative contact0.79s | 0.499968mm | 0.713727→0.640840 | NOT_SOLVED; normal still fails |
| Lane1 at2.17s, relative contact0.63s | 0.499975mm | 0.060141 | 3.038216m/s |
| Lane0 at2.49s, relative contact0.29s | 0.493707mm | 0.000000629 | 2.486208m/s |
| Lane1 at2.49s, relative contact0.31s | 0.491879mm | 0.000000357 | 1.895202m/s |

The first row spends the authorized0.5mm refinement budget to improve normal; it is not a claim that final refined position remains0.168mm. Raw q_anchor still does. The three normal-feasible candidates pass exact original0.12 normal gate but fail original0.75m/s requested velocity gate, with several arm rates at existing±5rad/s bounds. No projection is relabeled as zero error.

| Stage | 8853ccc prior record | fd0e9d9 server replay | Final patch replay |
|---|---:|---:|---:|
| Position within3mm |13/24|13/24|13/24|
| Position + normal |0/24|2/24|3/24|
| Position + normal + requested speed |0/24|0/24|0/24|
| New candidate complete18D prediction calls |0|0|0|

The extra third pose comes from retaining a feasible intermediate iterate, not raising iteration count. All three failed speed candidates remain rejected. Analytic inverse-target computations are not robot-feasible returns. No native hit rate inferred.

## Three-command dynamics replay

All commands replay from the same recorded tick249 state and actual previous18D history. Recorded future commands are diagnostic inputs only, never online planning inputs.

- Review code: historical_actual18, frozen_base18 and frozen_legs_planned_arm18 all fail solver at node9 (10th interval).
- Coupled-slip change: all pass that interval, then all fail the **original joint-speed gate at node10 (11th interval)**.

| Source | FL_calf predicted speed | Native corresponding speed | Original cap |
|---|---:|---:|---:|
| Historical actual18 |−53.699682rad/s|−30.000004rad/s|30rad/s|
| Frozen base18 |−53.699182rad/s|−30.000004rad/s|30rad/s|
| New combined18D |−53.710274rad/s|−30.000004rad/s|30rad/s|

The native column is the saved historical trajectory for all rows, not a native counterfactual for the new combined commands. Only the historical-command row provides a same-command model comparison. Root position difference from that recorded baseline is about3.36/3.41/4.66mm respectively; differences for altered commands are not isolated model prediction errors. Dynamic residual maxima about1.3e−12 to3.1e−12, zero position/torque/friction violations, but velocity violation about23.7rad/s, so none is accepted. Native `quadsmash/tasks/stand/coupled_env.py` sets `velocity_limit_sim=30.0` for legs. This supports investigating missing/different simulation speed-limit enforcement in prediction; it does **not** prove that arbitrary velocity clipping or unlimited braking impulses are valid physical models. Those have not been added.

Original predictor's nine-step short prefix is not executed. Entire-prefix feasibility is still unproven. New numerical convergence is not equivalent to model agreement with PhysX.

## Timing and remaining gates

Final planning mean425.21ms, p95 470.28ms across8 calls (CPU perf_counter, OMP/BLAS=2). Extra exact constrained solving costs time. Per user clarification this is **not a failed offline timing gate**; keep timing for later optimization. Ordinary native inline planning remains disabled; no asynchronous/current-state execution claim is made.

Remaining: task-compatible reachable speed, simulator-consistent velocity/contact predictor, calibrated causal longer-horizon base forecast, complete physically accepted18D reference, then native execution/handoff validation. This round did not implement new WBC tuning or long-horizon base replacement, and did not claim raw .8s extrapolation is validated.

Native adoption, actual racket speed, PhysX string contact, net/landing, Recovery outcome and complete5 are all **NOT_RUN**. No training, SuperMimic changes or service/process manipulation. Protected source hashes remain identical to the prior teacher baseline (`PROTECTED_SHA256.json`). No EVALUATION_COMPLETE or native candidate manifest was emitted.

## Artifacts

`RESULTS.json`: per-candidate matched geometry/velocity comparisons. `REPLAY_REVIEW.json.gz` and `REPLAY_FINAL.json.gz`: full fixed-input replay. `PAIRED_REVIEW/COUPLED/DELIVERY.json.gz`: original and changed paired predictions, preserved independently. `SAME_INPUT.json`: historical geometry reproduction. `SOURCE_DIFF.patch.gz`: actual code/test diff from3308200 including imported review. `COMMANDS.md`: commands and exit codes.
