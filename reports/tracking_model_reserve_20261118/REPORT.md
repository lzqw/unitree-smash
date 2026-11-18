# Empirical tracking-model reserve in full-prefix joint refinement

Base 023c46cfe0a70755e41ccfa62390c7b8f3915a76. Offline teacher development, not a native evaluation. No training, SuperMimic operation, Recovery, physical-limit, asset or scorer change. No128-case run.

## Evidence-based change

Previous full steps repeatedly exceeded the original0.25rad tracking gate by approximately0.000076–0.000094rad, while independently checked half steps passed. This indicates local response error near an active constraint, not permission to round down violations.

The opt-in `--tracking-model-reserve` derives a local planning reserve from the immediately preceding accepted prediction/actual tracking difference, divided by accepted step scale squared. Here that is4.005491e-5/(0.5²)=0.000160219640rad. This is an empirical smooth-model heuristic, not a guaranteed bound, statistical confidence interval or native-model accuracy claim. Contact-mode changes can invalidate it.

Only local linear inequalities are tightened. Each side's reserve is capped at half the baseline distance to that side, keeping the unchanged verified baseline feasible. Physical bounds and the final nonlinear0.25rad test stay unchanged. Exact histories and the accepted response are checked before using the error estimate. NaN/mismatched responses and unsupported flags are rejected. Defaults remain unchanged.

## Same-event result

| Quantity | Baseline | New complete-step proposal |
|---|---:|---:|
| Position error | 149.904946mm | 119.511374mm |
| Normal-vector error | 0.329378624 | 0.296120171 |
| Requested racket velocity error | 2.983944701m/s | 2.973319680m/s |

The full step independently passed33/33 intervals and all three task residuals decreased. No backtracking was needed. Maximum tracking error:0.249900545617rad. Maximum local linear-vs-full CPU tracking error:0.000150450784rad (node30, arm index0), below this run's requested reserve. This one observation does not prove future coverage or PhysX equivalence.

Final3mm/0.12/0.75m/s gates all fail. The candidate is not a qualified teacher reference, is not installed or exported as one, and final ball-side collision success is not inferred. PhysX reference, native hit, Recovery and complete5 remain NOT_RUN. The analytic goal's existence is separate from realization by the robot.

## Validation

62 tests passed, exit0. New tests verify reserve provenance, exact signed-response consistency, finite/error checks, local-only tightening, zero-step feasibility and unchanged physical acceptance. Baseline q/v exactly reproduces the verified predecessor. Source/input/protected hashes were checked after execution.

26 fresh full-prefix model calls; 219.180160s; CPU exit0. Maximum CPU action mapping round-trip error is4.4408920985006262e-16rad. Native command auditing is NOT_RUN. Full raw commands, reference histories, local/actual response vectors, diagnostics, test logs and manifest are retained.

## Remaining work

Position and normal are improving but remain outside final contact gates, and the requested velocity residual is still about2.97m/s. Continue examining joint task reachability/actual forward output instead of interpreting weighted-cost reduction as hit success. As tracking headroom shrinks, preserve the original fallback and nonlinear validation; this heuristic must never justify accepting a known violation. Independent PhysX prediction calibration and actual contact/Recovery closure remain outstanding. No128-case evaluation is justified.
