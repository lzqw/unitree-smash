# Causal push-off task through the existing implicit-drive WBC

Base `703e5d2`. Independent teacher worktree only. No native run, 128-case evaluation, training, formal cases, SuperMimic changes, or Recovery modifications. Original shared QP implementations, physical limits, scoring and assets are unchanged.

## Implemented control path

`PushOffSupportQPV1` adds an explicit PUSH_OFF task on the repository's existing `ImplicitCommandSupportQP`. It uses the original full-command position interface, preserves the supplied six arm commands, and solves for leg targets with the full loaded robot dynamics. It does not apply an upright/standing servo during push-off. Its COM/angular/joint baseline comes from predicting the original current command from the same current state; a requested COM acceleration increment is a cost, never an injected state or external base force.

The full18 command has original physical position bounds. Actual velocity, torque, contact and geometry remain checked in nonlinear160-substep prediction. No additional target slew or measured-q envelope is introduced. Airborne/other-phase/Recovery calls return existing-controller ownership with no QP proposal. Each candidate/controller owns its mutable model and command history.

`rollout_push_off_prefix_v1` tests one predetermined support window: first8 ticks of the existing29-tick rejected prefix. A rejected QP leaves the original command intact. Accepted proposals are independently replayed through the authored-sphere, original stop/speed-response model. All commands, states, per-node QP results, physical diagnostics, arm tracking and normalized-action roundtrip are retained. Changed commands are not blended after verification.

The request derives from the previous conditional87mm reach deficit: COM acceleration increment [1.385885,0.142683,4.120765]m/s² over an assumed80ms support window. Translating a root reach deficit into a COM task is only an optimization seed. The remaining movement must be produced by actual joint commands; this is not proof of an optimal or sufficient takeoff state. No parameter sweep was performed.

## Interface issue discovered and corrected

The first prototype used the older instantaneous-torque support formulation. Its prescribed arm torque plus fixed-contact constraints were infeasible, including with zero requested COM increment. All8 support-window calls rejected. Nonlinear prediction of the original command was nevertheless valid and established the fourth contact at microstep1. Therefore infeasibility did not establish that the desired task was physically impossible. Contact differences are evidence, not a uniquely proven root cause.

Final code uses the existing backward-Euler drive command mapping: the arm equality applies to its position target, not an assumed constant initial PD torque. This is a model/interface correction with the same request, settings and physical bounds, not gain tuning. It still assumes unsaturated fixed contacts within the QP, so all proposals need independent nonlinear validation. The initial source and all outputs are retained; the final code is not the instantaneous prototype.

## Actual CPU outcome

| Quantity | Original-command fallback | Final implicit-drive push-off |
|---|---:|---:|
| QP proposals accepted |0/8|7/8|
| Full nonlinear intervals completed |29/29|29/29|
| Path constraints |pass|pass|
| Terminal contact position error |0.764064m|0.598398m|
| Terminal normal-vector error |0.403364|0.261512|
| Terminal task-adapted velocity error |4.884297m/s|4.431020m/s|
| Terminal root height |0.487309m|0.503568m|
| Complete task/impact admission |FAIL|FAIL|

Node6 QP is infeasible and uses the original reference. The other7 changes reach the CPU nonlinear predictor as actual final commands. This is **CPU adoption**, NOT native actuator adoption. Maximum full18 mapping roundtrip error is4.44e-16rad versus the unchanged2e-6 gate. Maximum arm tracking error is0.237825rad, below the existing0.25rad path gate. Full nonlinear rollout takes11.29s on this run; offline only.

The final task is still rejected before impact evaluation. Position3mm, normal0.12 and velocity0.75m/s gates are unchanged. FULLBODY_PREDICTION_VALID=false, valid_teacher_reference=false. PhysX/native hit/Recovery/complete5 are NOT_RUN. No complete or valid-native marker exists.

A conservative post-rollout reach calculation leaves a conditional57.879mm translation deficit at this new base pose/time/strip, versus86.998mm before. The same optimistic arm reach bound1.159398m remains; the new target shoulder distance is1.217777m. This is still outside the geometric envelope even ignoring joint limits. It does not certify a minimum vertical jump height or prove that another time/pose/strip cannot work.

## Validation and evidence

Final25 tests passed, exit0: phase authority, original full-command bounds, exact prescribed arm target under the implicit map, independent dynamics/PD residuals, existing admitted-arm and implicit-QP regressions, causal base transport and reach bounds. The synthetic feasible-QP test is an interface check, not a native task success.

RESULTS/DIAGNOSED retain the initial instantaneous formulation and zero-increment audit. PREFIX is the initial path replay. PREFIX_STRICT adds explicit original0.25rad arm tracking and terminal/forward-impact gates and leaves the same fallback trajectory. PREFIX_IMPLICIT is the final mapped-drive result. All invocations exit0; experiment rejection is stored separately from process exit status. No native restarts or score-driven reruns occurred.

## Remaining work

1. Use phase/contact-based push timing and action-conditioned base transport to propose the next bounded takeoff correction; the present80ms task is not enough to close the remaining reach deficit.
2. Re-solve the contact task only when the resulting base trajectory places the ball inside the legitimate arm envelope. Do not continue a fixed-face rate search at an unreachable position.
3. Improve WBC's within-interval contact/drive consistency; node6 rejection remains. Keep nonlinear rejection and original-command fallback.
4. Revalidate every complete revised sequence and final forward ball task before considering an isolated native reference. This round has no candidate eligible for native evaluation.
