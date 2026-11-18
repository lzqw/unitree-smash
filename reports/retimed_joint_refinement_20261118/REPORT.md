# Joint full-prefix refinement at the original 0.33-second event

This is offline development, not a native evaluation or qualified reference. No training, SuperMimic modification, physics/limit/scorer changes, or Recovery changes. No128-case run.

## Implementation

A new retimed-event adapter verifies the originating observation clock and input hashes, causally reconstructs the ball state, and restricts the event to the original frozen time window. It independently calls the forward ImpactOutcomeEvaluator on the proposed analytic goal and checks all task margins/flags. An accepted label alone, missing attack5 evidence, or a sub5m/s target cannot pass. The adapter transports the actual new target into the original fixed planning frame.

The complete hold-tail seed is reproduced from original q/v and previous actual final command. All seed q/v states match the saved prediction exactly. It then uses the existing24-parameter coupled leg/arm endpoint-rate correction with original command constraints, independent terminal-task nonregression, and fixed backtracking scales1,1/2,1/4. No IK endpoint is directly installed. Every response and proposal is freshly integrated from the original state with its own history.

## Same-event results

At0.33s, the unchanged seed has position error202.401244mm, normal-vector error0.334498436 and requested racket velocity error2.999898897m/s. The normal/velocity goal has been recomputed for this event; these errors cannot be directly compared against the old0.29s goal. The analytic goal passes the forward ball-side task, but this does not prove the robot can realize it.

All25 baseline/local derivative paths passed. All three new combined proposals failed:

| Step scale | First rejected node (zero based) | Arm tracking error |
|---|---:|---:|
| 1 | 14 | 0.255406076rad |
| 1/2 | 15 | 0.253507753rad |
| 1/4 | 16 | 0.252052551rad |

The original0.25rad path gate is unchanged. No failed prefix is accepted or executed. `improved=false`, selected_index=0. The original seed remains selected; position/normal/velocity and full task gates remain failed. PhysX reference, native adoption/hit, Recovery and complete5 are NOT_RUN. No teacher package is promoted or exported.

## Interpretation

The new terminal task inequalities prevent trading worse final normal/velocity for better position, but they do not yet constrain the predicted tracking error at every intermediate joint/time. The complete physical validator correctly catches that missing local-model constraint. The next justified improvement is to linearize signed per-node arm tracking responses from these same full-prefix derivatives, constrain each within the original±0.25rad envelope, then still independently validate the entire nonlinear sequence. Do not add smaller unbounded backtracking steps or relax the envelope.

## Evidence and limitations

53 tests passed, exit0. New tests verify changed-event target transport, source immutability, independent forward-task rejection despite accepted labels, nonunit normals, missing attack5 evidence and off-grid event rejection. Existing reference isolation, action bridge, bundle and optimizer checks pass.

28 fresh model calls; runtime220.880729s; CPU exit0. Source/input/protected hashes checked after completion. RUN.log, full trial trajectories, reference history, command audit and first violations are preserved. This is experimental CPU contact physics, not PhysX equivalence or realtime evidence.

RESULTS.metadata preserves original seed diagnostics for provenance; only the retimed fields listed in METADATA_SCOPE.json describe the new0.33s target. In particular, inherited source IK/attack_geometry/velocity-retarget labels are not current feasibility certificates. `retimed_forward_goal` and `retimed_task_margins` are the freshly checked analytic outcome; `robot_target_verified=false` remains explicit. A future metadata cleanup should nest inherited diagnostics rather than leave them alongside retimed fields.
