# Conditional task-goal projection

This change tests a specific failure mechanism: maximizing analytic task margin
can demand a racket state farther from the full-body response than necessary.
It does not change or execute the 18D command sequence. No native evaluation,
training, SuperMimic operation or 128-case evaluation was performed.

## Implementation

`project_task_goal` jointly varies target normal (two tangent coordinates) and
world racket velocity. It minimizes normal/velocity distance to the complete
CPU-predicted terminal state, with independent non-regression constraints on
both errors. Every candidate uses the actual forward ImpactOutcomeEvaluator.
All six normalized speed, downward, forward, net, court and target margins must
be nonnegative, and all four forward contact/task flags must be true. A seed
failing the actual forward task is rejected. Solver success alone never grants
acceptance; the selected goal receives an uncached final forward check.

One deterministic SLSQP solve, maximum 32 iterations, no random restarts or
parameter scan. Normal chart coordinates are bounded to [-2,2], analytic
velocity targets to [-40,40] m/s. These are goal-space domains, NOT robot
actuator limits or executable commands. The original 5 rad/s arm limit remains
unchanged. No robot feasibility certificate is emitted.

The replay verifies source and ancestor hashes, complete trajectory and signed
arm tracking, and terminal FK consistency. It uses only a causal CPU future
state for this offline diagnostic. It does not consume future native states.

## Frozen input and results

Source: `reports/tracking_model_reserve_20261118/RESULTS.json`, the default
global-rate basis, 0.33 s event, complete 33-interval model prefix. The optional
late-rate experiment was deliberately not promoted or used as the baseline.

| Quantity | Original goal | Projected legal goal |
|---|---:|---:|
| Normal vector distance to predicted racket | 0.2961201714 | 0.2254648269 |
| Velocity distance to predicted racket (m/s) | 2.973319680 | 2.779230081 |
| Minimum normalized forward task margin | 0.145876840 | 0.115077269 |
| Predicted outgoing velocity (m/s) | [6.235692, -0.224149, -1.145877] | [6.155050, -0.486052, -1.148205] |

Both conditional analytic goals satisfy the original attack5 and landing
requirements. The new goal sacrifices some positive task margin for closer
robot-state matching. Runtime: 0.689 s, 73 distinct forward queries, solver
success. CPU replay exit 0; 30 regression tests passed, exit 0.

**The actual predicted robot state did not improve.** Its position error remains
119.511374 mm, beyond the final 3 mm gate. Even against the new goal, normal
0.225465 exceeds 0.12 and speed error 2.779230 exceeds 0.75 m/s. The analytic
impact assumes ideal contact at the causal ball location; it is not an impact
at the actual displaced racket. No target is installed. The exact-target
continuation contract was not bypassed.

## Gates and next work

- Conditional ANALYTIC_GOAL_VALID: true, old and projected goals.
- Actual terminal POSITION_FEASIBLE / NORMAL_FEASIBLE / velocity tracking: false.
- Complete 33-node model path: previously passed; no new dynamics run this turn.
- FULLBODY_PREDICTION_VALID for the complete hitting task: false.
- PHYSX_REFERENCE_VALID / NATIVE_HIT / RECOVERY_SUCCESS / COMPLETE5: NOT_RUN.
- CONTROL_IMPROVED: false; only conditional target matching improved.

The result supports incorporating explicit goal adaptation into a future
full-prefix optimization, not replacing a target silently. Such integration
must keep original state, history, event, ball prediction and physics hashes;
independently validate the new goal; recompute the baseline residual under it;
and fully validate every proposed 18D sequence. The remaining 119.5 mm position
and 2.78 m/s speed deficits still require actual whole-body command changes.
Do not increase fixed-face rate sampling or treat this goal projection as a
successful teacher reference. PhysX calibration remains outstanding.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
