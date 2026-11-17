# Implemented scope

Baseline 910d6f2; independent existing teacher worktree; no reset or branch rollback.

- G0: new task_constrained_impact_goal_v2 maximizes normalized worst margin with SLSQP inequality constraints on actual ImpactOutcomeEvaluator output. Six margins: speed/5m/s, downward/1m/s, forward/5m/s, net/0.1m (also net width), legal court/1m, target/1m. Acceptance separately checks actual forward flags and nonnegative margins. Failed solves are initialization only, not valid goals.
- G1: downward height necessary condition using frozen center tape height + radius. Full net/drop comes from forward integration at actual crossing y; low-height rejection before pose/rate work. No height tolerance or asset changes.
- G2: task_constrained_hit_search_v2 jointly optimizes q6, qd6, UV2 in one authored strip, and worst-margin scalar. Contact time remains the original discrete three-event window per planning call. Outgoing velocity is computed from actual collision, not a free variable allowed to lie about the result. The target normal varies with the outgoing impulse direction; <=0.12 vector agreement and <=0.5mm position are still required. Task-valid snapshots survive solver termination, but never bypass Hermite/full18D/final-impact gates. No rate-only retarget in this mode.
- G3: output height correction necessary lower bound, with time/velocity/pitch changes explicitly unidentifiable rather than fabricated. No implemented jump command. Two arm-level successes mean this run does not establish a need to change takeoff yet.
- G4: independent from-reset historical command PhysX calibration, original two lanes and all800 ticks, persistent teacher-only service, no new candidate or policy inference. Separate analysis compares measured-state one-step versus accumulated CPU predictions (160 substeps) and keeps failures MODEL_UNCERTAIN.

New mode is opt-in; prior default control unchanged. CPU replay uses original input hashes and24 time candidates, no random sample increase or algorithm parameter sweep. It changes goal formulation, so its optimized points need not equal the old goal configurations; immutable per-event anchors remain recorded.

After the single fixed replay, only diagnostic refinements were applied: geometry early return for low-height states (none of the24 inputs), actual crossing/drop metadata, final-vs-original normal fields, and explicit fullbody rejection labels. Raw replay retains its original stale POSE_TASK_INCOMPATIBLE label on the lane1 candidate that subsequently passed joint search; CANDIDATE_STAGES.json corrects the summary from the saved physical result. No second replay or scoring-driven algorithm update was performed.

Final accepted-candidate ranking in v2 uses actual final task margin, not distance to the old ideal racket velocity. This branch was not exercised in the fixed replay (no fullbody pass); it is covered by production integration tests rather than a claim of native adoption.

Fullbody failure-state/solver trace retention was also added after the replay; no physics equations or admission tolerances changed. Native calibration ran the separately frozen calibration script, unaffected by planning/diagnostic edits. Its script hash is retained in its attempt manifest.
