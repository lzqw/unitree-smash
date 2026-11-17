# Action-conditioned base reference and conditional reach diagnosis

Base `fe3f876`; independent teacher worktree `/home/autodrive/LZQW/unitree_smash_teacher_eval`. This round reads the previous complete CPU prediction, not future native observations. No native simulation, new 128-case run, training, SuperMimic, Recovery or physics modification.

## Implemented

- `ActionConditionedBaseV1`: immutable base pose and twist at each original control node, derived from a causal predicted full18D prefix. Transforms root positions/rotations and body-frame twists into the initial planning frame. Validates complete node order, 25D configuration/24D velocity, finite/unit rotations, and identical logged18D commands. No out-of-horizon extrapolation or unverified interpolation. Context SHA256 binds initial q/v and the complete command sequence; changing either arm or legs invalidates it. Returned data are copies. This is an optimization reference, never execution authority.
- `refine_action_conditioned_contact_v1`: reads all complete-terminal-rejected predictions from the previous fixed panel, identifies their unique planning events by full command hash, reconstructs only the original current observation and re-solves contact position against the predicted base. It can run the existing coupled task optimizer when the original0.5mm position gate passes. A changed trajectory still requires a new Hermite/full18D/final-impact validation.
- `string_strip_reach_bound`: conservative necessary reach sphere from the actual serial-link offsets and original authored string rectangle. Uses the maximum tool-offset norm at rectangle corners and triangle inequality. Joint limits/collisions/dynamics are ignored, making the bound optimistic. Outside proves conditional geometric exclusion; inside never proves feasibility.

The default controller is not switched to this reference automatically. This offline module replaces coarse base transport in the explicit refinement entry only. The general online causal rollout worker and bounded revalidation loop remain unfinished; do not describe them as implemented.

## Evidence

Previous fixed-panel search had two analytic contact candidates but only one fully integrated29-step prefix. The offline refinement processes that one eligible prefix; it does not select a lane based on a success list, add time candidates, expand random searches or read future measured states. Lane is used only to locate the originating current observation.

| Quantity | Result |
|---|---:|
| Original terminal miss under complete18D |0.764064m|
| Same ball/time target in model-predicted future body |[0.784627,0.349596,1.137910]m|
| Position-only best residual with corrected base |0.244774m|
| Position solver |48 evaluations; not converged|
| Distance from fixed shoulder to target |1.246896m|
| Optimistic reach over entire selected authored string strip |1.159398m|
| Necessary base translation lower bound, fixed orientation/time |0.086998m including0.5mm position allowance|
| Joint task velocity refinement |NOT_RUN: fine position gate fails|
| New complete18D reference |NONE|
| PhysX / native hit / Recovery / complete5 |NOT_RUN|

The245mm numerical IK residual is not a proof of global impossibility. The independent triangle-inequality bound is: at this particular predicted base pose and contact time, no joint configuration and no point within the selected original strip can close the87mm envelope deficit. This does NOT rule out other base poses, contact times, other legal strips or physically generated takeoff trajectories.

The0.48371mm old anchor and0.244774m new anchor use different assumed future base states; do not call this loss of IK precision on the same target. The old precise solution relied on base transport not produced by the actual candidate commands.

The87mm quantity is a necessary translation magnitude if orientation/time/strip stay fixed, not a required vertical jump height, optimal takeoff velocity, executable correction, or a stability certificate. Minimum time, vertical-only, pitch and angular-velocity corrections remain UNKNOWN. No ideal base state is injected or treated as achieved.

## Verification and provenance

125 CPU regressions passed, exit0, including nonzero base rotations, body-twist transport, no mutation/cross-command reuse, out-of-grid rejection, original FK inside the conservative reach bound, strict fullbody/bundle/ownership and task gates. Raw log FINAL_TESTS.txt.

Initial conditional refinement exit0 (`RESULTS.json`). Adding the independent reach diagnostic and rerunning the identical single conditional calculation also exited0 (`RESULTS_WITH_BOUND.json`). Both outputs retained; same anchor residual and no algorithm/parameter sweep. Neither invocation is a new24-event search or native candidate evaluation.

Original source replay SHA, current-observation SHA and context command SHA are saved in the results. The command sequence is the previous causal predicted18D output, never recorded future native commands. Existing fullbody evidence remains rejected and unchanged.

## Next implementation

The evidence now supports moving to feasible base/takeoff planning instead of increasing wrist iterations for this event. Generate bounded preparatory/push-off modifications through existing WBC, simulate their actual18D targets, then re-solve contact against their action-conditioned base state. Preserve frozen baseline as fallback and unchanged Recovery priority. Do not specify a desired upward velocity as if it had been executed. Other contact times/strips remain separate bounded alternatives, not permission to relax the0.5mm/3mm/5rad/s/attack5 constraints.

A new complete trajectory must pass all original physical and final ball-task gates before any frozen native reference trial. Closure remains NOT_COMPLETED; this turn establishes a reusable causal transport interface and a concrete conditional reach obstruction.
