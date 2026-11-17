# Bounded impact-rate validation — 2026-11-18

Worktree: `/home/autodrive/LZQW/unitree_smash_teacher_eval`.
Baseline `9df85ec`; imported review `edeabbf71f42e6e8b8f211f881c13f91fdaf0cf2` by fast-forward. Existing untracked assets/configuration, attempts and audit directories were not staged or changed. No SuperMimic task, weights or training was started or modified. No native evaluation ran.

## Actual changes

- Retained the review's optional task-valid bounded-rate search and final selected-plan export ordering.
- Fixed its second seed: a world outgoing vector cannot be transformed by a future-body-to-planning rotation. Use the actual planning-frame contact normal, transform it to future-body coordinates, then select bounded joint rates along that direction. Reject improper rotations. Preserve original requested velocity errors.
- Hardened the bundle writer itself: CPU-valid exports require final selected-plan/full18D/terminal-impact evidence and matching command sequences within the original 2e-6 rad tolerance. Callers cannot increase the final 3 mm threshold. The replay associates results by lane, plan clock and sequence, not just a matching contact index.
- Added per-task failure counts and best rejected-trial evidence. A legal return without attack5 may have no generic rejection_reason; it must remain visible and cannot authorize execution. These diagnostics were added AFTER the replay, without altering search or acceptance. RATE_TASK_AUDIT.json reconstructs these facts from the saved raw replay.

## Validation and gates

Imported six-file regression: 69 passed (exit 0). Intermediate expanded suite: 121 passed. Final suite including diagnostic regression: **122 passed**, exit 0, 2.16 s. `git diff --check` exit 0. See actual outputs, not previous-round test counts.

One fixed CPU replay, exit 0: 8 planning calls, 24 original time candidates. Input hashes and effective predictor contract are in RESULTS.json; raw evidence is losslessly compressed as REPLAY.json.gz with both hashes recorded.

| Gate | This run |
|---|---:|
| Position anchored | 14/24 |
| Position + normal feasible | 4/24 |
| Original/adapted velocity accepted | 0/24 |
| Complete analytic ball-side task | 0/24 |
| Full18D predictor calls | 0 |
| Exported references | 0 |
| PhysX, native adoption/hit, Recovery, complete5 | NOT_RUN |

The 4 rate searches made 1471 distinct trials (317/394/394/366), within 12127 overall impact calls. These are optimizer queries, not independent episodes or successes. 1411 queries predicted legal returns, 427 target landings, but **all 1471 failed attack5**. See RATE_TASK_AUDIT.json for per-component and per-event counts.

- Lane 1 at 2.17 s: 302/317 below 5 m/s; the other 15 fail downward velocity. Closest outgoing [4.11355,-2.09074,-1.05155] m/s, error 0.09905 m/s. Its desired outgoing [4.13897,-2.07604,-1.14615] is itself below 5 m/s: the reachable objective is not a task-valid certificate. Strict final gates correctly reject it.
- Lane 0 at 2.49 s, contact offsets .27/.29: all 394 trials each fail downward speed. Closest z velocities +.20274/+.03063 m/s; errors 1.38480/1.59007 m/s. Legal returns do not imply attacks or target landings.
- Lane 1 at 2.49 s, offset .31: all 366 fail downward speed; 336 also below 5 m/s. Closest z +.18995 m/s, error 1.56537 m/s.

These bounded local searches do NOT prove global infeasibility under the original joint limits. We did not widen the limits, add search sweeps, or promote a near-valid candidate.

## Limits and model status

No new candidate reached fullbody verification. Thus this run does not resolve the historical 160-substep contact convergence failure and gives no new fullbody/PhysX correspondence evidence. Historical model failures remain MODEL_UNCERTAIN, not proof of physical infeasibility. No native reference validator is claimed complete.

Original Recovery, actuator interface, collision, scoring, authored geometry and physics files match the previous protected SHA256 list (PROTECTED_SHA256.json). The additional 2e-6 test is export sequence matching; it is NOT a new native actuator round-trip measurement. There was no runtime adoption to audit.

Offline compute duration is not a rejection condition. Raw planning/rate timing remains in the replay. No online real-time claim is made. Full closure remains blocked at task-valid rate generation, before native gating.
