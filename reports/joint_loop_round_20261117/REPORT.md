# Joint-loop probe: completed simulation, control closure NOT achieved

Baseline 782375a. One frozen 2-environment development probe, indices 0 and 64, seed 882072, 800 ticks / 8 seconds. Started 2026-11-17 12:02:07 UTC; finished 12:21:20 UTC. Child exit 0, no signal termination. SuperMimic was neither modified nor launched. No training.

| Result | Count |
|---|---:|
| Native actual contacts / qualified airborne hits | 0/2 / 0/2 |
| Ball-side task / complete5 | 0/2 / 0/2 (necessary contact requirement fails) |
| Conditional recovery | 0/0, undefined rate |
| New joint-prefix adoption / Recovery adoption | 0 / 0 |
| Numerical failures | 0 |

**Scoring export limitation:** calling the unchanged complete5_summary on probe episodes fails with `KeyError: jump` (exit 1). The small-probe exporter lacks the full evaluation's jump record. No field was fabricated and no scorer was changed. Both complete5 values must be false because neither episode contacts the ball, but full independent scoring validation is NOT passed. The native completed.json certifies only this probe's simulation completion; it is not a full128 evaluation acceptance marker.

One planning event generated 54 candidates per lane. Lane 0: 47 task residual, 4 joint step limit, 3 ball/string position mismatch. Lane 1: 53 task residual, 1 unreachable velocity. No candidate reached accepted full-body replay or actual execution. Lane 0's mismatch was reproduced at 100.23 mm: the causal ball target still receives a second legacy nominal-target displacement. Source remained frozen; this defect was NOT fixed mid-run or followed by another candidate. See IMPACT_REJECTION_REPLAY.json.

Frozen base inference counts: walk 400, standing 800, strike 800; weights unchanged. Approach/Recovery learning heads and policy optimizer were not instantiated. Predictive cloned-base counts are empty because physical candidate evaluation was not reached. Do not label this a pure programmatic teacher or claim zero base-network inference.

1270 source/config hashes match the candidate manifest at completion. Candidate file SHA matches native reported SHA. Actual 2e-6 adopted-command audit is NOT_EXERCISED (zero adopted new commands), although CPU round-trip tests pass. Final focused CPU regression: 53 passed, exit 0; earlier broader output: 107 passed. Full128 evaluation: NOT_RUN.

The patch connects causal frozen-base leg generation, native-state 18D admission, finite-mass impact refinement, task-ranked search and exact prefix execution with Recovery priority. **It does not yet demonstrate a working native joint-hit loop.** Geometry investigation attributes historical first-step penetration to Head_lower starting almost on the ground, not a millimetre root reconstruction discrepancy. No constraints were loosened.

Next round first fixes duplicate target compensation and the probe's missing original scoring records; then verifies adopted commands on a separately authorized frozen candidate. No candidate expansion, tolerance relaxation or second native probe was performed. Historical 29/128, 20/29, 10/128 are not comparable performance estimates from these two cases.

Flags: evaluation_stage=teacher_debug; formal_eligible=false; formal_acceptance=false; formal_cases_read=false.
