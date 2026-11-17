# Audit

Baseline/worktree and scope: see REPORT.md. Pre-existing untracked assets/robots/piper_standard/configuration, attempts/, audit/ were not edited or staged. No process signals, service changes, GPU work or training commands were issued. SuperMimic PID snapshot is read-only; it is not a guarantee against external termination.

| Evidence | Source change | Verification |
|---|---|---|
| Coupled pose loss sacrificed position | new position_anchored_contact_search_v1 + opt-in search dispatch | same-input old git blob vs measured-seed anchor; independent event and nullspace protection tests |
| Nominal optimizer could overwrite per-time better q | bypass legacy coupled loop; each original time owns q_anchor | production branch two independent valid times with nominal failure |
| Projection errors could masquerade as task success | original request/achieved velocity retained, speed only after pose gate, strict actual impact | bounded speed and invalid attack tests; no native result inferred |
| Missing/nonfinite physical diagnostics could pass | evaluator fail closed | None/NaN/Inf/negative/missing tests + fullbody regressions |
| Recorded actual command predicts RR past native hard stop | opt-in coupled unilateral stop reaction | paired three-command replay + generalized momentum coupling unit test; subsequent nonconvergence preserved |
| Runtime exceeds100ms and long base predictor unvalidated | anchored native inline disabled; CPU replay explicit | runtime gating regression and TIMING.json |

Original Recovery, actuator mapping/limits, collision and scoring sources compared to30daad2 in PROTECTED_SHA256.json. Existing native observation and command files are read only. Recorded future commands are used only by the explicitly offline paired diagnostic. New controller search does not read case IDs, history outcomes or future native states.

The experimental stop reaction is implemented only in unilateral Approach predictor and is false by default. It has not passed full trajectory calibration. Full closed loop remains NOT_COMPLETED, with exact gaps in GATES.json and REPORT.md. No candidate/native completion marker is emitted.
