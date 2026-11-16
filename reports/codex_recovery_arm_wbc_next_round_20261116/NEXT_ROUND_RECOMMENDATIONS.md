# Next Round Recommendations

This round is stopped. These are recommendations only.

1. **Arm contact timing and reference shaping (highest priority).**
   - Evidence: contact point speed maximum rose from 2.689 to 3.011 m/s, but airborne hits fell from 29/128 to 28/128 and complete5 fell from 10/128 to 6/128.
   - Hypothesis: removing software slew exposed faster commands without sufficiently improving planned contact timing/face alignment; the arm speed change alone did not produce better task outcomes.
   - Suggested change: tune the DTG/Approach contact-time, racket-normal, and nonzero velocity reference together, with explicit Jacobian-conditioned joint participation.
   - Required verification: CPU reference-boundary tests, one frozen native candidate, and contact timing/normal/velocity telemetry before judging success.

2. **Recovery direct-search feasibility and objective.**
   - Evidence: 897 search calls, 7176 samples, 8 full-feasible candidates, 2 adopted 32-node sequences; 7018 physical violations and 158 prediction failures. The search was actually adopted, but aggregate recovery and complete5 did not improve.
   - Hypothesis: the feasible region is sparse and the current one-iteration sample set does not reliably optimize support/momentum quality after the history fix.
   - Suggested change: improve bounded sequence parameterization and normalized support/momentum objective while preserving 32 nodes, physical admission, and current fallback semantics.
   - Required verification: prove full-horizon feasibility and first-command execution consistency offline before any native run.

3. **Approach/WBC handoff.**
   - Evidence: command ownership and `2e-6` audit pass, while H and complete5 both regress; raw recovery-success episode flags are not identical to the metrics' stricter conditional recovery count.
   - Hypothesis: handoff is numerically continuous but not task-optimal at the contact-to-flight boundary.
   - Suggested change: inspect phase transition timing, base compensation, and support-QP admission around the first contact; keep Recovery sole owner after takeover.
   - Required verification: per-case takeover/contact alignment and an unchanged command audit.

4. **Do not change yet.**
   - Physics, assets, actuator limits, scoring, SuperMimic, or learned action weights have no evidence of being the cause in this round. Keep them frozen until the teacher-side evidence above is improved.
