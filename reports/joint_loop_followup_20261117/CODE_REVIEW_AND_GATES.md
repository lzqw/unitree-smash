# Joint contact-loop follow-up (source-only patch)

Baseline: `44dc53453d2108c1062f7650a5cb5374e6386f08`.
Scope: **development patch only**. No Isaac native simulation, candidate evaluation,
SuperMimic modification, or learned-policy training was performed.

## Implemented

1. **Causal shuttle target double compensation.** The production controller's
   per-contact-time callback now maps the causally propagated world shuttle
   position directly into the fixed planning frame. The legacy
   `plan.position_m - nominal_target` displacement is still applied by the
   historical frozen-base control path but is **not** applied twice in the
   new causal shuttle-position branch. Arm FK still supplies the intended
   ball-center/contact geometry offset. A translated, rotated-frame test
   reproduces the old ~100.23 mm error as an explicit anti-regression.
2. **Actual jump records and original scoring.** The small native development
   probe now joins `FullRacketEnvV4.rally_records` with the chronologically first
   `JumpEnv.episode_records` per lane, as the original native evaluation
   does. Missing jump/rally records fail closed. It uses unchanged
   `summarize_training`, `independent_episode`, and `complete5_summary`,
   checks complete8 agreement, and only then writes its completed marker.
   It saves `complete5_breakdown.json` and independent scoring metadata.
   All results remain **TRAIN-development / formal_eligible=false**.

## Added tests

- `tests/test_contact_first_base_transport_v1.py`: causal time-varying
  shuttle-position invariant in a translated/rotated frame, including the
  historical 100.23 mm double-offset anti-example.
- `tests/test_probe_rally_export_v1.py`: real-first join contract,
  missing-jump rejection, duplicate-rally rejection, bad-case/missing-first
  rejection, and no fabricated record.

Suggested CPU commands on the original IsaacLab interpreter:

```bash
bash scripts/python.sh -m pytest -q \
    tests/test_contact_first_base_transport_v1.py \
    tests/test_probe_rally_export_v1.py \
    tests/test_contact_first_trajectory_search_v1.py \
    tests/test_joint_prefix_execution_v1.py \
    tests/test_impact_outcome_evaluator_v1.py
```

**The tests above have not been run by this review environment.** The
historical 107 CPU passes, 53 focused passes and 2-env/800-tick 0-contact
probe belong to baseline `44dc534`, not to the patched branch.

## Remaining technical gates (not silently modified)

- **No accepted native joint prefix yet.** The previous probe sampled
  2 first episodes, 0 contact, 0 joint-prefix adoption. Validate a
  *new, separately authorized frozen* small panel before 128; verify
  predicted target, actual selected 18D command, actuator dispatch,
  and 2e-6 round-trip on **adopted** commands.
- **Lane 1 reachability.** Its best prior task residual was approximately
  0.127 m. Re-evaluate from the corrected target before tuning search
  freedom or IK tolerances.
- **Geometry.** Prior near-ground `Head_lower` initial witness was
  approximately 0 m; one interval later it penetrated 1.19/1.23 mm in the
  CPU model. Root reconstruction was within sub-micrometre tolerance.
  Retain hard nonfoot-ground rejection. Improve takeoff prefix selection
  and validate full-body state/contact models instead of modifying geometry,
  friction or clearance thresholds.
- **Approximate impact.** The current analytic predictor uses modeled
  string-surface contact and uncalibrated impulse/drag, not measured
  PhysX. The IK terminal position tolerance (35 mm) and independent
  contact-position mismatch gate (3 mm) are intentionally distinct:
  these are not interchangeable proof of hitting. Diagnose each gate
  separately; do not increase the contact tolerance without physical data.
- **Joint takeoff optimization is not finished.** The causal 12D leg
  reference comes from a frozen walking/strike base extrapolated under
  predicted states, not a new optimized takeoff-force/flight trajectory.
  Next integrate low-dimensional takeoff-state variables with supported
  WBC/actuator commands, and validate the predicted future base and
  contacts against native traces.
- **Recovery remains separate.** Keep the verified 32-node Recovery MPC,
  its causal ownership, physical limits and command audit untouched.
  Do not claim a tested Recovery improvement from a no-contact probe.

No source hashes from this branch should be confused with the historical
frozen candidate. Run a new manifest and source hash before any new native test.
