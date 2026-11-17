# Next round: contact-first trajectory search

## Current evidence

- CPU controller and full-body admission regression: 68 tests passed.
- The planner benchmark remains 300 plans, about 5.38 ms mean on the recorded
  CPU path; native timing is not claimed.
- The native development retry completed 2 environments × 800 control ticks,
  exit code 0, with zero numerical failures. It did not launch the ball and
  recorded zero contacts, so it is not a hit-rate result.
- The historical CPU full-body sweep was limited mainly by the application
  one-tick `admitted_target_bounds` envelope: at a 0.30 s horizon, acceptance
  was 8/25 at 5 cm and 2/25 at 10 cm; at 0.60 s it was 16/25 and 8/25.
  The contact-prefix evaluator now disables that extra envelope by default,
  while retaining the original physical position, velocity, torque, contact,
  and tracking checks.

## Ordered next actions

1. **Provide a real DTG/ball launch event to the native probe.** The current
   zero-contact result is an input/protocol limitation, not evidence that the
   planner cannot hit. Use the existing causal predictor and record planned
   contact time, actual contact time, racket position, normal, and relative
   velocity. Do not use future simulator state.
2. **Run one small native contact-only development check** with the same
   frozen candidate and per-lane evaluator factory. Measure candidate
   rejection reasons and first adopted command before considering a full
   panel. Do not include landing success in this check.
3. **Tune only the DTG contact event and Approach timing** after the native
   contact trace exists. The planner now ranks feasible candidates by terminal
   residual plus physical speed and one-tick step margin, so the next tuning
   variable should be contact-time/velocity prediction rather than a larger
   search grid.
4. **Keep Recovery and SuperMimic unchanged.** A native contact trace must
   show that the Approach handoff is continuous and that Recovery still owns
   the final action after its original trigger. No training, replay sharing,
   landing optimization, or second native candidate is justified by the
   current evidence.

## Unproven items

Native hit rate, contact timing error, racket-speed tracking, and stable
trajectory adoption remain unmeasured. The current artifacts must not be
reported as native success or as evidence of landing performance.
