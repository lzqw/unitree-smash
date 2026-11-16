# SuperMimic sequential R8/A8 evaluation

## Run identity

- Training run: `/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/refresh_full_r8a8_20261116_061227`
- Evaluation run: `/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/eval_refresh_full_r8a8_20261116_192406`
- Code commit at launch: `f4fc3bc`
- Candidate checkpoint: `candidate_0016.pt`
- Candidate SHA256: `fb74bd688f03a6137326036b28f63a27d14b976cf3548ca5f200794f279b2fa1`
- Source policy SHA256: `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- GPU: `cuda:0`, GPU PhysX enabled, 128 environments, seed `882072`
- Protocol: 800 control ticks / 8 seconds; no MPC, teacher, or video

## Training receipt

- Recovery accepted: `8/8`
- Approach accepted: `8/8`
- Proposal attempts: `16`; rejected/backtracked: `0/0`
- Refresh: `800` ticks, `102400` transitions
- Refresh world updates: `32`
- Refresh native-RTG critic updates: `20`
- Total world updates: `48`
- Total critic updates: `1022`

## Formal evaluation

| Metric | π0 | R8/A8 candidate | Change |
|---|---:|---:|---:|
| Native return | -12.52678435 | **-12.12484468** | **+0.40193967** |
| Qualified airborne string contact | 74/128 | 73/128 | -1 |
| Ball-side complete except recovery | 53/128 | 53/128 | 0 |
| Conditional recovery | 34/74 | 38/73 | +4 successes, changed denominator |
| complete5 | **27/128** | **25/128** | **-2** |
| Body-ground flag | 65/128 | 61/128 | -4 |
| Numerical failures | 0 | 0 | 0 |

Independent head/all-body contact labels were unavailable and remain `unknown`; they were not treated as false.

## Per-case changes versus π0

- Qualified contact lost: `B-720402-050`; new: none.
- Ball-side lost: `A-720401-017`, `A-720401-032`, `A-720401-057`.
- Ball-side new: `A-720401-026`, `A-720401-040`, `B-720402-018`.
- Recovery lost: `A-720401-025`, `B-720402-057`.
- Recovery new: `A-720401-011`, `A-720401-022`, `B-720402-001`, `B-720402-013`, `B-720402-022`, `B-720402-031`.
- complete5 lost: `A-720401-017`, `A-720401-025`, `A-720401-032`, `B-720402-057`.
- complete5 new: `B-720402-013`, `B-720402-031`.

The per-case native return delta has mean `+0.40193967`, with range `[-71.78083350, +71.23542513]`. The higher mean is driven by large gains on some episodes, while four complete successes are lost and only two are added.

## Decision

The candidate improves native return and preserves the 53/128 ball-side count, but fails the required complete5 gate (`25 < 27`). It is **rejected** and must not become `native_accepted_reference`; no larger training block is authorized from this candidate.

The full evaluation completed successfully with exit code `0` and 128/128 episodes. Raw metrics and episode summaries remain in the evaluation run directory above.
