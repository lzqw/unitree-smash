# Sequential small-budget GPU validation

## Scope

- Code base: commit `6abbcd1` (parent `9ce88c7`), pushed to `feat/supermimic-native-physical-loop-20261114`.
- Source policy: verified SHA256 `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`.
- Mode: `sequential`, `shadow` ball-side diagnostics; no MPC, teacher, video, or parameter scan.
- GPU: RTX 5080, Isaac GPU PhysX, `cuda:0`, 128 environments, 800 control ticks, 8 seconds.
- Timebase: imagination 0.64 s, world window 0.16 s. Single-step command budgets remained 0.005 normalized and 0.04 rad.

## Training

Recovery and Approach were each capped at four proposals. All eight proposals were accepted: Recovery 4/4 and Approach 4/4; rejected 0, backtracked 0. The run produced 32 native control ticks and 4096 new transitions, 8 world-model updates, and 662 critic updates. Both heads had nonzero actor gradients and parameter changes. The intermediate and final checkpoints are retained in the run directory; hashes are in `checkpoint_sha256.txt`.

The run did not report complete-task success during the 32-tick training block; those ticks are training diagnostics only.

## Formal evaluation

The candidate `candidate_0008.pt` was evaluated once on the fixed 128-case panel. No optimizer updates occurred during evaluation.

| Metric | Result | Gate |
|---|---:|---:|
| Native undiscounted return | **-9.27615043** | > -12.52678435: pass |
| Qualified airborne string contact | **74/128 (57.8125%)** | reference 74/128 |
| Ball-side complete except recovery | **52/128 (40.625%)** | >=53/128: **fail** |
| Conditional recovery | **37/74 (50.00%)** | diagnostic |
| Complete5 | **29/128 (22.6563%)** | >=27/128: pass |
| Numerical failures | **0** | pass |
| Extra MPC interventions | **0** | pass |
| Extra teacher interventions | **0** | pass |
| Body-ground seen | **56** | contact audit field |
| Head/non-foot contact recovery labels | unknown | paired labels unavailable; not defaulted to false |

The candidate is **rejected by the predefined acceptance gate** because ball-side performance is 52/128, one case below the required 53/128, despite the improved return and complete5. No further evaluation or tuning was run.

## Interpretation

The result is a real native GPU evaluation, not an imagined score. It shows the bounded sequential update changed the policy while preserving qualified contact count and increasing the reported recovery count relative to the historical 34/74 reference. It did not satisfy the required ball-side protection gate, so it is not promoted as the next accepted starting policy.
