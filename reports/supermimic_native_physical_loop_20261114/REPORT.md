# Physical SuperMimic integration — in progress

Base: 52624f09ae501c2f3640cbe4a96c9c001b37ff3b. No asset/reward/control-period changes.

## Native initialization capture (new evidence)

Original 128 lanes, seed 882072, 800 control ticks: 102400 native transitions.
Mean undiscounted return -12.52678434973663; 74/128 qualified contacts,
53/128 ball-side complete, 34/74 recovery, 27/128 complete5. No optimizer updates.
Independent head/all-nonfoot audit is unavailable, not zero.
All 800 reward/task-call replays verified; maximum reward error 7.45e-9.

The child completed simulation and cleanup but aborted during interpreter
finalization (GIL/PyThreadState, signal 6). Exit134 is retained. Data completeness
and clean process exit are reported separately; no repeat baseline was run.

## Physical adapter

Current actor -> cloned G07 execution -> final limited command -> PhysicalWorldV2
and ball response -> supervised task-memory transition -> once-only reward
increments -> physical794 observation -> contextual critic.
All future native commands/events are supervision only. NumPy interception and
historical NumPy request decisions use explicit stop-gradient; continuous Torch
base feedback, mixer, arm slew and physical inputs retain gradients.
This is a task adaptation, not a claim of an exact G1 SuperMimic reproduction.

The original and racket candidate models both completed64-step GPU gradient
checks through the same adapter. Approach mean-network gradients are finite,
nonzero; model parameters did not change. Model error is not subjected to the
2e-6 command-interface tolerance. Initial GPU command parity is exact.

A bounded512-update task-memory fit was necessary for the missing state path.
Its held-out recovery/body/cross event recognition remains weak. These metrics
are retained explicitly; aggregate loss is not task success or return evidence.

## Pending

Bounded256 actual actor updates, new native interaction, and one frozen128
comparison have not yet completed. No improvement claim is made. The initialization
capture is reusable; mixed-policy training returns must not be labeled frozen
candidate scores.

## Critic full-task sampling correction

The first 13 recoverable actor updates sampled imagined starts at native ticks 224/240/256/272 only. This did not expose the contextual critic to terminal success reward. Actor approach sampling remains unchanged; critic-only updates now use overlapping 64-step windows with current-state starts at 768,720,...,0, covering all 800 native ticks. The terminal window stops at the true 8-second boundary. No future archive labels enter these rollouts. Window length, losses, reward, learning rates and action permissions are unchanged. Forty related CPU tests passed.

Block 03 was deliberately interrupted at a boundary with 13 actor updates and 56 native ticks durably saved. Isaac cleanup subsequently segfaulted (exit 139); this is not a clean process exit. Resume uses the saved actor/model/critic/optimizers/replay/RNG and counters, rather than discarding updates. The separate failed block 02 performed 8 updates but left no recoverable policy weights; those compute costs remain separately recorded.

## Final finite physical-loop result

The bounded block resumed from the durable block-03 checkpoint at 13 actor updates and completed the fixed budget: 256 actor updates, 256 world-model updates, and 10,560 critic updates. Only the six `actor.approach.{0,2,4}` parameter tensors were trainable; base/recovery remained unchanged on every actor step. The native training monitor completed 130 mixed-policy episodes and ended at mean raw return -33.489 (mixed policy, not a frozen score), below the initialization native mean -12.52678435.

The single frozen 128-lane evaluation completed all 800 control ticks and wrote 128 terminal records. Isaac cleanup then aborted during interpreter finalization with SIGABRT/exit 134; `process_result.json` records this separately and the metrics are complete. Candidate SHA256 is b17416c68f7ceb69489c4b33bb8932959a289d3edf7b87039ed5538db0bd1898.

| metric | initialization | candidate | change |
|---|---:|---:|---:|
| raw native return mean | -12.526784 | -20.813214 | -8.286429 |
| qualified airborne string contact | 74/128 (57.81%) | 77/128 (60.16%) | +3 |
| ball-side complete except recovery | 53/128 (41.41%) | 39/128 (30.47%) | -14 |
| conditional recovery | 34/74 (45.95%) | 26/77 (33.77%) | denominator changed by measured contacts |
| complete5 | 27/128 (21.09%) | 17/128 (13.28%) | -10 |
| numerical failures | 0 | 0 | 0 |

The paired per-case audit has 54 positive return deltas and 74 negative deltas; 24 of the original 74 qualified-contact cases recover under the candidate. The candidate is therefore **not an improvement** and is rejected for deployment. It demonstrates that the physical loop and actor gradient chain execute, but approach-only updates destabilize the learned batting/task chain under this finite budget. The full return and paired case files are in `paired_initial_vs_candidate_0256/`.

Head and independent all-nonfoot contact labels remain unavailable in this environment and are reported as unknown, never as false. No online MPC or extra teacher was used.
