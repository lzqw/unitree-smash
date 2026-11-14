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
