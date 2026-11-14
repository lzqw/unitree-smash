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
