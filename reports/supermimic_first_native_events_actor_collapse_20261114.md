# First native task events and actor-collapse diagnosis

The broad-gate run reached the first complete native episode boundary at approximately iteration 800. Before any actor update, the accumulated native records contained:

- 520 terminated episodes, including 483 full timeouts;
- 76 qualified airborne string contacts;
- 13 ball-side-complete records excluding recovery;
- 242 recovery-success records;
- 6 `complete5` records;
- 165 frame hits and 429 body-ground observations.

This is a training-stream count, not a formal 128-case evaluation.

The first imagined actor update was unsafe: imagined return was about `-13.3`, critic loss about `432`, actor gradient norm about `18`, and native reward dropped from `+0.18` on the update step to approximately `-0.013` on the next step. Three actor updates were recorded before the run was stopped. The anchor penalty was effectively zero on the first update because the reference and initialized actor still matched; it did not compensate for the untrained critic gradient.

The learner now supports critic-only imagination updates and requires `200` critic optimizer updates before actor BPTT updates. This keeps the world-model and critic fitting path active while preventing the actor from following an uncalibrated value gradient.

