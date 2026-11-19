# Contact-event data and WM calibration report

## Scope

This report covers the first control-boundary event dataset built from the frozen native archive and an actor-blocked GPU WM calibration. No Actor, Critic, contact-BPTT, or promotion update was run.

## Window choice

The dataset uses 4 control ticks before the event and 64 control ticks after it. With the native 10 ms control period this is 40 ms before impact and 640 ms after impact. Four pre-event ticks cover approach and action delay while retaining the same 64-step horizon used by imagined rollouts. The post-event window is long enough to measure immediate separation, ball flight, and short recovery context.

The event center is the transition that caused the latch. A state latch visible at `t+1` is indexed to transition `t`, so `state[t]`, `final_action[t]`, and `next_state[t]` describe one control interval.

## Data source and contents

Source packed archive:
`/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt`

The source contains 800 control ticks from 128 environments, or 102400 real PhysX transitions. The generated event dataset is:
`/home/autodrive/LZQW/experiments/supermimic_contact_event_v1_20261120_fixed`

It contains 1026 episode-safe windows split by episode/lane/event identity.

| Event class | Windows |
|---|---:|
| qualified_contact | 75 |
| frame_hit | 37 |
| crossed_net | 61 |
| landed_target | 55 |
| recovery | 37 |
| body_ground | 61 |
| fault | 16 |
| control-boundary negatives | 684 |

The negative windows are explicitly marked as control-boundary negatives. The packed archive does not contain complete PhysX substep traces, so these records are not claimed as substep-certified no-contact examples.

For qualified contact windows, the one-transition ball velocity change has mean 12.345 m/s, p95 14.249 m/s, and maximum 15.723 m/s. Frame-hit windows have mean 11.545 m/s, p95 13.965 m/s, and maximum 15.154 m/s. These are the high-value ball response samples in the current archive.

## WM calibration

Output:
`/home/autodrive/LZQW/experiments/supermimic_wm_calibration_20261124_v13_fast2`

The GPU calibration performed 32 actor-blocked WM optimizer updates. It used the fixed native archive as the initial anchor. A fast event-pool sampler was used for this calibration because the online mixed-policy path's episode-capped integer allocator was taking most of the wall time on the static archive. The online training path still retains the joint quota allocator.

- Actor updates: 0
- Critic updates: 0
- WM updates: 32
- Calibration wall time: about 109 s after Isaac startup
- GPU: RTX 5080, approximately 6.3 GB process memory during calibration
- Candidate checkpoint: `wm_calibration_candidate.pt`
- Candidate status: unverified and not promotable

The reported mean WM loss decreased from the first update's 86.512 to the final update's 145.392 on the last sampled batches, while the aggregate over the 32 updates was 202.889. Batch losses are highly heterogeneous, so this is diagnostic only. The candidate has not been promoted based on training loss.

An additional 32-window GPU validation compared the initial model and the
calibrated candidate on fixed validation-lane windows. The calibration run did
not exclude those lanes, so this comparison is diagnostic and cannot be used
as an independent holdout receipt. The candidate's state RMSE was 0.369 at one
step and 0.387 at 64 steps, compared with 0.296 and 0.359 for the initial
model. Its 64-step racket-position RMSE was 0.092 m compared with 0.018 m for
the initial model. The candidate is rejected for promotion and the initial WM
remains the anchor.

## Fixed native response recheck

The same four-lane, three-control-tick native probe was evaluated with the calibrated physical and ball models and the existing V13 action model.

| Metric | Before calibration | After calibration |
|---|---:|---:|
| Ball velocity response RMSE | 0.08539 m/s | 0.08511 m/s |
| Response direction cosine, lane 96 | -0.079 | -0.005 |
| lane 99 | -0.026 | 0.017 |
| lane 102 | -0.638 | 0.643 |
| lane 115 | 0.215 | -0.091 |

The RMSE improvement is approximately 0.3% on this four-lane diagnostic. Direction improved for one lane and remains unreliable for the others. The candidate therefore remains `FIT_COMPLETED_NOT_ADMITTED`; contact-aware Actor BPTT is still blocked.

## Engineering changes

- Legacy archives can now provide `--g07-execution-root` explicitly when the manifest predates that field.
- `wm_calibrate` skips unnecessary full G07 Critic-state construction.
- `wm_calibrate` emits progress records and uses the fast event-pool sampler.
- Added `supermimic-contact-event-v1` builder and episode-safe window tests.

The full repository test command still has unrelated failures caused by missing external Piper URDF/checkpoint artifacts and pre-existing dirty files. The new event dataset, impulse, and action-ball regression tests pass with 27 passed.

## Independent whole-lane calibration

The training entry now accepts `--wm-holdout-lanes` with explicit lane ranges.
The event builder also accepts a whole-lane split manifest and records its SHA256
in the dataset manifest. This prevents a validation lane from entering WM
calibration through a different event window.

The first independent calibration used lanes 0--95 for WM fitting and lanes
96--127 for validation. It completed 32 GPU WM updates in about 109 seconds,
with zero Actor and Critic updates. On the fixed 96--127 lane windows, the
initial anchor remained better than the candidate.

| Horizon | Initial ball velocity RMSE | Independent candidate |
|---|---:|---:|
| 1 step | 0.000318 m/s | 0.000416 m/s |
| 16 steps | 0.194889 m/s | 0.945072 m/s |
| 64 steps | 1.230170 m/s | 3.249692 m/s |

The candidate is rejected. These values are recorded as a strict lane-separated
diagnostic and do not authorize Actor training.

## Native probe collection

An event-aligned native probe collection is running on the 12 selected branches
from the first three specifications. The specs cover eight independent
validation lanes across qualified-contact and frame-hit edges. Each branch uses
the original 128-lane Isaac environment, the causal execution contract, and
the same packed-archive seed. The collector runs at most two Isaac children in
parallel and writes one immutable `native_probe.pt` per branch.

The collection is diagnostic only. A contact-validation receipt will be created
only after all branches complete and the measured nominal/minus/plus response,
substep timing, direction, and amplitude checks pass.
