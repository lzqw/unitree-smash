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
