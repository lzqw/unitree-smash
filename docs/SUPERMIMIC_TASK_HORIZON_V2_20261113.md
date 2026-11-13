# SuperMimic task-timebase v2 — code-only change

Base: `18f5bc563ac0110abd6031091719bca034bda6ec`.
No Isaac process, GPU training, or formal evaluation was started for this change.
No performance improvement or recovery success is claimed.

## Time horizons

The current environment controls at 0.01 s, while the authors' published control
rate is 50 Hz. Copying H=32 therefore shortens the physical imagination window
from 0.64 s to 0.32 s. The first proposed task configuration is:

| Item | Seconds | At 100 Hz |
|---|---:|---:|
| Physics integration | unchanged | unchanged |
| Policy interval | 0.01 | one action each tick |
| Model fitting | 0.16 | 16 transitions / 17 states |
| Differentiable policy rollout | 0.64 | 64 transitions |
| Whole native episode | 8.0 | 800 control ticks |

These are engineering starting values, not measured optimal durations. The
code also accepts 0.96, 1.28 and 1.60 s imagination windows, but does not enable
those automatically. Longer prediction is not a cure for wrong actions or an
inaccurate contact model. Do not change physics dt/decimation to lengthen a
learning window. Do not unroll all 8 s by default.

`Timebase` validates integral control-step counts and stores the effective
seconds/counts in the checkpoint. The default `physical_time` discount mode
uses gamma(dt)=0.99**(dt/0.02) and lambda(dt)=0.95**(dt/0.02); both are explicit
100 Hz task adaptations. `per_step` keeps numerical gamma=.99, lambda=.95.
Do not claim the two modes have identical physical discount horizons.

Beyond the differentiable rollout, the critic supplies continuation value.
The finite 8 s task end masks bootstrap, and later model-only steps receive no
reward or critic fit weight. True physical time is not model-predicted.

## Changed files and entry point

* `supermimic_timebase_v2.py`: seconds-to-steps validation; initial-state BPTT
  return; survival; GAE return; full-range action decoder.
* `supermimic_sequence_replay_v2.py`: device-resident time-major replay with
  exact-length windows, episode/session/ring-wrap guards. It never silently
  substitutes one-step fitting for a requested multi-step window.
* `supermimic_horizon_learner_v2.py`: multi-step fitting, BPTT with endpoint
  value, separate detached GAE critic update. Microbatches split the batch,
  not the time graph; actor steps once after accumulating the logical batch.
* `supermimic_env_v1.py`: residual-v4 commands go through the existing
  `FullRacketEnvV4._pre_physics_step` decoder. The legacy direct jump-event
  branch remains separate. Final joint targets are captured for auditing.
* `scripts/train_supermimic_horizon_v2.py`: new opt-in entry. Requires the
  existing residual PPO artifact; no random-policy evaluation, no MPC/BC,
  no compulsory full-panel/800-tick actor warmup. A complete real model window
  is fitted before policy updates start. Counters and resume timebase are saved.
* `scripts/audit_supermimic_event_horizon.py`: consumes saved episode times;
  reports takeoff-to-hit, hit-to-robot-touchdown, touchdown-to-stability
  quantiles and horizon coverage. Explicit field names are required so shuttle
  landing is never mistaken for robot touchdown.

The historical trainer and reports remain in the repository. For this revision
use the NEW entry, not `scripts/train_supermimic_v1.py`. Loading legacy replay or
optimizer state into this corrected action/time contract is intentionally
rejected. Initialization loads the ORIGINAL PPO actor, not a zero-score actor
trained with the incorrectly scaled execution interface.

## Algorithm reference and limitations

The BPTT start-return, soft-survival, endpoint bootstrap and GAE equations were
checked against `HybridRobotics/supermimic`,
`src/supermimic/learners/policy_learner.py`, blob
`f13731760f9d0937ad4c2f169ddf60be66c31e03`.
This is independently implemented task glue, not a verbatim full-paper port.

The current observation-delta/reward predictor and residual policy loader are
still imported from v1. That predictor is NOT the official physical-state
representation, analytic reward or sensor-history reconstruction. Full
leg-arm/contact state validation is still necessary. The uniform v2 sequence
ring is explicit; it does not claim to preserve the old rare-event archive.
Critic epochs/minibatches default to 5/4. The reward ceiling defaults to a
conservative task upper bound of 100 per step, not the G1 tracking ceiling.
The native collection exploration distribution remains that of the existing
actor wrapper. These are recorded differences, not concealed reproductions.

The new native zero-update comparison uses an independent original controller
and FINAL joint targets. It covers the first collected tick only. It is not a
replacement for a saved multi-phase native action trace audit or proof of
initialization success. Unit tests use a lightweight parent implementing the
old final-target formula and do not simulate PhysX.

The v1 evaluation recorder's missing head-contact/fault fields and post-reset
fallback snapshots remain outside this timebase change. Do not accept a formal
head-contact metric from that path before its recorder is corrected. No new
success numbers are reported here.

## Tests run in this revision

`PYTHONPATH=. pytest -q tests/test_supermimic_horizon_v2.py`

Result: 33 passed on CPU (PyTorch), plus Python compile checks.
Includes 32/64/128-step repeated BPTT+GAE updates on small test networks,
full-batch vs microbatch gradients, terminal/soft-survival handling,
reset/session/ring-wrap replay guards, action decoder delegation,
checkpoint/timebase rejection and missing event-time handling.
These tests do not establish native policy performance, GPU memory usage, or
physical world-model accuracy.

## Next server-side steps (not launched)

1. Read this diff without changing assets or the 128-lane evaluation protocol.
2. Verify final joint targets against the historical working controller across
   preparation, takeoff, swing and recovery, before tuning horizon lengths.
3. Use saved real event times to inspect physical durations. Also validate
   model errors at 0.16/0.32/0.64/1.28 s; duration coverage alone is insufficient.
4. After separate authorization, run the new entry on the 5080, retaining 128
   lanes for formal evaluation. Measure memory/throughput; do not change
   physics or truncate BPTT to fit VRAM. No training was started by this patch.
