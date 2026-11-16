# Teacher joint evaluation change plan

`evaluation_stage=teacher_debug`, `formal_eligible=false`, `formal_acceptance=false`,
`formal_cases_read=false`.

The fixed comparison is the existing 55/74 execution-consistent report at
`reports/execution_consistent_repair_result_20261110`. Its 52/74 predecessor,
55/74 corrected result, 128-lane seed, lane order, physics and scoring are read
as evidence only. No fixed-74 case is passed into a controller.

The single control change is in
`scripts/pure_control_state_driven_long_mpc_v1_qp_continuity_repair.py`:

1. Keep the normal 32-node state-driven plan unchanged.
2. Keep the existing per-node repair unchanged.
3. Only after both fail, run one deterministic bounded full-sequence search:
   five control nodes `(0, 8, 16, 24, 31)`, eight samples, one iteration,
   fixed seed `20261111`, all 18 absolute joint targets, and original
   `preview_recovery_execution` plus full authored physics checks at every node.
4. Accept a candidate only after all 32 nodes pass. Rank feasible candidates by
   separately normalized head and non-foot clearance, tilt, COM speed, angular
   speed, command continuity, and terminal support. If none passes, keep the
   55-version fallback.

The native entry is a fresh teacher-only wrapper. It uses the verified public
source factory and existing support/recovery controllers, disables the exported
residual actor and all learned recovery/approach heads, and records actor and
optimizer counters. The original Recovery arbiter, position limits, command
history and 2e-6 replay check remain in place.

No RL/PPO/BC/DAgger/distillation/critic update is run. No SuperMimic source,
checkpoint, replay or training process is touched. No physics, asset,
termination, scoring or task definition is changed. Existing failed direct
search attempts in the other baseline worktree are preserved and are not used
as this candidate's result.
