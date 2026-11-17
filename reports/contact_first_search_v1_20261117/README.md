# Contact-first jump-and-hit search v1

This is a CPU-validated first implementation for the DTG Approach teacher. It is based on the time/state terminal constraint and dynamic-horizon idea in “Real-time Trajectory Optimization and Control for Ball Bumping with Quadruped Robots” (arXiv:2210.05195), adapted to the existing six-axis racket Jacobian and causal DTG intercept event.

## Implemented

- `quadsmash/control/contact_first_trajectory_search_v1.py`
  - deterministic contact-time offsets and contact-velocity scales;
  - ballistic plane-crossing helper for a low-cost event fallback;
  - DTG event-compatible target position/normal/velocity inputs;
  - moving-base velocity compensation, including base angular velocity crossed with racket offset;
  - six-joint damped task inverse for racket position and face normal;
  - optional bounded Gauss-Newton relinearization through the real six-joint
    forward-kinematics callback;
  - Hermite position/velocity trajectory to the contact state;
  - physical joint/velocity/step prechecks;
  - explicit terminal position/normal/velocity residual rejection;
  - Hermite intermediate-speed rejection and measured-arm-velocity seeding;
  - short post-contact command handoff to avoid an endpoint jump;
  - optional full-body evaluator callback, with contact-prefix acceptance only.
- `quadsmash/control/full_racket_rally_controller_v4.py`
  - opt-in `seed.contact_first_search` configuration;
  - per-lane plan state and partial-reset isolation;
  - optional arm-prefix overlay until the planned contact time;
  - explicit `set_contact_first_evaluator()` hook for a causal Pinocchio/WBC
    full-body candidate check;
  - exclusive per-lane evaluator-factory hook for fresh native-state replay;
  - no behavior change when the configuration block is absent.
- `scripts/contact_first_fullbody_evaluator_v1.py`
  - deterministic candidate replay from one measured `public794` state;
  - optional direct `LoadedPinocchioModelV1.PublicState` input for a future
    native-state adapter, avoiding observation-based root reconstruction;
  - `from_native(...)` constructor that reads the repository's live tensor
    contract and still performs CPU-only replay.
  - original 18-joint position bounds, velocity/torque/contact diagnostics,
    and authored non-foot ground-clearance witness;
  - measured-contact-only policy, so an airborne prefix cannot invent ground
    support; the evaluator is attached through the explicit controller hook.

The planner does not use future simulator state, TRAIN case IDs, learned action outputs, or landing success. The existing Recovery MPC and its 32-node protocol remain separate.

The upstream DTG source is recorded at commit
`1e625f1914848abe290dd624ecd031e4a75216e8` under its MIT license. Its causal
history, frozen predictor, predicted hit state and sigmoid timing are documented
in `UPSTREAM_DTG_EVIDENCE.json`; this worktree does not replace the current
training dependencies with that code.

## Current limits

This is a trajectory generator and Approach integration hook, not a completed native performance claim. The full-body predictor callback must be supplied for physical admission; without it the planner only performs kinematic and actuator prechecks. No native evaluation was started in this implementation turn.

The planner is intentionally deterministic and low-dimensional, but it is not a replacement for the existing full-body WBC or Recovery controller.  With a forward-kinematics callback it relinearizes at the proposed six-joint contact state; without one it applies the supplied Jacobian exactly once.  A plan with a large terminal task residual is rejected instead of being silently accepted.

The actual `FullRacketKinematicsV4` callback was exercised on CPU and accepted a
32-node plan (`KINEMATICS_CALLBACK_SMOKE.json`); this is a controller smoke only,
not an Isaac/native hit-rate result.

The combined planner plus `ContactFirstFullBodyEvaluatorV1` was also exercised
on CPU with 21 pre-contact nodes.  It accepted one deterministic candidate with
0.3556 m minimum authored ground gap and zero reported torque, joint-limit,
friction, or normal-force violations (`FULLBODY_EVALUATOR_CPU_SMOKE.json`).
This is a Pinocchio plus implicit-position-interface admission witness, not a
PhysX replay or a native execution certificate; native evaluation remains
`NOT_RUN`.

A 100-sample diagnostic sweep through the same full-body admission showed
25/25, 18/25, 8/25, and 2/25 accepted at 1, 2.5, 5, and 10 cm local target
offsets for a 0.30 s contact horizon (`FULLBODY_LOCAL_TASK_SWEEP.json`).  With
the same seed and targets but a 0.60 s horizon, acceptance was 25/25, 24/25,
16/25, and 8/25 (`FULLBODY_LOCAL_TASK_SWEEP_0p60.json`).  These are CPU
feasibility rates only.  The dominant rejection remains the original admitted
arm-target velocity bound, so the current bottleneck is available approach
time and whole-body coordination; the gate is not being relaxed to improve a
number.

The planner applies the inexpensive terminal task residual gate before calling
the full-body replay callback.  This keeps the same acceptance contract while
avoiding physical replay for candidates that already miss the causal
position/normal/velocity tolerances.

It also reports a strict necessary lower bound on contact time from the
selected joint displacement and original per-joint velocity limits.  If all
allowed timing offsets are below that bound, the search exits before candidate
generation with `contact_time_below_joint_speed_bound`; this is an explicit
unreachable-time diagnostic, not a relaxed or substituted trajectory.

With the nullspace branches enabled, 300 repeated 96-node plans with the real
kinematics callback took 5.38 ms mean (5.69 ms p95), with five
re-linearizations and five posture branches; see `CPU_BENCHMARK.json`.  This
is a planner timing measurement, not a real-time Isaac guarantee.  The extra
4 branch evaluations are deliberate: they trade CPU time for fewer
velocity-bound rejections.

A fixed 100-sample sweep of randomly selected reachable terminal poses accepted
12/100 with five re-linearizations before the nullspace branch was added; that
file is retained as a historical baseline.  The current full-body sweeps below
include the new branches and the original physical velocity gate.  Neither
sweep is a hit-rate claim.

For local DTG-like perturbations around the measured racket pose, acceptance was
100/100 for 1, 2.5 and 5 cm position offsets and 96/100 at 10 cm, with contact
times from 0.45 to 0.8 s.  This is a bounded kinematic feasibility result, not
an Isaac hit-rate result; see `DTG_LOCAL_TASK_SWEEP.json`.
