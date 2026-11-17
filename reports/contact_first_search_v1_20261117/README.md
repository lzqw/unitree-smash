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
  - no behavior change when the configuration block is absent.

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

On the same CPU path, 100 repeated 96-node plans with the real kinematics
callback took 2.09 ms mean (2.11 ms p95) with five re-linearizations; see
`CPU_BENCHMARK.json`.  This is a planner timing measurement, not a real-time
Isaac guarantee.

A fixed 100-sample sweep of randomly selected reachable terminal poses accepted
12/100 with five re-linearizations, versus 6/100 with the previous three.  The
remaining candidates were rejected by terminal task residual or joint-step
limits; the sweep is a diagnostics fixture, not a hit-rate claim.  Details are
in `REACHABLE_TASK_SWEEP.json`.

For local DTG-like perturbations around the measured racket pose, acceptance was
100/100 for 1, 2.5 and 5 cm position offsets and 96/100 at 10 cm, with contact
times from 0.45 to 0.8 s.  This is a bounded kinematic feasibility result, not
an Isaac hit-rate result; see `DTG_LOCAL_TASK_SWEEP.json`.
