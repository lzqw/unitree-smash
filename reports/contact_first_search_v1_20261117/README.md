# Contact-first jump-and-hit search v1

This is a CPU-validated first implementation for the DTG Approach teacher. It is based on the time/state terminal constraint and dynamic-horizon idea in “Real-time Trajectory Optimization and Control for Ball Bumping with Quadruped Robots” (arXiv:2210.05195), adapted to the existing six-axis racket Jacobian and causal DTG intercept event.

## Implemented

- `quadsmash/control/contact_first_trajectory_search_v1.py`
  - deterministic contact-time offsets and contact-velocity scales;
  - ballistic plane-crossing helper for a low-cost event fallback;
  - DTG event-compatible target position/normal/velocity inputs;
  - moving-base velocity compensation, including base angular velocity crossed with racket offset;
  - six-joint damped task inverse for racket position and face normal;
  - Hermite position/velocity trajectory to the contact state;
  - physical joint/velocity/step prechecks;
  - optional full-body evaluator callback, with contact-prefix acceptance only.
- `quadsmash/control/full_racket_rally_controller_v4.py`
  - opt-in `seed.contact_first_search` configuration;
  - per-lane plan state and partial-reset isolation;
  - optional arm-prefix overlay until the planned contact time;
  - no behavior change when the configuration block is absent.

The planner does not use future simulator state, TRAIN case IDs, learned action outputs, or landing success. The existing Recovery MPC and its 32-node protocol remain separate.

## Current limits

This is a trajectory generator and Approach integration hook, not a completed native performance claim. The full-body predictor callback must be supplied for physical admission; without it the planner only performs kinematic and actuator prechecks. No native evaluation was started in this implementation turn.

The planner is intentionally deterministic and low-dimensional, but it is not a replacement for the existing full-body WBC or Recovery controller.
