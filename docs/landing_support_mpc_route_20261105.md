# Landing support MPC route

The current best fixed policy is 27/128 complete5. Its recovery audit shows 26 ball-good recovery failures; all 26 already have a qualified foot landing, and 22 later register non-foot contact. The first non-foot event occurs 0.333–3.707 s after landing, while the phase774 recovery gate is active for at least 98.39% of the observed pre-event control steps. The immediate control problem is therefore sustained support after touchdown.

## Reusable implementation

The repository already contains a full loaded Go2 + PiPER + racket Pinocchio model and an OSQP whole-body support QP. `scripts/landing_support_mpc_v1.py` adds a risk-triggered receding-horizon wrapper around that QP:

```text
current public state
    ↓
risk trigger: tilt / root speed / angular speed / support count
    ↓
fixed-contact support QP for each 10 ms prediction stage
    ↓
Pinocchio state rollout over the short horizon
    ↓
terminal support-margin check
    ↓
first normalized joint-position target, or explicit rejection
```

The MPC keeps the existing position-target interface, joint limits, velocity limits, torque bounds, friction facets, contact mode and no-external-force contract. It does not add a force to Isaac, teleport a state, or silently average an infeasible plan with the PPO action. If a stage or terminal check fails, it returns an explicit rejection for the caller to handle.

The first implementation is intentionally conservative: it holds the observed contact mode over the short horizon and does not invent a new foot touchdown. It is a sequential shooting MPC over the verified support QP rather than a claim of full nonlinear whole-body MPC.

## CPU result on actual failure states

`scripts/benchmark_landing_support_mpc_v1.py` benchmarks the controller on the 26 real ball-good recovery failures in the 27/128 fixed-policy trace. The benchmark uses reconstructed public states and the authored loaded-body Pinocchio model; it does not dispatch commands.

For a 5-step (50 ms) horizon, all individual QP stages were feasible in the sampled landing windows, with about 6.6 ms median and 9.5 ms P95 wall time. The strict terminal support gate rejected the samples because the short fixed-contact rollout did not predict enough reduction in tilt/root speed. This means the current bottleneck is the prediction/terminal model and contact-mode coverage, not an OSQP infeasibility report.

## External implementations used as references

- [Reactive Landing Controller for Quadruped Robots](https://github.com/iit-dlslab/reactive_landing_controller) provides an open-source Go1 landing controller based on proprioception and a variable-height springy inverted-pendulum model. It targets touchdown foot placement and post-landing stability, but its ROS/Gazebo torque interface is different from this Isaac position-target Go2+PiPER setup. The paper reports a Go1-specific controller, so its code is a reference for the airborne foot-placement part rather than a drop-in module. [paper](https://arxiv.org/abs/2305.07748)
- [OCS2](https://github.com/leggedrobotics/ocs2) provides a mature switched-system legged MPC stack and example 1 s / 50 Hz configurations. It is C++/ROS-oriented and not installed in this environment; importing the whole stack now would create a separate runtime and interface rather than reuse the current Isaac controller.
- [Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and MPC](https://arxiv.org/abs/1909.06586) supports the split between centroidal/contact planning and whole-body execution. Its torque/WBC assumptions are different, so the current implementation keeps the original position-target execution path.

## Next native experiment

The first native experiment should run the MPC as a selective landing-support filter on a small set of post-touchdown states, with all commands passing through the existing residual mixer and original servo/limit path. It must record:

```text
student raw residual and log_prob
MPC status, iterations, runtime and rejection reason
MPC first target and final dispatched target
contact mode, support count, tilt/root speeds
first non-foot event and complete5 outcome
```

Promotion requires two fixed-seed evaluations with extra MPC disabled. The MPC-on result is execution assistance; the MPC-off result is the learning result. A solver rejection is an explicit fallback event, not evidence that the hardware is impossible.
