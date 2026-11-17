# Offline reference generation policy revision

User clarification after commit8853ccc: full-body teacher trajectories are primarily intended as offline references for later RL imitation. Real-time performance is a later engineering objective.

This revision supersedes the **compute acceptance interpretation**, not the measured results, in REPORT.md/GATES.json:

- P95 below100ms is not a current acceptance prerequisite. Recorded mean374ms/P95 421ms remains valid performance evidence, not a reason to reject a physically valid offline trajectory.
- CPU search currently has no100ms cancellation/deadline. No application timeout needed removal in the inspected planner/runner. Bounded optimizer iterations are algorithmic configuration, not a wall-clock limit; they remain unchanged pending separate evidence-based changes.
- Keep dt=.01s and original trajectory/episode time semantics. Slower wall-clock generation must not change simulated physics or consume extra simulated control ticks.
- The explicit offline runner may spend longer generating and validating each reference. Full18D dynamics, original joint/torque/contact/geometry constraints, final3mm contact and actual-command2e-6 checks remain mandatory. Partial or invalid sequences are not valid imitation references.
- The ordinary native action loop still does not silently run this offline planner. Removing that guard is a separate execution integration change; it is not required to relax offline compute acceptance.
- The current fixed-input pose failure and subsequent contact-solver nonconvergence remain blockers. Relaxing computation time does not make these trajectories physically admissible.
- Continue recording time, solver status, rejected candidates, inputs and command history. Preserve numerical/resource/process protections.
- No RL/PPO/BC/distillation or SuperMimic action is authorized or started by this policy change. No128 evaluation is started.

Priority now: simultaneous position/normal/velocity feasibility and stable full-body prediction, then validated reference generation/execution. Optimize wall-clock performance later. Historical reports remain immutable evidence of the earlier acceptance policy.
