# Contact-first jump-and-hit search v1

This is a CPU-validated first implementation for the DTG Approach teacher. It is based on the time/state terminal constraint and dynamic-horizon idea in “Real-time Trajectory Optimization and Control for Ball Bumping with Quadruped Robots” (arXiv:2210.05195), adapted to the existing six-axis racket Jacobian and causal DTG intercept event.

## Implemented

- `quadsmash/control/contact_first_trajectory_search_v1.py`
  - deterministic contact-time offsets and contact-velocity scales;
  - ballistic plane-crossing helper for a low-cost event fallback;
  - DTG event-compatible target position/normal/velocity inputs;
  - optional causal `target_state_fn(duration_s)` so each contact-time
    candidate re-propagates the shuttle position and velocity instead of
    reusing one fixed intercept point;
  - moving-base velocity compensation, including base angular velocity crossed with racket offset;
  - optional causal base linear-acceleration extrapolation to the planned
    contact time, defaulting to zero for the historical path;
  - optional causal base angular-acceleration extrapolation for the racket
    offset term, also defaulting to zero;
  - bounded tangent-plane face search around the requested normal;
  - position-priority refinement followed by an explicit adaptive face target
    bounded by `adaptive_normal_target_max_deviation_rad`;
  - explicit nonzero contact-speed gate and reachable-velocity projection,
    with the requested and selected velocities kept separately in metadata;
  - six-joint damped task inverse for racket position and face normal;
  - deterministic position-priority IK branch for cases where the full
    position-plus-normal solve stalls near a wrist limit; normal residual
    remains a hard admission condition;
  - bounded base acceleration prediction with raw-versus-used diagnostics;
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
  - contact-prefix mode does not apply the extra one-tick
    `admitted_target_bounds` software envelope by default; the historical
    envelope remains an explicit compatibility option.  Original physical
    joint velocity, position, torque, contact and tracking checks remain
    active.
  - measured-contact-only policy, so an airborne prefix cannot invent ground
    support; the evaluator is attached through the explicit controller hook.

The planner does not use future simulator state, TRAIN case IDs, learned action outputs, or landing success. The existing Recovery MPC and its 32-node protocol remain separate.

The upstream DTG source is recorded at commit
`1e625f1914848abe290dd624ecd031e4a75216e8` under its MIT license. Its causal
history, frozen predictor, predicted hit state and sigmoid timing are documented
in `UPSTREAM_DTG_EVIDENCE.json`; this worktree does not replace the current
training dependencies with that code.

## Current limits

This is a trajectory generator and Approach integration hook, not a completed native performance claim. The full-body predictor callback must be supplied for physical admission; without it the planner only performs kinematic and actuator prechecks. A 2-environment native development probe exists, but formal native evaluation remains `NOT_RUN`.

The planner is intentionally deterministic and low-dimensional, but it is not a replacement for the existing full-body WBC or Recovery controller.  With a forward-kinematics callback it relinearizes at the proposed six-joint contact state; without one it applies the supplied Jacobian exactly once.  A plan with a large terminal task residual is rejected instead of being silently accepted.

The controller now supplies that callback from the locked drag model using the
current causal ball state.  Each time offset gets a separately propagated
ball position and velocity, while repeated queries at the same quantized
candidate time are cached within one plan.  The callback remains deterministic
and does not read future simulator observations.

The controller now also supplies a causal `base_state_fn(duration_s)` for the
contact-first path.  It propagates the current body-frame translation and
linear/angular velocity with bounded acceleration and uses a Rodrigues
rotation to map future-base racket targets back into the current planning
frame.  Each timing candidate therefore solves its own ball state and future
base pose.  The callback is opt-in through the existing contact-first seed;
without that seed the historical controller path is unchanged.

For a reachable contact whose requested racket velocity is outside the six-axis
Jacobian/velocity bounds, the planner may project the event velocity onto the
actually reachable command.  Projection requires a nonzero absolute racket
speed and a minimum alignment with the requested relative velocity.  The
requested velocity, selected velocity, projection flag, and speed are all
audited; a zero-speed candidate is rejected.

The position-priority branch can select the final position-refined racket face
as the event normal when its angular deviation from the causal requested face
is within the configured bound.  The requested, pre-projection, and selected
normals are all recorded.  This is a contact-first feasibility adaptation; it
does not claim that the selected face gives the desired outgoing shuttle
trajectory.

`ContactFirstFullBodyEvaluatorV1` also accepts an optional
`racket_measure_fn(state)` hook.  When supplied, terminal position, face
normal, and velocity are checked on the predicted actual full-body state after
implicit-PD replay; a good joint trajectory alone is no longer enough for
admission.

`BASE_POSE_CPU_REPLAY.json` is a read-only replay of the previously saved
800-tick, two-lane public observations.  It confirms the updated controller
path reaches the planner and records the current per-lane residuals and
rejection counts.  One lane now passes CPU kinematic admission and the other
remains rejected by the face task residual; this is development evidence and
is not a native contact result.

After the CPU checks, a native **development probe** was run with the same controller path for 2 environments and 800 control ticks.  The first attempt was preserved and failed at the CUDA-to-NumPy boundary when passing causal base acceleration into the CPU planner.  The explicit `.detach().cpu().numpy()` bridge fixed that engineering fault; the retry exited 0 and completed all 800 ticks with zero numerical failures.  Correction from raw evidence: this probe did launch the ball (one launch per lane), recorded zero contacts and zero completes. It is a two-lane development failure, not a full-panel result. See `LAUNCH_EVIDENCE_CORRECTION.json`; the previous no-launch statement was incorrect.  Evidence is in `native_probe_2env_800tick/`; formal native evaluation remains `NOT_RUN`.

The retry's per-lane planner log showed a more substantive issue: all 45
contact candidates were rejected by `kinematic_task_residual`.  At the
planning tick the desired body-frame target was about 0.43--0.47 m above the
current racket point, while the base was entering a jump.  The planner had
been compensating base velocity for contact speed but not base translation
for the terminal position, so it incorrectly required the six arm joints to
produce the entire jump displacement.  The current patch subtracts a causal
`v*t + 0.5*a*t^2` base translation from the terminal arm target before IK and
records that displacement in candidate metadata.  Native confirmation of this
fix is still pending; the pre-patch diagnostic is preserved under
`native_probe_2env_800tick/retry2_base_compensation/`.

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
16/25, and 8/25 (`FULLBODY_LOCAL_TASK_SWEEP_0p60.json`).  Those historical
sweeps used the one-tick envelope and are retained as a baseline.  The current
contact-prefix evaluator no longer applies that application-level envelope by
default; a fresh native contact run is still required to measure the resulting
physical hit rate.  These are CPU feasibility rates only.

The planner applies the inexpensive terminal task residual gate before calling
the full-body replay callback.  This keeps the same acceptance contract while
avoiding physical replay for candidates that already miss the causal
position/normal/velocity tolerances.

When the opt-in planner is used by `FullRacketRallyControllerV4`, base linear
and angular accelerations are estimated per lane from consecutive public
observations.  The first tick after reset has no history and uses zero;
partial reset clears only that lane's history.

It also reports a strict necessary lower bound on contact time from the
selected joint displacement and original per-joint velocity limits.  If all
allowed timing offsets are below that bound, the search exits before candidate
generation with `contact_time_below_joint_speed_bound`; this is an explicit
unreachable-time diagnostic, not a relaxed or substituted trajectory.

Accepted candidates now carry peak joint-speed ratio, peak single-tick step
ratio, and terminal joint displacement in their metadata.  These values are
diagnostics for saturation and continuity; they do not replace the original
execution limits or alter candidate ranking.

With the nullspace branches enabled, 300 repeated 96-node plans with the real
kinematics callback took 5.38 ms mean (5.69 ms p95), with five
re-linearizations and five posture branches; see `CPU_BENCHMARK.json`.  This
is a planner timing measurement, not a real-time Isaac guarantee.  The extra
4 branch evaluations are deliberate: they trade CPU time for fewer
velocity-bound rejections.

`CPU_BENCHMARK_BASE_POSE.json` records a separate 300-plan measurement with
time-varying ball state and the future-base pose callback.  The synthetic
0.8 m/s target-speed path measured 1.54 ms mean and 2.33 ms p95 without the
FK callback, and 3.55 ms mean, 4.02 ms p95, and 8.51 ms max with five
kinematic iterations and position-priority refinement.  It is an engineering
timing result, not a native contact or hit-rate result.

A fixed 100-sample sweep of randomly selected reachable terminal poses accepted
12/100 with five re-linearizations before the nullspace branch was added; that
file is retained as a historical baseline.  The current full-body sweeps below
include the new branches and the original physical velocity gate.  Neither
sweep is a hit-rate claim.

For local DTG-like perturbations around the measured racket pose, acceptance was
100/100 for 1, 2.5 and 5 cm position offsets and 96/100 at 10 cm, with contact
times from 0.45 to 0.8 s.  This is a bounded kinematic feasibility result, not
an Isaac hit-rate result; see `DTG_LOCAL_TASK_SWEEP.json`.

## Measured-frame transport correction

The causal acceleration estimate now transports the previous measured body
velocity into the current body frame before differencing. This removes the
spurious acceleration caused by changing orientation, for both linear and
angular velocity. The observation contract provides world +x in body coordinates
through the locked goal normal and world +z through negative gravity. Per-lane
basis history follows partial reset. Existing acceleration caps are unchanged.

Validation: 77 tests passed (exit 0), including finite body rotation at constant
world twist, true inertial acceleration expressed in the current body, and
partial-reset history isolation. Native confirmation of this correction remains
NOT_RUN. No SuperMimic code, process or weights were changed.

## Joint leg/arm replay interface

See `JOINT_REPLAY_PROGRESS.md`. The full-body evaluator now accepts an explicit
12-leg target sequence and optional full-state replay evidence. It also fixes
the contact terminal's extra 10 ms prediction interval. This is CPU-tested
infrastructure for joint takeoff-hit planning; the causal leg producer, contact
transition model and native execution integration remain incomplete.
