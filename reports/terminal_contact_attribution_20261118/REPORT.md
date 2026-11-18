# Terminal contact attribution before further optimization

This change adds an independent offline diagnostic of the selected complete causal CPU trajectory from 4528db9. No control algorithm, physics, scoring, base policy, Recovery or SuperMimic changes. No native/128-case run or training. No future native states are consumed.

## Evidence

All measurements below use the same predicted terminal root pose, original ball event, authored local contact offset and six-joint mapping. Original prefix and source hashes are checked before analysis; computed actual contact residual agrees with the saved prefix to 1e-10 m.

| Fixed-terminal-base diagnostic | Position error |
|---|---:|
| Actual predicted arm state | 385.969 mm |
| Counterfactual exact tracking of planned arm reference | 543.119 mm |
| Position-only local IK | 206.657 mm |
| Position + normal local IK | 224.605 mm |

The vector identity actual−target = (actual−reference)+(reference−target) closes exactly. Its component lengths are not additive. Tracking displacement is 272.786 mm, but removing it would increase contact error in this counterfactual geometry; tighter arm tracking alone is not established as the remedy. Counterfactual poses do not account for changed arm/base dynamics and are never executable references.

The existing conservative serial-link/string-strip bound places this target 49.449 mm outside its optimistic reach sphere (including 0.5 mm allowance) at this base pose/time. This bound is for the current authored strip, not all possible contact times, base poses or robot trajectories. The two local IK results are not global infeasibility proofs. The final arm joint reaches its limit in both local solves. Position-only IK has normal error 1.058, while position+normal reaches 0.1042 but retains 224.6 mm position error.

The minimum-norm radial translation of that sphere toward the target is approximately [14.43,1.87,47.26] mm in world coordinates. A translation aligning the locally found position-only pose is [60.31,7.81,197.51] mm. These are **unrealized geometric targets**, not componentwise necessary lower bounds, proven global minima, valid takeoff states or generated leg commands. Moving the base can change contacts, dynamics, timing and the arm solution; complete revalidation is required.

## Velocity attribution

Actual predicted contact velocity: [-0.76185,1.47084,-0.66709] m/s.
Base-motion contribution: [-1.89444,-0.79942,0.29950] m/s.
Arm-rate contribution: [1.13258,2.27026,-0.96659] m/s.
The root linear/angular transport formula and the full Jacobian agree within 2.23e-16 m/s. The base's negative X contribution outweighs the arm's positive X contribution. These are model values, not measured native racket speeds or evidence of a collision.

## Verification

30 tests passed (0.90s), exit0. Production loaded-model contact Jacobians were checked by tangent-space finite differences; returned arrays remain independent of mutable Pinocchio workspace; nonzero rotation/translation and body-twist transport were checked. The source-URDF reach geometry and loaded-USD FK were compared at two deterministic configurations. An initial 1 nm comparison failed with a 50 nm position discrepancy between these differently serialized models; failure output is preserved. The model-agreement test uses an explicit 1 micrometre / 1e-6 normal-vector tolerance. No physical/contact gate was changed. The conservative reach result should be interpreted at this model precision, not as nanometre-exact PhysX geometry.

Initial diagnostic RESULTS.json and final DIAGNOSED.json both exited0; the latter adds explicit unrealized target labels. Original results are preserved. No qualified reference is exported. POSITION/VELOCITY and full task gates remain failed; PhysX reference, native hit, Recovery and complete5 remain NOT_RUN.

## Next action supported by this evidence

Use full-prefix leg/base response to bring the ball event into the reachable set while protecting the already improved normal and velocity residuals. Include terminal root translation/velocity and contact time in diagnosis before spending further wrist-only iterations. The current arm reference was designed for a different future base motion and cannot be repaired solely by enforcing its tracking more tightly. Any retargeted arm reference or changed push-off sequence must begin from actual command history and pass the complete original dynamics and final task gates. Do not inject the geometric translation/velocity target into the state or start a 128-case evaluation.
