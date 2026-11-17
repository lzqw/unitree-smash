# Prediction error attribution and initialized PhysX property audit

Base f9ecf56. No controller, physics, asset, scorer or SuperMimic changes. No training or policy evaluation. This turn does not claim a new prediction accuracy improvement.

## What was established

1. Added symmetric FK error attribution: base and joint contributions sum exactly to the total contact-proxy error vector. Hybrid configurations are mathematical probes only; never trajectory states or controller inputs.
2. Added whole-body centroidal momentum comparison from predicted and recorded native q/v using the same model. Angular momentum is about the whole-body COM in world axes; it is not merely body angular velocity or a direct force-sensor measurement.
3. Performed one independent reset-only PhysX property inspection. There were zero calls to the control step, zero actor forwards, zero optimizer steps and zero episode advancement. Initial794D observation exactly matches history (max delta0). Exit0. This is not a second trajectory calibration or a scored rollout.
4. Actual PhysX tensors match the authored model: masses and COM translations have zero difference; max inertia difference7.9674e-9 kg m². All18 armatures and friction coefficients are0. Position/speed/effort limits, stiffness and damping match the frozen interface exactly. Therefore no parameter edits were made.

## Attribution

At the final800ms node, lane1 contact-proxy error83.24mm is dominated by the base contribution83.53mm; joint contribution1.25mm. Contributions are vectors and can partially cancel, so their norms must not be added as percentages. Lane0 total29.04mm has base28.35mm and joint1.07mm contributions.

The shorter320ms window still has maximum proxy error3.121mm. At its final node lane0 base contribution2.471mm and joint contribution0.718mm explain total3.121mm. This does not justify increasing IK iterations or relaxing3mm.

In nominal airborne portions, reconstructed native-state angular momentum and the ideal CPU model diverge. The present data do not uniquely attribute that difference to contact, numerical integration, speed-limit handling or unrecorded simulator properties. No compensating external force or fitted torque was added.

NATIVE_FRAME_CHECK.json independently compares FK/Jacobian against saved native public tool measurements for both lanes at ticks201..281. Maximum origin-position difference1.241e-6m, normal difference2.142e-6 and velocity difference5.850e-6m/s. This strongly limits a coordinate/measurement-contract explanation for the centimeter-scale trajectory error; it is not a dynamics validation.

## Native property inspection boundaries

`property_inspection01` used the existing line_supervisor, separate teacher slice and unique systemd unit `unitree-teacher-properties-20261118-01`. Identity, child cgroup, command, input script hash, GPU before/after and terminal exit are preserved. GPU free memory was approximately12GB before launch; the inspection initialized only the original two-lane development environment, read tensors and closed. No other service was stopped or signalled.

PROPERTIES.npz contains actual initialized tensor values. Inertias are column-major matrices at COM expressed in the body-prim frame, as documented by the installed tensor API. The source and model name mappings were checked; no mass/inertia fitting was done.

Rigid-body angular/linear damping and gyroscopic USD attributes remain absent/None even in the initialized stage. These are UNKNOWN runtime-engine settings, not proof of zero damping. Reading the SDK's fallback schema would not establish their realized values.

Pure-USD inspection attempts before native initialization encountered missing libusd_tf, missing libpython and unregistered schema plugins (each exit1). A process-local library/plugin path resolved import/read access (exit0); no package install or shared environment modification occurred. Authored/schema joint armature/friction0 was then confirmed against native tensors. No trajectory run was retried for these import probes.

## Tests and next action

FINAL_TESTS.txt:6passed, exit0. Covers exact vector attribution, mutable Pinocchio workspace isolation, real native property comparison, explicit mass perturbation detection, nonfinite rejection, and existing coupled historical-step regression. Native property inspection independently exited0 with PROPERTY_INSPECTION_COMPLETE; no EVALUATION_COMPLETE was written.

Next, exercise the already task-valid candidate on the improved short-horizon physical predictor while retaining all original contact, joint, torque, clearance and terminal collision checks. Historical-command accuracy alone cannot certify an altered arm sequence. Remaining long-horizon error and simulator-contact/limit response must remain explicit; no new reference or native hit is claimed.

Goal remains active and incomplete. Fullbody prediction, PhysX reference, native hit, Recovery andcomplete5 for a new candidate remain unproven. No128-case evaluation was run.
