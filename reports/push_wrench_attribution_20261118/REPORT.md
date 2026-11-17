# Attribution of the push-off QP/nonlinear momentum discrepancy

## What changed

Added an opt-in, read-only wrench audit to `predict_speed_limited_spheres`. It integrates each authored contact's world impulse and its moment about the pre-step COM, drive torque, and per-joint saturation counts over the unchanged 160 physical prediction substeps. `audit_push_wrench_v1.py` replays the previously frozen 29-command CPU sequence; it does not generate a new controller candidate or read future native states.

The audit reconstructs moment arms from the same sphere bottom points used by the contact Jacobian. Internal actuator and joint-limit torques are not counted as external whole-body moments. The external contact moment is independently compared with the fullbody COM angular momentum change. Numerical integration error is reported separately from the local dynamics residual.

## Results and attribution

All 29 endpoint q/v states are **exactly identical** to the previous saved sequence. Audit exit is 0. Maximum contact angular impulse / COM momentum integration discrepancy, expressed as an interval-average moment, is **0.084572 Nm**. This is small relative to the previous **11.421963 Nm** QP/nonlinear discrepancy, but is not zero and is not a PhysX accuracy measurement.

For the eleven QP-adopted intervals:

- Changing only the moment arm from foot-body origin to authored sphere bottom, keeping the QP forces fixed, changes the moment by at most **2.139971 Nm**.
- The remaining integrated contact-force/time response term reaches **13.291335 Nm**. Terms are vectors and can oppose each other; their maxima must not be added as scalar contributions.
- At node 11, QP predicts front-foot vertical forces of **58.658 / 57.808 N**, whereas the fine-step model averages **81.260 / 81.118 N**. The contact mask stays on those two feet, so this discrepancy is not explained solely by a different active-foot set.
- At node 0, RR contact appears at microstep 1 (62.5 microseconds), then releases at microstep 90. The QP held its initial three-foot support set fixed. Thus contact-mode mismatch also exists in the sequence.
- At node 0, both front thighs and calves spend all 160 substeps at the original effort cap. The largest mean drive-torque mismatch is RL calf: QP **6.040 Nm**, fine-step average **9.591 Nm**. The saturated joints and the joint with largest average mismatch are not necessarily the same.

These data establish a model-contract mismatch between the constant-acceleration, end-step unsaturated QP and the time-varying saturated/contact-switching rollout. They do **not** identify a unique physical cause in native PhysX, nor prove that correcting only one term will fix the entire trajectory. Merely moving the QP contact point cannot explain away the remaining discrepancy.

## Verification and limits

`TESTS.txt`: **54 passed**, exit **0**. New test checks audit-on/off q/v and diagnostics equality, unchanged input/history, torque integral bounds, saturation count bounds, and independent wholebody momentum balance on a recorded initial state. Existing physical admission, diagnostic contact, push-off, implicit drive and MPC tests also ran. Original constraint tolerances were not changed.

The replay does not create a qualified reference. Previous terminal rejection remains: 584.672 mm position error, normal 0.258904, velocity error 4.392869 m/s. No new strike, Recovery success, complete5 or native result is claimed. PhysX and 128-case evaluation are NOT_RUN. SuperMimic, assets, scoring, training weights and default control remain unchanged.

## Reproduction

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.audit_push_wrench_v1 --output /tmp/push_wrench_new_attempt.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_prediction_wrench_audit_v1.py tests/test_experimental_sphere_admission_v1.py tests/test_diagnostic_nonfoot_prediction_v1.py tests/test_push_off_support_qp_v1.py tests/test_implicit_command_support_qp_v1.py
```

The original output is `RESULTS.json`; the audit refuses overwrite. `DRIVE_ATTRIBUTION.json` derives joint-labelled saturation and mean drive errors from those raw arrays using the loaded model's joint order. `MANIFEST.json` records implementation and frozen input hashes.

## Next implementation

Use a saturation- and contact-event-aware command-response model for the offline push-off task, or solve a bounded local nonlinear command correction through the existing fine-step validator. Preserve current limits and original-command fallback. Judge the resulting task response by the integrated force and angular momentum measurements, not only the QP objective. A changed command sequence must then undergo complete trajectory and terminal ball-task validation; the current trace cannot certify it. Do not increase push-off gains or relax constraints to compensate for this model discrepancy.
