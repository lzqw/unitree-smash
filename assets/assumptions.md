# Coupled asset assumptions

- Go2 is loaded from the local Unitree USD recorded in `reports/G00/environment.lock.json`.
- PiPER is the standard AgileX `piper_v1.usd` from the pinned `piper_isaac_sim` commit. The
  two prismatic gripper joints and vendor root fixed joint are excluded from the 18-DOF task.
- `standard_racket_v1` is a parameterized payload, not a manufacturer calibration: mass 0.09 kg,
  square collision blade with 0.11 m half-width, fixed to PiPER `link6` through `RacketMount`.
  Its source and status are recorded in `assets/rackets/standard_racket_v1.json`.
- Leg and arm actuator effort, velocity, stiffness, and damping values are simulation assumptions
  in `quadsmash/tasks/stand/coupled_env.py`; they are not claims about the physical robot.
- The combined USD preserves the full PiPER articulation and the racket rigid-body payload. No
  external force or per-step pose teleport is used to couple the arm to the Go2 base.

## Candidate v2 correction (2026-10-22, pending full G01)

The earlier camera-demo USD is not the validated standard-arm model. The candidate now uses the pinned official standard `piper_no_gripper_description.urdf`, resolving mesh URIs only and preserving all masses/inertias. The gripper is removed as permitted by the original task. Arm mass is 4.169990896 kg; the earlier demo has a 10.162549 kg `arm_base` and different link masses. No old policy is accepted on the candidate.

The previous mount was a massless fixed constraint with a 0.45 m offset. Candidate v2 defines an explicit 0.3 kg, 0.22 × 0.16 × 0.012 m plate, centered at [0.05, 0, 0.064] m in the Go2 base frame; the arm is attached to its top surface. These mechanical dimensions and mass are uncalibrated simulation assumptions permitted by G01, not manufacturer values. The racket remains 0.09 kg; its blade is 0.015 × 0.22 × 0.22 m with its COM 0.18 m from link6. This fixes the earlier scale/translation inconsistency before retraining.

All numerical source/asset hashes are in `assets/standard_piper_manifest.json`. The candidate combined USD is `reports/G04/standard_assembly/robot.usd`. Actuator limits remain 23.5 Nm / 30 rad/s for legs and 100 Nm / 5 rad/s for the arm, with the same gains; these are still simulation assumptions (arm limits are present in the source URDF), not calibrated hardware performance.
