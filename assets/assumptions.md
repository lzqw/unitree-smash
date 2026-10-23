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

## G07 net profiles (2026-10-23)

`environment-local-net-v2` corrects the earlier double world/local translation but uses a constant 1.524 m box. Its motion/ball results remain curriculum evidence; it does not implement the original 1.55 m side height. The unused formal v1 protocol is superseded for that reason.

`environment-local-profiled-net-v3` restores the original center/side heights of 1.524/1.55 m. A closed static triangle mesh has a 32-segment piecewise linear parabolic upper edge; its width remains the declared 2 m benchmark and its thickness remains 0.02 m. The parabolic sag is a geometric assumption, not a measured net. Collision response/material and the full robot remain unchanged. Ordered crossing clearance uses the same piecewise linear top at the crossing y. `profiled_net_native_control01` measured all 20 cloned meshes and passed 16 ball-clearance cases plus 4 deliberately colliding robot sensor positive controls.
