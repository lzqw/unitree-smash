# G01 corrected standard asset audit

The old fixed-base/camera-demo evidence was invalidated and archived. This result uses a floating Go2 with the pinned official standard PiPER no-gripper URDF, an explicitly assumed 0.3 kg mounting plate and 0.09 kg racket. Total mass is 20.64699130 kg. Source inertials are preserved and checked against PhysX per link. Mount/tool dimensions remain uncalibrated assumptions documented in configs/assembly_v2.json and assets/assumptions.md.

With self-collisions enabled, 100/100 independent 10-second standing/slow-arm-motion episodes passed, with zero numerical failures. All 18 paired actuator response tests passed. Free-flight arm movement caused 0.551446 rad/s paired base response; COM ballistic error was 0.004914 m and angular-momentum drift 0.003179 kg m²/s. These diagnostics initialize in flight only to test coupling; they are excluded from G04.

Feet-on-ground forces were 44.29–57.68 N, airborne forces 0 N. The frozen contact calibration records force thresholds, clearance and hysteresis. Actuator effort/velocity limits remain unchanged. Thirty-six interior joint-limit poses provide a sampled workspace and observed tool speeds; this is not an exhaustive reachable set or evidence of smash performance.

Native raw CSV, logs and source snapshots remain under reports/G01/native_20261022T030602_028800/stand, reports/G01/native_20261022T030602_028800/calibration and reports/G01/native_20261022T030602_028800/stand. Repair GPU usage is charged to the explicitly authorized G04 cumulative ledger, not reset as a new budget. G02/G03 must still be repaired and independently revalidated before G04 training.
