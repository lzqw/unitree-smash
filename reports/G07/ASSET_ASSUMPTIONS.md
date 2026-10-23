# G07 incremental scene assumptions

The original signed G01 asset assumptions remain in [assets/assumptions.md](../../assets/assumptions.md). This file records the separately versioned G07 net scene.

## G07 net profiles (2026-10-23)

`environment-local-net-v2` corrects the earlier double world/local translation but uses a constant 1.524 m box. Its motion/ball results remain curriculum evidence; it does not implement the original 1.55 m side height. The unused formal v1 protocol is superseded for that reason.

`environment-local-profiled-net-v3` restores the original center/side heights of 1.524/1.55 m. A closed static triangle mesh has a 32-segment piecewise linear parabolic upper edge; its width remains the declared 2 m benchmark and its thickness remains 0.02 m. The parabolic sag is a geometric assumption, not a measured net. Collision response/material and the full robot remain unchanged. Ordered crossing clearance uses the same piecewise linear top at the crossing y. `profiled_net_native_control01` measured all 20 cloned meshes and passed 16 ball-clearance cases plus 4 deliberately colliding robot sensor positive controls.

The v3 net retains the original ground-to-top solid benchmark collision envelope. Its bottom edge and compliance are not a calibrated real net; this is a declared 2 m wide research obstacle. Successful above-net trajectories and physical robot limits are reported within this model, not as real-court certification.
