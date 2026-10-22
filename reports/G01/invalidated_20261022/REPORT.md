# G01 coupled Go2 + PiPER asset audit

The official AgileX `piper_isaac_sim` repository is vendored at commit `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2`. The generated task USD references the local Go2 USD and the PiPER `piper_v1.usd`, removes only the two gripper prismatic joints and the vendor root fixed-joint override, and connects the PiPER arm base to the Go2 `base` through `/World/MountJoint`.

The Isaac Lab/PhysX smoke created one articulation with 18 active joints (12 legs plus six arm joints), verified the fixed mount body relationships, and ran 128 ten-second standing episodes across 32 environments. All episodes remained upright and had zero numerical failures; the measured stand success rate was 1.0.

The USD was rebuilt after auditing the composed reference paths; the root-layer override now deactivates `/World/Piper/piper_camera/root_joint`, and the standing smoke was rerun on that asset.

The final asset includes a 0.09 kg parameterized standard_racket_v1 rigid payload, collision blade, and RacketMount fixed to PiPER link6; the full stand audit was rerun after this addition.
