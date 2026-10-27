# Completed contact-flight recovery-v5 residual15 evidence

This package contains the actual completed15-update TRAIN job, its exported
policy and canonical ground checkpoint, all15 diagnostic checkpoints, all738
closed training episodes, fourteen recorded100Hz trajectories, and a separate
zero-noise original128-case cold evaluation. The cold result is0 complete,
0 legal airborne string contacts,0 qualified contact recovery, and0 numerical
failures. The training result8 air/1 recovery/0 complete is a changing-policy
result and does not transfer to the frozen exported policy.

The actual imported runtime is the separately versioned recovery-v5 tree at
commit9311c086590dfb61c1d7559156426ebedc63b5e1. Each run includes an execution
source snapshot distinct from the Main GPU scheduler snapshot. The physical
asset, court/net, original128 cases, original target and50/70/85 thresholds
remain unchanged. The only explicit scoring revision associates recovery with
the actual qualified contact flight. It does not manufacture a smash success.

Required companion for complete runtime assets and pinned inputs:
[G07_contact_recovery_v5_cold128_complete_inputs_and_assets_20261027.zip](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_contact_recovery_v5_cold128_complete_inputs_and_assets_20261027.zip),
64,120,557 bytes, SHA256
`e0a9e87cdec749caf39d4f86230e7851e35d7dddc13ec8af425809fee798db1d`.
Its restore program reconstructs1016 original path records including all931
then-held inputs, all ten composed USD layers, standard PiPER URDF and the Go2
walking policy. Use the manifest and explicit restored prefix. The companion
has already passed a fresh CPU restore and public-command replay. This new
package supplies the completed15 learner and cold data excluded from that
earlier source-only companion.

Pinned installed IsaacSim5.1, IsaacLab, Python3.11/Torch2.7 CUDA environment
and the built-in OmniPBR shader are still external dependencies. Preserve the
original absolute USD/precision paths in a clean namespace, or audit relocation
explicitly. No fresh native-physics replay or arbitrary mid-episode restoration
is claimed. `training15/checkpoint.pt` is the canonical ground restart;
`training15/ppo/model_*.pt` are diagnostic boundaries and must not be treated
as canonical physical restart files.

Use the exact commands in each `job.json`, the separate execution snapshots
and the SHA manifests for native reproduction through the original GPU0
guard. `cold128/public_and_actions_first2.npz` and
`cold128/native_command_inputs.json` provide real public794-channel inputs,
actual actions, joint defaults and limits for CPU command checks. Formal A/B
cases were not read and formal acceptance remains false.
