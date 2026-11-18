# SuperMimic native gates audit 2026-11-19

## Scope

本轮只完成候选安全筛选、GPU PhysX 事件对齐探针和原生动作响应方向诊断。没有启动Recovery、Approach、R8→A8→R8或大规模策略训练。

## Environment

- Worktree: `/home/autodrive/supermimic_work_20261118`
- Base HEAD: `4ba436aee54535a064c8c004f76f5190bf8e67fe`
- Interpreter: `/home/autodrive/Isaac/env_isaaclab/bin/python`
- Torch: 2.7.0+cu128
- GPU: RTX 5080, CUDA device `cuda:0`
- Isaac Sim: 5.1 headless GPU PhysX
- Execution overlay: `/tmp/supermimic_g07_execution_overlay_20261118`
- CPU tests: `434 passed, 1 warning`

## Gate 1: candidate safety screening

Candidate SHA256:
`80f0610827342ed6ee85cc08176fd52463c034f4f75fa413b02421c0e2bd262d`

Decision file:
`/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261118_233655/action_ball_candidate_screen_v2.json`

Decision: `passed=false`, rejected because held-out preimpact 4-step metric degraded from `2.46004` to `2.83483`. The candidate remains diagnostic only.

## Gate 2: GPU event-aligned probes

Completed native paired-prefix probes with four branches `nominal_a`, `nominal_b`, `minus`, `plus`, zero actor/world/critic updates.

- epsilon `0.001`: 11 independent contact-lane bundles, aggregate SHA256 `4bd95a623f02b11687579948b9abb2a9db56722a74248b606a1b97aeed59a0d9`
- epsilon `0.01`: 6 independent contact-lane bundles, aggregate SHA256 `8622a9fb768fe456712daa27002e0b4ef928340b8f6a1dd7cdeaff64ce4ca845`
- All executed branches exited with code 0.
- Frame-only lanes were excluded from Approach contact evidence.

## Gate 3: native response direction

Legacy candidate on epsilon `0.001` contact11 produced response cosines including `-0.981`, `-0.424`, and `-0.168`. Local Jacobian cosines had the same negative cases.

Legacy candidate on epsilon `0.01` contact6 produced response cosines including `-0.232` and `-0.107`.

The finite calibration utility was run for exactly 256 GPU updates with declared disjoint splits. Results did not generalize:

- train10 / holdout37 at epsilon `0.001`: final holdout cosine `-0.9223`
- train1 / holdout37 at epsilon `0.01`: final holdout cosine `0.9398`, but one held-out sample is insufficient for a receipt
- train5 / holdout65 at epsilon `0.01`: final holdout cosine `-0.5159`

The calibration artifacts explicitly have `receipt_status=not_issued_requires_independent_native_direction_validation`.

## Gate 4: native receipt

No native response-direction receipt was issued. The current evidence shows inconsistent direction across independent contact lanes. No passing bounds were invented and no negative-direction sample was filtered.

## Remaining blockers

1. The action-conditioned ball residual still does not reproduce the PhysX response direction across independent contact modes.
2. The valid response support is heterogeneous. Some probes have very small native response norms, while others have larger responses with incorrect directions.
3. Event/action timing and final-command-to-ball response need further diagnosis, including contact mode and control-cycle alignment.
4. A native receipt cannot be issued until an independently declared held-out split passes direction, amplitude, repeatability and local-Jacobian checks.
5. The GPU probe path is operational, but this does not authorize Actor or WM training.

## Follow-up timing correction

The probe specification now supports an explicit `action_lead_ticks` field and
`capture_substeps=true`. This allows a perturbation to be applied before the
recorded contact edge while the native substep trace is captured at the event
tick. The change is covered by CPU contract tests and is intended to diagnose
actuator-delay and contact-edge alignment. It does not alter the policy action
contract and it has not been used to issue a receipt.

## Training status

- Actor updates: 0
- World-model updates: 0
- Critic updates: 0
- R8→A8→R8: not run
- Large-scale training: not run
- Formal 128-environment evaluation: not run

## Conclusion

The safety gate is working and rejects a degraded candidate. GPU event-aligned probing is implemented and executed. Native response-direction validation remains failed, so contact-aware BPTT and all policy training remain blocked.
