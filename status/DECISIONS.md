# 决策日志

初始：仅有任务规范与证据校验器，所有机器人阶段未运行。
后续按时间记录版本、资产假设、实验修改和预算授权。

## 2026-10-22：训练环境核对

- 依据 NVIDIA Isaac Lab 2.3 官方安装文档（https://docs.nvidia.com/isaaclab/2.3/isaac-lab/local-installation），Isaac Sim 5.1 使用 Python 3.11，工作站建议至少 16 GB GPU 显存；本机锁定的 Isaac Sim 5.1、Python 3.11 和 RTX 5080 满足该条件，实际 GPU smoke 已通过。
- 依据 Isaac Lab 官方 RL 文档（https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/03_envs/configuring_rl_training.html），新任务优先使用 RSL-RL PPO；本项目固定使用 RSL-RL 3.0.1，并保持训练、恢复和独立评测的观测/动作契约一致。
- 当前正式训练配置：单 GPU、headless、Isaac Lab PhysX、RSL-RL PPO；先从 32 个并行环境验证，再按显存和数值稳定性逐步扩展。G04–G07 仍受 `configs/budget.json` 的显式 GPU 小时授权约束，不能因本机有 GPU 就自动启动。
- 训练入口复核后补齐了两项可追溯性约束：`--seed` 现在传入原生 Isaac runner，`--max-gpu-hours` 在逐次 PPO iteration 间检查并写入训练报告；系统 Python 下的 `train.py --help` 也不再提前导入 NumPy。预算闸门行为保持不变。
- 原生 launcher 还会拒绝 `allowed_gpu_ids` 为空或未列出的设备，避免把共享 GPU 的存在误当作本项目的调度授权。
- 在启动子进程前增加 `nvidia-smi` 独占性检查；当 `max_concurrent_gpu_processes=1` 且存在其他计算进程时，launcher 返回 `BLOCKED_RESOURCE` 并保留工作区不变。
- 2026-10-22：用户明确授权 G04 使用 GPU0、最多 8 GPU-hours；已写入 `configs/budget.json`，累计授权上限为 10 GPU-hours（含 G00–G03 的 2 GPU-hours）。实际启动仍等待 GPU0 独占释放，未终止 SAFEFlow 或桌面进程。
- 用户随后授权停止 SAFEFlow；其 14 个训练/启动进程已收到 SIGTERM 并退出。GPU0 仅保留 MPS 守护进程和桌面渲染代理，launcher 将它们视为基础设施，只阻止其他训练计算进程。

## 2026-10-22：恢复训练前的证据重审

- 原 G01 站立测试使用固定基座，不能支撑自由站立/反作用结论；G01/G03 旧 PASSED 已撤回，G02 依赖待复核。保留所有历史文件。
- 对照锁定版本 DirectRLEnv、InteractiveScene 和 Articulation 源码确认 reset 不写物理状态。官方示例也要求显式 write_root_pose/velocity 和 write_joint_state：https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/03_envs/create_direct_rl_env.html 。
- 8 环境零动作 before/after 仿真：重置根状态最大误差 6.972367 → 0，关节位置 0.172519 → 0，关节速度 0.083182 → 0；其他环境状态变化均 0。结果在 reports/G04/reset_audit_{before,after}。
- 即使零动作，原环境也达到约 1.99m；修复 reset 后仍出现异常起跳及高处足接触，继续审计碰撞场景，不将其归为策略能力。
- 三轮训练修订已记录；当前工作为工程错误修复和依赖重验，未启动额外性能调参轮次。

- 进一步审计：旧PiPER相机演示USD的质量分配与标准URDF不一致。固定commit中的标准无夹爪URDF已转换，质量/主惯量逐项核对；新候选包含显式0.3kg安装板与正确尺寸0.09kg工具。资产及安装参数尚待G01复验，旧checkpoint不迁移为有效证据。
- G02旧验证包含同表达式解析比较及永久latch，不能证明收敛/重复接触。G01/G02/G03旧证据已归档失效，详细问题见reports/G04/engineering_reaudit/REPORT.md。
- status/GPU_LEDGER.json采用历史2h/25M步保守预留，之后按作业墙钟及交互上限累计；这不是历史实耗统计。

- G01已以标准完整负载重建资产重新通过。实际公开入口100回合复验、自碰撞开启、18关节配对响应、源质量/质心/惯量与PhysX张量比较、接触标定、离地反作用及COM/角动量诊断均留有日志；verify_goal G01通过。发布assets/quadsmash_standard_v2.usd，并将旧G04权重移入invalidated目录。
- G02/G03仍未通过，当前禁止正式PPO；具体恢复步骤见status/NEXT.md。

2026-10-22 corrected G04 engineering baseline: G03 native public suite and train/safe-stop passed. Retain canonical-reset checkpoint protocol; no mid-contact replay claim. Integrated full-COM/same-flight/contiguous recovery metrics and non-foot-ground force checks without changing acceptance JSON or frozen panels. Capability search 256 lanes: 17 qualified jump landings, zero joint success. Use TRAINING_SPEC PPO starting architecture/gamma/lambda in new jump_ppo_v2.json; keep old configurations/revisions and all costs. Initial 128-env 10-update run only proves execution, not learning acceptance.

2026-10-22 G04 explicit reward correction: stopped corrected_v2_main01 after observed ~98% non-foot-ground contact and no qualified jumps. Initial repaired reward omitted the dangerous body-contact cost explicitly required by TRAINING_SPEC section 7, rewarding low effort while lying down. Add one-time -4 contact cost, weak preparation support reward and graded post-landing recovery quality. Frozen success protocol, assets, actuator limits and action scales unchanged. New config jump_ppo_v2b.json records reward version, uses weight initialization (not claimed identical full resume) from stopped checkpoint. This attempt and all earlier three invalid-physics revisions remain logged; neither GPU nor interaction budget resets. Do not launch an open-ended series of reward searches.

G04 command telemetry correction: RslRlVecEnvWrapper clip_actions=1 discarded raw Gaussian commands before env.action_before_clip. Public native train/eval wrappers now use clip_actions=None; env clamps at the identical boundary before targets, so applied commands/reward/history are unchanged. Clipped-action counters from earlier runs including corrected_v2b_main01 are not valid estimates of raw policy saturation; their zeros must not be used to claim absence of clipping. New runs retain raw commands and count actual clipping.

2026-10-22 Explicit G04 shaping revision after native evidence: corrected_v2b_main01 completed 200 PPO updates / 13,107,200 interactions. Independent panels each 0/200, all 400 non-foot-ground contacts. policy_consistency_v1 controlled comparison: zero actions 0/32 body contacts; deterministic policy nominal 32/32, stochastic nominal 0/32; stochastic max COM rise .00438 m, maximum joint speeds average 29.63 rad/s. Joint perturbation changes normalized observation by at most .341, nearly identical initial actions, so the failure is not an evaluation-only reset mismatch. Positive vertical-speed integration credits oscillations without net takeoff progress. Explicit config jump_ppo_v2c replaces ONLY that term with a bounded timed full-COM crouch/apex tracking reward (reference reset COM, -0.06 m at .93 s, +0.14 m at 1.25 s, 0.06 m width; active .65–1.7 s). Other reward terms, PPO coefficients, noise initialization, hardware, action scales and frozen acceptance remain unchanged. Weight initialization from v2b is labeled a new revision, not exact continuation. All old three invalid-physics revisions and subsequent engineering versions/costs remain retained; no fresh GPU or interaction budget. This is a recorded hypothesis, not a learned-jump claim. Avoid further unbounded reward revisions.

2026-10-22 G04 fixed-config chain finished: v2c reached global800 updates; latest independent400 episodes success0%, body contact39.25%, max COM rise.0095839m, numerical failures0. G04 explicitly FAILED; G00-G03 gate checks still passed. All driver/GPU jobs exited; cumulative conservative cost 3.427470248h /99634208 interactions, under8h/200M. 104 CPU tests passed. No extra v2d execution: its additional-revision authorization question remains unanswered. Complete measured policy/evidence/comparison/curves retained; no hardware impossibility claim.

2026-10-22 supplementary authorized sensor audit: reports/G04/foot_ground_audit_v1 completed 128x400 policy steps. Independent per-foot static-ground filters matched production net force vectors/contact/airborne labels across256000 environment physics substeps, including real support and flight; no production sensor fix justified by this finite test. Curated replay of existing capability parameters produced30/32 qualified landings and1/32 complete recovery; this newly observed engineering example is not trained-policy acceptance or proof of robustness. Latest deterministic PPO remains0/400. Total charged3.432439632h/99685408 interactions; no active GPU jobs. Additional one-revision question still unanswered; no new training.
