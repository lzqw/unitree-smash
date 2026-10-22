# G04 训练目标与探索问题：CPU 审计及修订准备

**本轮准备目标已完成；G04 正式状态仍为 FAILED，独立策略评测仍为 0/400。没有新任务训练、GPU 仿真或 GPU 诊断作业。** 基准提交为 `65396a21bd8065400df91437432bd12a0a46bf81`，工作分支为 `g04-diagnostics-preparation-20261022`。用户本轮要求完成诊断代码、CPU 测试和待授权提案后停止，不是继续原训练目标。

输入为 [REQUEST.txt](REQUEST.txt)。根据粘贴审查文本在本地复现测试，没有声称运行外部 sandbox 链接中的 ZIP 或原始 5 项测试。所有历史结果、失败报告、模型与冻结契约继续保留；本目录的合成案例不参与正式成功率统计。

## 六项交付与证据

| 要求 | 已完成的交付 | 验证范围 |
|---|---|---|
| 小跳主要恢复收益的 CPU 反例和正反测试 | `scripts/audit_g04_rewards_cpu.py`、`tests/test_g04_reward_review.py`、[CPU 输出](cpu_probe/results.json) | 合成状态机输入，不证明真实动力学可行 |
| 奖励、动作裁剪、均值越界、逐关节 std 遥测 | `jump_reward_audit.py`、`jump_diagnostics.py`、`action_diagnostics.py` 和原生接口的只读接入 | 默认奖励逐位等价；CPU 验证不改 RNG、参数、归一化、梯度或分布对象；原生观察器未运行 |
| COM 参考与可行动作时序诊断脚本 | `scripts/diagnose_g04_rewards.py`、`scripts/analyze_g04_alignment.py`、[历史时序分析](historical_alignment/alignment.json) | 历史事件时间实测；新的配对仿真及分项回报比较未执行 |
| 合格跳跃门控与弱引导草案 | `recovery_draft`，版本 `qualified-recovery-draft-v1` | 旁路计算，未替换生产奖励、未训练验证 |
| 验证加载后 std 重设 | `reset_initial_std`、[真实旧检查点 CPU 审计](cpu_probe/results.json)、`initial_noise_assertion.json` 的未来运行输出接口 | 证实已有重设逻辑存在并生效；增加断言和记录，不宣称修好了缺失逻辑 |
| 两个独立训练提案、预算和停止规则 | [方案说明及待授权命令](EXPERIMENT_PROPOSALS.md)、两份 proposal JSON | 仅草案，未新增授权；guided 的数据/BC/前缀训练器尚未实现 |

## 关键发现

### 1. 奖励偏差成立，正式验收没有误判

合成输入为地面支撑 → 40 ms 离地、COM 上升约 2 mm → 稳定落地 1 s。原状态机得到 `valid_jump_landed=false`、`recovery_seen=true`、`formal_success=false`。两项旧主要恢复收益合计 **4.9999997**；新草案的主要恢复收益为 **0**，弱引导为 **0.0500000**。

这里隔离的是 `3*stable_recovery + 2*recovery_quality` 两项，其他项设零；不能把该数字当作真实控制器的完整总回报。短时高跳、长时低跳的反例也成立。合格飞行后稳定恢复的正例保留主要收益；机身触地、仅站立与不连续恢复的反例继续不能通过正式判定。

草案用当前 `valid_jump_landed & ~body_ground_seen` 门控两项主要恢复收益，保留安全落地后的分级弱引导，速率 0.05/s、**每回合最多 0.1**。弱引导累计量只在真实新回合清零，不能通过重复小跳补充额度。该权重是待验证的工程草案；没有证据证明它能让 PPO 学会跳跃。

`JumpEnv._get_rewards` 的默认算式保持原样。观察器打开或关闭时，24 组 CPU 张量输入的输出与从基准提交保存的原函数逐位一致；分项重建允许浮点加法顺序造成的 2e-6 绝对误差。正式 `jump_metrics.py` 未改动。

### 2. COM 时序需要真实分项轨迹，现有摘要不能补造

已有完整恢复轨迹 env50：起跳 **1.28199065 s**，落地 **1.53598738 s**。v2c 参考的正向 Gaussian 项中心为 **1.25 s**，比该实际起跳早约 **31.99 ms**。这是项中心，不宣称它恰好等于完整参考函数的精确极大值，更不能据此证明该参考不可能实现。

同一个已有参数组（源搜索 env65）的三个重放环境 40/50/60，连续恢复分别约 **0.3060 / 0.9480 / 0.1300 s**，只有中间一个完整成功。它们是精选诊断轨迹，不是独立随机鲁棒性估计。

历史回合摘要没有该工程重放的实际逐步 COM/vz、reset COM 与奖励分项。因此本轮只出具历史时序分析，不比较其真实总回报或绘制虚构轨迹。准备好的配对原生脚本将采集这些缺失量，之后分析脚本才能画实际 COM/参考、竖直速度与分项奖励曲线。

### 3. std 重设本来就存在；遥测不应再引入扰动

在 CPU 加载真实 `corrected_v2c_main01/native/checkpoint.pt` 后，18 维 std 原值约 0.692–0.792，重设后均为 **0.149999991**。其余模型与 normalizer 状态逐张量一致，原检查点文件不变，没有对该模型做优化器更新；`std_remains_learnable=true`。

新动作统计记录 rollout 实际原始动作的裁剪比例、均值越界比例、逐关节 std/min/max、解析裁剪概率和“裁剪均值与裁剪后期望动作”的差距。PPO 的统计用确定步幅抽取至约 4096 行并记录步幅/样本数，不把它说成全部 rollout 的精确统计。命令裁剪与关节目标/力矩饱和分开，不挪用早期 wrapper 提前裁剪造成的无效计数。

只读统计直接执行锁定版 actor/normalizer 前向，不采样、不更新 normalizer，也不替换 `policy.distribution`。CPU 测试比较了 RNG、参数、buffer、梯度和模块状态；还用 8 样本的合成 PPO 单元夹具验证带统计的一步更新与 stock PPO 更新产生相同参数、loss 和 RNG。该单元测试没有仿真环境或任务 rollout，不是一次新的任务训练。

## 诊断实现边界

- 原生配对脚本默认只打印 [执行计划](native_diagnostic_plan.json)，无 Isaac 导入。`--run-native` 需要实际 `run_gpu_job.py` 父进程和对应 RUNNING 记账。本轮测试了默认入口及拒绝非受控原生调用，没有启动其原生路径。
- 独立诊断初态种子 311001、噪声种子 311002，8 组 × 16 环境，各 400 策略步，共最多 51,200 次交互。模型默认为原 200/800 更新权重，以实际路径和 SHA 标识。
- 每个环境只统计首回合，在自动 reset 前记录终态和奖励。实际初态、同场景观测差异、COM、参考、vz、起落时刻、奖励/惩罚项、裁剪前后动作、目标关节角都会保留；原模型 std 与执行噪声 std 分开记录。
- 恢复计时逐物理子步采样，分别保留支撑、姿态、水平/角/关节速度、无机身接触及联合条件的连续窗口。`all` 表示任意落地后联合条件，仍需结合合格落地标志；不得把各条件不同时间的最长片段拼成正式成功。
- 新奖励只是旁路数值，既不传给 PPO，也不改正式成功条件。未来若启用它训练，需要单独版本、接入审查与授权。
- 每 25 更新的具名 checkpoint 复制自同一次规范重启保存，避免再次 reset；目前仅完成代码与编译检查，没有运行 GPU 保存/恢复验证。

## 测试、完整性与资源

- [全套 CPU 测试](all_tests_final.log)：**121 passed**；[JUnit XML](all_tests_final.xml)。覆盖原有 104 项及本轮 17 项新增测试。
- 最后一次诊断 CLI 参数调整后的 [定向复验](diagnostic_final_tests.log)：2 passed；[XML](diagnostic_final_tests.xml)。
- [G04 原验收检查](goal_verification.json)：G00–G03 依赖通过；G04 仍仅因 `jump_recovery_success_rate=0.0 < 0.9` 失败。
- [基准清单](baseline_manifest.json) 的 701 个历史证据、状态、物理/指标实现、资产及冻结契约文件逐字节未变。GPU 账本、原 `STATUS.json` 和正式策略结果未改。
- 本轮 GPU 作业 0、任务训练 0；原记账仍为 **3.432439632 / 8 GPU h，99,685,408 / 200,000,000 交互**。没有重置预算。
- 命令见 [commands.log](commands.log)，代码差异见 [code_changes.patch](code_changes.patch)，交付核对见 [completion_audit.json](completion_audit.json)。

## 后续决策

先审阅 [两个互斥提案](EXPERIMENT_PROPOSALS.md)。低噪声对照最多 50 更新，保持 v2c 奖励不变，避免同时修改噪声与奖励后混淆归因；教师引导方案须另行实现数据、BC 和前缀接管接口，并保持独立实验身份。当前新目标不允许执行任何一个 GPU 命令。

本轮按要求在报告、代码差异、测试日志和待授权命令交付后停止。**准备工作完成不等于 G04 策略达标，也不构成下一轮训练授权。**
