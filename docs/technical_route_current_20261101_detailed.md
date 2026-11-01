# Go2 + PiPER 腾空扣杀：技术路线、程序入口与当前证据

更新时间：2026-11-01
仓库：`lzqw/unitree-smash`
当前代码分支：`g07-contact-flight-recovery-v5-20261027`
当前任务：G07，GPU0 持续训练与独立评估

这份文件只记录当前实际采用的路线、已经运行的程序、可复现入口和已验证结果。模型、视频、轨迹和 Isaac 运行目录留在本机，不作为 Git 文件上传。

## 1. 目标和验收口径

目标是在真实 Isaac Lab/PhysX 场景中，让 Go2 搭载 PiPER 和球拍从地面起跳，完成同一回合中的：

1. 空中实际触球；
2. 出球速度达到当前开发阈值 5 m/s；
3. 羽毛球实际越网并落入合法目标区域；
4. 机器人完成触地、缓冲和连续稳定恢复。

8 m/s 是历史协议，仍保存在旧评估字段中作对照。当前开发主指标是 5 m/s。部分冻结评估文件中的 `complete` 和 `attack_outgoing` 仍按旧的 8 m/s 定义，因此必须使用逐回合 5 m/s 事件链重新统计，不能把旧字段的 `0` 直接解释成当前 5 m/s 成功为零。

任务保持真实场地、球网、球拍碰撞、关节限位、执行器映射、起跳和落地判定。没有降低网高、增大拍面、添加悬空外力、传送机器人或提前吸附羽毛球。

## 2. 当前总体路线

```text
固定真实物理协议和任务判定
        ↓
冻结的 shared-target 全身基础控制器
        ↓
18 维全身 residual Gaussian PPO
        ↓
拍速、击球对齐、落地质量和完整事件链奖励
        ↓
击球后选择性恢复教师（训练时 blend 0.4）
        ↓
成功示范保留和 anchor 监督，防止策略漂移
        ↓
5120 环境大规模训练
        ↓
固定 128 例冻结评估、视频和逐回合事件审计
        ↓
关闭额外 post-hit 教师，检查学生是否真正学会恢复
```

核心判断是：完整成功不是单独的挥拍问题，而是“起跳—触球—出球—越网—落地—持续稳定”的合取事件。当前路线先用可解释的基础控制和局部恢复教师把策略带入有效区域，再让 PPO 学习全身 residual，最后逐步减少额外教师依赖。

## 3. 各模块的实际职责

### 3.1 基础控制器

基础控制器由已有的 shared-target/full-racket 控制链提供名义全身动作和球拍执行。它负责任务接口、球拍接触相关几何、关节目标混合、限速和原有恢复参数。它不使用未来回合结果，不读取隐藏的成功标签。

相关代码：

- `quadsmash/control/full_racket_rally_controller_v4.py`
- `quadsmash/control/full_racket_teacher_v4.py`
- `quadsmash/control/full_racket_body_residual_v4.py`
- `quadsmash/control/full_racket_residual_v4.py`
- `quadsmash/tasks/stand/contact_recovery_origin0_v5.py`

### 3.2 Residual PPO

策略不是从零直接输出完整关节目标，而是输出 18 维全身 residual。residual 经过统一的动作缩放、关节限幅、手臂目标速率限制和已有执行接口后，才进入 PhysX。

当前训练使用 Gaussian PPO，策略观测包含球、机身、腿、机械臂、拍面和任务阶段等公开状态。训练使用真实下发接口的结果计算奖励和下一状态。

相关代码：

- `scripts/train_shared_target_ground64_success_retention_v1.py`
- `quadsmash/rl/full_racket_body_residual_ppo_v4.py`
- `quadsmash/rl/native_ppo.py`
- `quadsmash/control/full_racket_body_residual_v4.py`
- `scripts/initialize_residual_actor_v4.py`

初始化采用“actor transfer + fresh learner”：从已验证的当前最佳 actor/观测归一化状态初始化，critic、Adam 状态、随机数和物理回合状态重新开始。这样可以利用已有动作，又避免把不兼容的旧训练状态直接续接。

### 3.3 训练奖励

当前奖励同时考虑：

- 实际击球接触和拍面/字符串中心对齐；
- 真实释放后的攻击进展；
- 出球速度，5 m/s 处饱和，超过 5 m/s 不再单独鼓励无限加速；
- 落地姿态、机身角速度、水平速度和关节速度；
- 触地后的持续稳定窗口；
- 击球前 residual 对参考动作的轻量偏离代价。

拍速奖励仍然保留，但它不是唯一目标。只有球侧结果和机器人恢复同时满足，才计入完整成功。

相关代码：

- `quadsmash/control/success_priority_alignment_reward_v1.py`
- `quadsmash/control/landing_recovery_reward_v1.py`
- `quadsmash/rl/prehit_teacher_prior_v1.py`
- `configs/g07_success_priority_speed5_v1.json`

### 3.4 击球后教师 blend 0.4

训练命令中的：

```text
--posthit-teacher-policy recovery_assist_residual_trace_gated_posthit_policy_01/policy.pt
--posthit-teacher-blend 0.4
```

实际动作逻辑是：

```text
PPO 原始 residual = a_rl
击球后教师 residual = a_teacher
有效 residual = 0.6 * a_rl + 0.4 * a_teacher
```

教师触发条件是当前公开观测的 `policy[:, 773] > 0.5`，即已经观察到击球阶段；`teacher_all_phases` 没有打开。击球前、起跳前和挥拍准备阶段不调用这个外部教师。

教师主要提供击球后的收拍、机身角速度调整、腿部配置和落地准备方向。它不能决定击球时刻、球拍目标、出球方向或起跳计划。教师输出和 PPO 输出处于相同的 18 维 residual 空间，最终仍通过原有动作混合器和关节限制。

教师是冻结的 JIT residual actor，只在训练时加载；导出的 PPO `policy.pt` 不嵌入该外部教师。PPO 采样原始动作，环境执行混合动作，实际执行结果回传训练。因此 blend 0.4 是训练时的选择性执行辅助，不是永久的“教师代打”。

相关代码：

- `scripts/train_shared_target_ground64_success_retention_v1.py`
- `quadsmash/rl/posthit_exploration_v1.py`
- 外部教师权重（仅本机）：`reports/G07_full_racket_v5/recovery_assist_residual_trace_gated_posthit_policy_01/policy.pt`

### 3.5 持久恢复辅助

评估时不会调用上述外部 post-hit teacher，但当前导出的策略仍声明并使用基础控制器中的持久恢复辅助参数。它包括恢复局部时间、腿部偏置、滚转反馈、恢复 actor 增益和交接时间。

这和 `posthit_teacher_blend=0.4` 是不同的机制。配置中的 `recovery_actor_gain=0.4` 也不是 post-hit teacher 的 blend 权重；`recovery_blend_s=0.08` 是恢复交接的时间参数。

相关代码：

- `quadsmash/control/persistent_recovery_assistance_v1.py`
- `quadsmash/control/shared_target_assistance_v6.py`
- `scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py`

因此当前的“教师关闭”评估准确说是“关闭额外 post-hit residual teacher，但保留基础控制器和声明的持久恢复辅助”。后续会增加更严格的无额外恢复辅助对照，单独报告自主学习收益。

### 3.6 成功保留和 anchor

策略更新容易重新组合起跳、挥拍和落地动作，导致旧的完整成功被遗忘。当前训练把已经验证的成功轨迹按回合采样，加入 actor 的 anchor/拟合损失；它只约束已有成功状态的 residual，不改变物理任务判定。

当前 anchor 分支的关键参数：

```text
5120 environments
4 PPO updates
learning rate = 2e-5
steps per update = 64
anchor weight = 1.0
seed = 882097
post-hit teacher blend = 0.4
```

相关代码：

- `quadsmash/rl/success_retention_v1.py`
- `configs/g07_shared_target_ground64_attack_recovery_v6_anchor1.json`
- `scripts/train_shared_target_ground64_success_retention_v1.py`

## 4. 已验证结果

### 4.1 当前最好冻结结果

训练产物目录：

`reports/G07_full_racket_v5/transfer_blend04_env5120_02/`

冻结评估目录：

`reports/G07_full_racket_v5/transfer_blend04_frozen128_01/`

固定 128 例、5 m/s 主指标的逐回合统计：

| 指标 | 结果 |
|---|---:|
| 完整成功 | **20/128（15.625%）** |
| 空中合格触球 | 79/128 |
| 条件恢复 | 32 次 |
| 球侧合格（越网/落点相关） | 67 次 |
| 数值故障 | 0 |
| 重复固定面板结果 | seed 882072 与 882075 均为 20/128 |

原始强保留基线约为 17/128 完整成功、25 次条件恢复、56 次球侧合格。blend 0.4 分支提高了整体结果和恢复数量，但发生了旧成功丢失：旧成功中有 8 条未保留，同时新增 11 条，重叠 9 条。因此当前结果是改善而不是稳定解决。

原始 8 m/s 字段仍可能显示 `complete=0`；当前正式开发结论以重新计算的 5 m/s 完整事件链为准。

### 4.2 当前正在运行的 anchor1 分支

目录：

`reports/G07_full_racket_v5/transfer_blend04_anchor1_env5120_01/`

该分支已完成 4/4 更新并通过冻结评估：

- 5120 环境；
- 2,621,440 个训练控制环境步；
- 数值故障 0；
- 冻结 128 例的 5 m/s 完整成功：**15/128**；
- 空中合格触球：81/128；
- 条件恢复：27/81；
- 球侧合格：58；
- 该分支低于当前 20/128 基线，因此没有替换最佳策略。

训练过程中的 `air` 和 `complete` 仍然只是变化策略训练统计，最终选择以冻结 128 例为准。

## 5. 训练机制修订

审查 anchor1 后确认，4 次更新只覆盖每环境 5.12 秒，而完整事件链需要 8 秒回合末才能结算。后续训练把更新次数作为连续训练长度，评估只作为周期性快照，不在每个短周期末清空未完成回合。这样完整成功奖励可以跨 PPO rollout 边界真实回传。

训练入口新增可选参数：

```text
--posthit-teacher-recovery-phase
```

默认仍只在公开观测 `policy[:,773] > 0.5` 的击球后腾空阶段调用教师。打开该参数后，教师还会在公开观测 `policy[:,774] > 0.5` 的足端恢复阶段保持介入；它是独立诊断分支，元数据会记录门控版本，不会改变默认实验。恢复阶段扩展仍需通过实际冻结评估证明有效，不能仅凭介入次数判断。

当前连续训练分支：

```text
输出：reports/G07_full_racket_v5/transfer_blend04_continuous16_env5120_01
5120 环境，16 次更新，seed 882098
从 20/128 分支进行 actor transfer
post-hit teacher blend = 0.4
默认击球后门控，尚未打开恢复阶段扩展
```

## 5. 当前主要失败机制

失败是完整事件链没有在同一回合同时成立，主要集中在以下几类：

1. 已经空中触球，但出球速度或落点不够；
2. 出球和球侧结果合格，但击球后的机身角速度、腿部构型或触地状态不足，恢复失败；
3. 新 PPO 更新增加了某些新回球，却覆盖了原先有效的起跳/挥拍/落地组合；
4. 拍速、击球位置和落地可恢复性之间存在 trade-off，单独提高某项指标会损害另一项；
5. 评估中仍有持久恢复辅助，尚未证明没有任何额外恢复控制时的纯 residual PPO 能保持相同成功率。

数值故障不是当前瓶颈：重要训练和冻结评估均为 NF0。当前瓶颈是任务链组合稳定性、成功模式保留和有效击球覆盖范围。

## 6. 下一步实验顺序

### 第一步：完成当前 anchor1

不重启当前作业。完成 4 次更新后，保存 checkpoint、actor、训练统计和源代码快照。

### 第二步：固定 128 例复评

对 anchor1 的导出策略运行同一 128 例、同一 5 m/s 事件链，至少重复一个固定种子；同时保留旧 20/128 作为基线。

### 第三步：三种教师对照

用同一冻结策略分别评估：

1. 额外 post-hit teacher 开启；
2. 额外 post-hit teacher 关闭，但持久恢复辅助保留；
3. 额外 post-hit teacher 和持久恢复辅助都关闭。

这样可以分开执行辅助收益、学习收益和真正 residual PPO 自主收益。

### 第四步：短周期大规模 PPO

继续使用 5120 环境，但以 4–8 次更新为一个周期。每周期结束立刻冻结评估并保存三个检查点：

- 完整成功率最高；
- 出球和恢复平衡最好；
- 旧成功保留率最高。

不要只按训练 reward 或某一次更新的空中触球数选模型。

### 第五步：扩展成功保留集

把经过完整物理事件核验的新增 5 m/s 成功回合加入下一版成功保留清单，保留来源、随机种子、版本和动作接口信息。球侧失败但恢复成功的回合只能用于恢复学习，不能标为完整专家示范。

### 第六步：逐步减少教师依赖

在有效纠正片段上增加配对监督和未来介入需求预测，使策略提前改变起跳和挥拍，降低之后需要恢复教师的概率。只有在关闭额外教师后完整成功不下降，才扩大自主阶段。

## 7. 可复现实验入口

### 大规模训练

```bash
cd /home/autodrive/LZQW/unitree_smash_contact_recovery
CUDA_VISIBLE_DEVICES=0 /home/autodrive/Isaac/env_isaaclab/bin/python -u \
  scripts/train_shared_target_ground64_success_retention_v1.py \
  --output <output_dir> \
  --base-checkpoint <base_checkpoint> \
  --body-source-evaluation <body_eval> \
  --ground-source-evaluation <ground_eval> \
  --original-teacher-evaluation <teacher_eval> \
  --precision-receipt <precision_receipt> \
  --config configs/g07_shared_target_ground64_attack_recovery_v6_anchor1.json \
  --num-envs 5120 --updates 4 --seed 882097 \
  --initialize-actor <source_policy.pt> \
  --posthit-teacher-policy <posthit_teacher.pt> \
  --posthit-teacher-blend 0.4 --headless
```

### 冻结评估

使用：

```text
scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py
```

评估必须固定来球面板、目标区、5 m/s 主指标、恢复窗口、资产哈希和随机种子。每次结果都要保存 `job.json`、`metrics.json`、`episodes.json`、源代码哈希和 checkpoint SHA256。

## 8. Git 上传范围

本次 Git 提交只包含：

- 本技术路线 Markdown；
- 当前训练入口和评估入口；
- post-hit teacher、pre-hit prior、成功保留和持久恢复实现；
- 当前训练配置和必要测试；
- 与动作转移和诊断直接相关的源代码修改。

不上传：

- `policy.pt`、`checkpoint.pt` 和教师权重；
- 视频、轨迹、`.npz`、大型日志和 Isaac 运行目录；
- GPU ledger、临时状态文件和实验队列目录。

模型和视频通过本机报告目录保存，避免 Git 仓库和流量膨胀。

## 9. 当前结论

当前路线已经从“单独修改教师参数”转为“冻结基础控制器 + 全身 residual PPO + 选择性击球后恢复教师 + 成功保留”。blend 0.4 已经把冻结 5 m/s 成功率提高到 20/128，并改善了恢复，但还不能称为稳定完整扣杀，也不能称为完全自主 RL。

下一步的判断标准不是教师开启时某一次成功，而是：在相同固定 128 例上，策略关闭额外教师后仍能保留旧成功并增加新成功，同时没有数值故障和任务协议变化。

## 10. 审查后的连续训练对照结果

针对“完整奖励无法跨短作业回传”和“教师在触地阶段退出”的问题，实际运行了三组连续训练对照，均从当前 20/128 策略 actor transfer 开始，保持 5120 环境、5 m/s 主指标和相同固定面板。

| 分支 | 训练设置 | 训练期完整 5 m/s 回合 | 冻结 128 例完整成功 |
|---|---|---:|---:|
| `transfer_blend04_continuous16_env5120_01` | 连续 16 更新，默认击球后门控 | 250 | **8/128** |
| `transfer_blend04_continuous8_env5120_01` | 连续 8 更新，默认击球后门控 | 61 | **13/128** |
| `transfer_blend04_continuous8_recoveryphase_env5120_01` | 连续 8 更新，教师延伸到公开触地恢复阶段 | 71 | **17/128** |
| 当前最佳基线 `transfer_blend04_env5120_02` | 4 更新短周期分支 | — | **20/128** |

连续训练解决了完整终端奖励完全采不到的问题：16 更新分支在训练过程中真实采到 250 个完整回合，8 更新分支也采到 61 个。但 16 更新最终冻结性能退化到 8/128，说明后续 PPO 更新会破坏固定面板动作。8 更新保留得更好，但仍低于 20/128。

恢复阶段教师门控把冻结结果从 13/128 提高到 17/128，空中触球为 70、条件恢复为 27/70，说明延长教师到公开触地阶段有实际收益；它仍未超过 20/128，因此当前最佳策略保持不变。该结果支持“阶段覆盖需要改进”，也说明简单地把同一个教师延长到恢复阶段还不足以达到稳定完整扣杀。

这些结果共同确定了下一步方向：保留连续跨回合训练机制，让终端奖励真实进入学习；采用更短的冻结选择周期，避免 16 更新后的漂移；继续改进触地恢复教师的状态适配和交接，而不是直接把最终长跑权重当成最佳模型。
