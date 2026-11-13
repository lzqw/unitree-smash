# SuperMimic 修正与结果报告

日期：2026-11-13  
分支：`baseline-9cb7983-restore`

## 结论

本轮已经完成 SuperMimic 学习链、任务周期和事件生命周期的代码修正，并完成 GPU smoke、GPU replay/world-model/想象更新检查以及一次完整 128 环境、8 秒正式评估。正式评估正常跑完 800/800 控制 tick，但当前使用的初始化 checkpoint 不属于完整 Go2+PiPER+球拍全身任务的有效 PPO 策略，因此得到的 0/128 只能标记为**初始化合同失配结果**，不能作为 SuperMimic 在本任务上的算法上限。

下一步必须先选定与 `full-racket-v4-r3` 任务、794 维观测、18 关节动作语义完全一致的 PPO checkpoint，做零更新回归；在零更新回归恢复到有效击球基线之前，不应继续长时间 SuperMimic 训练。

## 已修正内容

### 1. 8 秒回合与训练生命周期

- 正式任务为 100 Hz、8 秒、800 个控制 tick。
- 完整任务奖励只在真实回合末结算，不在 rollout 截断处提前支付。
- `SuperMimicEnvV1._get_rewards()` 恢复 `_episode_has_stepped` 生命周期标记。
- 完整成功采用当前 5 m/s 任务合同 (`complete5`)；不再把历史 8 m/s 字段混入当前主指标。
- frame fault、body fault 等一次性事件改为锁存，避免每个 tick 重复扣分。

### 2. SuperMimic 学习链

- actor 可选用严格检查的原生 PPO actor 热启动；不使用 PPO clipped objective 更新 actor。
- actor 输出保持原生 PPO 的 raw Gaussian 语义，由环境端执行限幅。
- actor 更新门控为至少 `102400` 个原生 transition（128 环境完整 8 秒面板），并要求已经采到完整环境回合记录。
- resume 会恢复累计 native steps，避免断点续训时 warmup 计数归零。
- replay 按环境 stream 与 episode 链接；窗口不会跨环境、跨回合或跨 terminal 拼接。
- 覆盖的 rare transition 使用独立 archive，避免只在 ring buffer 当前槽位上打布尔标记。
- world model 使用自回归多步预测；terminal 使用 BCEWithLogitsLoss；survival mask 使用真实 terminal。
- imagination 阶段冻结 world-model 和 critic 参数但保留对输入状态的梯度；actor end-value bootstrap 不再 detach；critic 更新重新对 detached imagined states 前向。
- world-model 和 actor 更新使用同一 GPU 上的批量 Torch 张量。

### 3. 评估器

- 正式循环固定为 800 tick，不再执行第 801 个物理步。
- 回合末优先读取真实 terminal observation；缺少已 flush rally record 时使用 terminal snapshot，不额外 reset/step。
- 输出 `complete5`，并保留 qualified airborne contact、球侧条件、恢复和数值故障字段。

## 已完成验证

| 检查 | 结果 |
|---|---|
| Python 编译 | 通过 |
| GPU replay window | 通过，窗口不跨 stream/episode/terminal |
| GPU world-model fit | 通过，loss 有限 |
| GPU imagination actor/critic update | 通过，loss 有限，梯度可回传 |
| 8 环境短训练 | 4 iterations 完成，actor 按 warmup 保持冻结 |
| 128 环境 smoke | 2 iterations 完成，256 native steps，无 actor 更新 |
| 正式评估 | 128 环境、800/800 tick、GPU PhysX，完整运行 |

## 当前正式结果

结果文件：`reports/supermimic_ppo_init_zero_fixed_eval_20261113/metrics.json`

启动的零更新 checkpoint：`reports/supermimic_ppo_init_zero_fixed_20261113/checkpoint.pt`

| 指标 | 本轮结果 |
|---|---:|
| 合格空中拍线触球 | 0/128（0%） |
| 除恢复外球侧完整条件 | 0/128（0%） |
| 恢复成功 | 0/128（0%） |
| `complete5` | 0/128（0%） |
| 数值故障 | 7041 |
| MPC 介入 | 0 |
| teacher 介入 | 0 |
| 运行时长 | 1222.35 s |
| GPU dynamics | `cuda:0` |
| GPU broadphase | 128000 次，CPU exact query 0 |

评估完成，环境在 800 tick 时全部结束；但数值故障和 0/128 说明该 actor 没有复现完整任务的有效 PPO 行为。

## 初始化 checkpoint 审计

当前尝试的来源是：

```text
reports/G07/train_high_attack_all01/checkpoint.pt
sha256=7e3f7cb6dc5d0c70962d4ec6f72f587955a425ecb9ff8db02bbf5077d0db6510
```

它的文件接口表面上是 794 维、18 关节、`jump-event-v1`，但其训练记录来自 `BallEnv`/high-arm 训练分支，并非当前 `FullRacketEnvV4` 的正式全场 PPO 策略。零更新评估 0/128、7041 次数值故障已经证明：**仅凭维度和 action_family 一致，不能把它当作完整任务初始化策略。**

在相邻原始工程目录中还存在完整球拍 PPO checkpoint，但其 action family 是 `bounded-residual-joint-targets-full-racket-v4`，依赖冻结 base controller 和 residual contract；当前 SuperMimic 入口是绝对 `jump-event-v1` 目标，不能直接 strict-load 或静默转换。后续需先实现显式的 residual-to-absolute adapter，或找到同一 action contract 的 full-racket PPO checkpoint，然后重新做零更新最终命令回归。

## 训练规模与回合长度判断

当前 8 秒任务不适合在只有 4 秒/环境的训练作业里开放 actor。`128 env × 800 ticks = 102400 native transitions` 是首次允许 actor 更新的最低门槛；世界模型可以在此前拟合，但 actor 必须先看到真实击球、恢复和回合末 terminal 的完整分布。

事件奖励应保持一次性语义，连续风险/能量项才乘以 `dt`。当前修正已避免 frame/body fault 每 tick 重复扣分，并使用 `complete5` 与正式 5 m/s 任务协议一致。仍需在有效 full-racket PPO 初始化后监控 reward 分布、terminal 计数、world-model event accuracy 和 imagined/native return 差距。

## 结论与下一步

1. 保留本报告中的 0/128 作为初始化失配诊断，不覆盖历史完整任务基线。
2. 不从 `train_high_attack_all01` 继续长时间训练。
3. 找到或构造与 `bounded-residual-joint-targets-full-racket-v4` 一致的显式 adapter；adapter 必须保存 base policy、normalizer、joint order、scale、限幅和最终命令映射。
4. 对 adapter 做零更新最终命令等价检查，再进行完整 128 环境零更新评估。
5. 只有零更新恢复到有效 PPO 行为后，才按 `102400 native steps` warmup、GPU replay、多步 world model 和 imagination BPTT 重新收集数据并训练。

本报告不把 0/128 解释为算法最终失败，也不把 GPU 运行通过解释为任务能力成立；当前阻塞点是初始化策略合同和正式任务环境之间的可追溯一致性。
