# SuperMimic 学习有效性检查（2026-11-14）

## 结论

本轮没有证明 SuperMimic 已经学到可部署的羽毛球技能。强锚定续训能够真实采样并完成 world model、critic、actor 更新；在当前 SuperMimic 评估入口下，训练后 checkpoint 为 **7/128 complete5**，同一入口的零 actor 更新 reset checkpoint 为 **2/128**。同一固定评估中新增了 TRAIN35、41、54、61、81，保留 TRAIN9、25。这个相对提升是有效的学习信号，但仍不能称为最终能力突破：它仍低于后续原生 PPO/控制路线报告中的约27/128，而且数值故障非常高。历史上该 source checkpoint 的独立 TRAIN 诊断曾记录 complete=0/128，说明“27/128”不能直接作为这个 source actor 的零更新基线；当前报告因此同时保留 reset2 和 source历史0。

## 训练链是否真的运行

强锚定续训目录：

`reports/supermimic_anchor5_lr1e6_resume3_20261114_0910`

- GPU：`cuda:0`，Isaac/PhysX 原生交互。
- 累计 native steps：459,264。
- 本次从 checkpoint 继续到 iteration 897；本段 actor 更新 19 次。
- 累计 world updates：802；critic updates：1,960。
- actor gradient norm：约 4.3--6.9。
- policy anchor penalty：从约 `6e-7` 增至约 `3.8e-5`。
- native reward mean：约 `-0.0114`，positive fraction 为 0。
- 本段只推进了约 0.2 秒的环境时间，未完成新的 8 秒原生回合，因此本段日志中的事件计数不能作为任务成功率。

第一次续训因为人工构造的 reset checkpoint 中 `actor_opt={}` 导致 `KeyError: 'param_groups'`，未执行任何 tick。修复为空 optimizer sentinel 时保留新建 optimizer 后，CPU 测试 **35 passed**，随后续训正常进入首步采样和更新。

## 同入口冻结评估

| checkpoint | 合格空中拍线触球 | 球侧除恢复外完整 | complete5 | recovery_success 字段 | 数值故障 |
|---|---:|---:|---:|---:|---:|
| zero-update reset actor | 22/128 | 3/128 | **2/128** | 67 | 39,993 |
| 强锚定续训后 actor | 22/128 | 7/128 | **7/128** | 69 | 37,462 |

两次评估均为 128 环境、800 控制 tick、8 秒、`cuda:0` GPU Dynamics，均没有 MPC 或 teacher 接管。续训 checkpoint 的正式评估耗时约 1222 秒；reset checkpoint 约 1202 秒。

## actor 漂移

在 checkpoint replay 的 4096 个观测上比较原始 PPO residual actor 与续训后的 actor：

- action RMS（原始）：约 0.01545；
- action RMS（续训）：约 0.01536；
- 差值 RMS：约 0.000226；
- 差值绝对值 p95：约 0.000450；
- 最大绝对差：约 0.00177。

因此，这次失败不是大幅参数漂移；更可能是任务对微小动作变化高度敏感，同时 world-model 想象回报仍为负且存在明显模型误差。此前 3.2 秒长时域诊断已显示 autoregressive state MSE 随 horizon 快速增长，本轮没有证据推翻该限制。

## 解释边界与下一步

历史原生 PPO 报告约 27/128 的结果来自另一套已验证的正式原生评估协议，不能把本次 SuperMimic evaluator 的 2/128 reset 结果当成那个策略的能力本身。当前最先需要做的是统一 SuperMimic 评估与原生 PPO 的动作执行、事件记录和数值故障口径；在此之前继续增加 imagined actor 更新会把“评估合同差异”和“学习收益”混在一起。

当前候选不应替换历史最佳策略。保留该 checkpoint 仅作诊断：它证明训练链可以运行，但**尚未证明学习有效或提高了真实任务能力**。
