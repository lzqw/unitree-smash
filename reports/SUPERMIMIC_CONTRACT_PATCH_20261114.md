# SuperMimic PPO 热启动执行合同修复（2026-11-14）

本轮只完成代码修复、CPU 正确性检查和版本记录，**没有启动训练，也没有启动正式 128 环境评估**。因此本轮没有新的性能成绩；旧实验结果保持原样。

## 已修复

### 1. 按实际 schema 解码动作

`SuperMimicEnvV1` 现在区分两种动作合同：

- `jump-event-v1`：保留 JumpEnv 的阶段映射（腿部 0.25/0.55、机械臂 0.12）。
- `full-racket-residual-v4`：将残差 mixer 产生的完整 18 关节归一化命令，严格按 `FullRacketEnvV4` 的公式还原到关节目标：

  `q = q_min + 0.5 * (a + 1) * (q_max - q_min)`

full-racket 分支不再经过 JumpEnv 的阶段偏移。请求动作与 mixer/执行器限幅后的最终动作不同属于执行效果，不能直接记为映射误差。纯 Torch 解码器位于 `quadsmash/rl/action_contract.py`，便于不启动 Isaac 的检查。

### 2. 移除无效的零更新验收含义

训练入口不再把 `trainer.action()` 与同一个 `trainer.actor()` 的比较写成执行合同验收。当前记录为 `native_saved_state_full_chain_replay_pending`。

真正的验收需要在相同保存状态、观测历史和控制器记忆下，分别重放旧完整执行链和新入口，覆盖准备、起跳、击球和恢复，并比较最终 18 关节目标。该原生 saved-state replay 尚未在本轮完成；没有用零对零结果替代它。

### 3. 对齐 SuperMimic 更新目标

- 世界模型训练窗口默认改为 16 控制步，即 **0.16 s**（100 Hz）。
- GPU 想象窗口默认改为 96 控制步，即 **0.96 s**。
- 想象 actor 目标改为 rollout 起点的回报，不再对所有后缀回报取平均。
- 增加 critic bootstrap、`gamma=0.998`、`gae_lambda=0.95` 的反向 GAE。
- actor 更新冻结 world-model 和 critic 参数，但保留状态输入梯度；critic 使用 detach 后的想象状态重新前向。
- 记录 actor 梯度范数、参数变化范数和参数范数，避免只看 loss 或 `log_std`。

### 4. 终止记录不再把缺失事件伪装成 False

评估脚本在消费终止记录时，对 `head_contact`、`body_ground_seen`、`rally_fault` 缺失字段保存为 `null`，并记录来源、终止时间和记录完整性。延迟 reset 的 live snapshot 不再将不可用的头部接触事件写成 `False`。球侧统计要求 `rally_fault is False`，缺失事件不会静默算作无故障。

## 初始化来源审计

当前来源文件：`artifacts/ppo_full_racket_residual_v4/policy.pt`。

- schema：`full-racket-residual-ppo-v4`
- SHA256：`772608a1c42ec78c09cdf3bd8b7bfe3fed1738d1273cde9d09610022553aa21b`
- native execution：`true`
- formal acceptance：`false`
- 已记录的源训练元数据：32 次更新、1,572,864 control-env steps、512 环境、base policy SHA256 `51a8b19b17e19e60bd7482c7bcb7ed1d4e8b7f50b9317cac3c49007089e8b62d`、actor JIT SHA256 `30ae7361decaa17827b32f91153a5626bf7e6f3e9a39bc118c4a67ec17de88b4`。

`formal_acceptance=false` 不被自动视为无效，但本轮也没有仅凭文件名或维度宣称它是历史最佳。源能力尚未在本轮重新原生复验。

## CPU 检查

| 检查 | 结果 |
|---|---|
| full-racket 归一化解码与原公式最大绝对误差 | `0.0` |
| mixer 请求/最终命令有限且有界 | 通过 |
| 请求与最终命令差异的解释 | 标记为执行效果，不是映射错误 |
| jump-event 阶段解码 | phases `0.0, 0.4, 0.8, 3.0` 通过 |
| world-model + imagined BPTT + GAE 小批量 CPU 检查 | 通过；actor 梯度范数与参数变化均为有限非零 |
| 原生旧完整执行链 saved-state replay | **待完成** |
| 正式 128 环境评估 | **本轮未运行** |

详细机器可读结果：

- [`action_contract_cpu.json`](supermimic_contract_patch_20261114/action_contract_cpu.json)
- [`gae_cpu.json`](supermimic_contract_patch_20261114/gae_cpu.json)
- [`source_checkpoint_audit.json`](supermimic_contract_patch_20261114/source_checkpoint_audit.json)

## 结果边界

本轮没有训练、没有正式评估、没有修改物理环境、没有改变 128 环境评分、没有改变原生控制周期。此前的训练/评估数字仍属于各自提交和配置，不在本报告中重算或拼接。下一步只有在原生 saved-state 全执行链复验通过后，才进入新的持续训练或正式冻结评估。
