# SuperMimic 初始化策略来源与历史强策略核对

日期：2026-11-14  
代码提交：`bdeec2a`  
分支：`supermimic-structured-history-event-metrics-20261114`

## 结论

当前 SuperMimic 诊断训练使用的初始化 checkpoint 是旧的
`772608a1c42ec78c09cdf3bd8b7bfe3fed1738d1273cde9d09610022553aa21b`。
该权重在当前独立诊断协议中记录为 `complete5=0/128`，后续零更新
SuperMimic 评估为 `2/128`，锚定续训后为 `7/128`。

历史报告中约 `27/128` 的强策略并不是这份权重。该结果使用的真实
checkpoint 是：

```text
/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/
  frozen_v2_smooth_approach16_env5120_01/policy.pt
```

其 SHA-256 为：

```text
8f5109fe218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399
```

对应历史 `metrics.json` 的固定 128 案例诊断为：

| 指标 | 数值 |
|---|---:|
| 合格空中拍线触球 | 74/128 |
| 条件恢复 | 34/74，45.95% |
| 实际球拍接触 | 110/128 |
| 越网 | 71/128 |
| 目标落点 | 64/128 |
| complete5 | 27/128，21.09% |
| 数值故障 | 0 |

## 兼容性检查

真实强策略文件的顶层 schema 为：

```text
g07-shared-target-residual-ppo-v1
```

它包含 `residual_actor_jit`、`base_controller`、18 维参数以及完整的
原生执行元数据，但不是当前 loader 接受的
`full-racket-residual-ppo-v4` schema。当前
`quadsmash/rl/supermimic_v1.py::load_ppo_initialized_actor` 的残差入口
只对后者进行严格加载；因此不能直接把 `8f5109…` 当作当前
SuperMimic 初始化权重，也不能用 `strict=False` 或只按 794/18 维度强行
装载。

当前 loader 的判断实际上是：

```text
schema == full-racket-residual-ppo-v4
    -> residual_actor_jit -> ResidualPPOInitializedActor
其他 schema
    -> 仅尝试旧 PPO model_state_dict 路径
```

所以把历史强策略复制到当前训练命令的 `--init-policy-checkpoint` 位置，
在未完成执行合同适配前会被拒绝或造成错误解释。当前 SuperMimic 的
`7/128` 只能解释为“从旧弱 source 的同入口相对改进”，不能称为
相对历史 27/128 强策略的学习提升。

## 当前已完成的结果

同一 SuperMimic evaluator 下，旧 source 的 reset 与锚定续训为：

| checkpoint | complete5 | 合格空中拍线触球 | 球侧除恢复外完整 |
|---|---:|---:|---:|
| zero-update anchor reset | 2/128 | 22/128 | 3/128 |
| anchor=5, actor lr=1e-6 续训 | 7/128 | 22/128 | 7/128 |

该续训真实完成了 native interaction、world-model 更新、imagined BPTT
和 actor/critic 更新，但它没有使用历史 27/128 checkpoint，且正式评估中
仍有大量数值故障。因此该结果保留为诊断证据，不替换历史最佳策略。

## 下一步必要工作

1. 先为 `g07-shared-target-residual-ppo-v1` 建立显式、严格的适配器，核对
   joint order、观测归一化、base controller、mixer、动作限幅和 PhysX
   最终命令映射。
2. 在零更新条件下，用相同原生执行链复现其 27/128 能力；复现失败时
   不启动 SuperMimic 训练，先修执行合同或明确评估协议差异。
3. 只有零更新复现通过后，才从该强策略创建新的 SuperMimic anchor，
   再比较训练前后同一评估协议的变化。

本报告不修改历史 checkpoint、不复制二进制权重到本仓库，也没有启动
训练或评估。
