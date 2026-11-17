# SuperMimic 当前训练与评估总报告（2026-11-17）

## 固定基线

当前接受的初始化策略为：

- native return：约 `-12.52678435`
- 合格空中拍线触球：`74/128`
- 球侧完整条件：`53/128`
- 条件恢复：`34/74`
- complete5：`27/128`

初始化策略 SHA256：`8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`。

## 各阶段结果

| 阶段 | 原生证据 | 结论 |
|---|---:|---|
| 早期继续训练 | native return 约 `-28.65` | 明显退化，停止 |
| v3 | 5 个完整回合均值约 `-16.82` | 恢复仍失败，身体触地 |
| v4 | 8 个完整回合均值约 `-29.32` | 继续退化，主要为 body-ground |
| Recovery gate 小规模候选 | `36/74` 恢复，`27/128` complete5 | 恢复增加2个，完整成功未增加 |
| Recovery 64 次运行 | 64 次更新完成 | Approach 首次更新零梯度，停止 |

## Recovery gate 候选正式评估

评估目录：`/home/autodrive/LZQW/eval_posthit_gate_recovery0008_20261117_0930`。

| 指标 | 基线 | Recovery gate 候选 |
|---|---:|---:|
| 合格空中触球 | 74/128 | **74/128** |
| 球侧完整条件 | 53/128 | **53/128** |
| 条件恢复 | 34/74 | **36/74** |
| complete5 | 27/128 | **27/128** |
| 数值故障 | 0 | 0 |
| body-ground 记录 | 未作为基线指标 | 63 |
| MPC/教师介入 | 0 | 0 |

该评估完整运行了128环境、800 tick、GPU PhysX，进程正常退出。`physical_recovery_success` 为37，但正式条件恢复计数为36/74；两者必须分开报告。

## Recovery 64 运行

运行目录：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1`。

- Recovery 更新：`64/64`
- Approach 更新：`0`
- World Model 更新：69
- Critic 更新：3224
- 训练流已完成回合：130
- 混合训练流 native return 均值：`-12.1638567289`
- 相对初始化训练流均值差：约 `+0.36293`
- Recovery checkpoint：`recovery_candidate_0064.pt`
- Recovery checkpoint SHA256：`aaa814534fcb54c59a6d36a19efbbf7648eb56d04fa8bb18d5d66bb92bc6ec26`（见 `metadata.json` 获取真实完整哈希）

训练流 return 不是固定128环境冻结策略正式成绩，不能直接替代正式评价。

## Approach 为什么没有开始

程序已经切换到 Approach 参数白名单，但第一项提案后发现所有 Approach 参数没有发生有效变化，触发保护：

```text
RuntimeError: approach accepted update had zero trainable parameter change; no effective gradient
```

根因是起点筛选使用原始观测字段 `observation[:,771:773]`，而真正的 JIT actor 使用 `module.gate(observation)`。两者不一致：

- tick 224：实际 gate 最大约 `1.2e-7`
- tick 240：实际 gate 全为 `0`
- tick 256：只有少量 lane 激活
- tick 272：多数 lane 激活

因此名义上被选为 Approach 起点的状态，实际可能没有 Approach head 的动作权限，导致 actor 梯度为零。Recovery head能够完成更新，是因为其使用了独立的接触后门控。

## 当前工程状态

- GPU PhysX 和128环境运行正常。
- Recovery gate 修复已通过 CPU 合同测试并产生正式评估收益。
- Approach 尚未完成有效更新。
- World Model/imagined return 仍不能替代 native episode return。
- TensorBoard：
  - 6005：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/tensorboard`
  - 6006：`/home/autodrive/LZQW/supermimic_posthit_gate_test_20261117_085000/tensorboard`

## 下一步最小修复

1. 用实际 `module.gate(observation)` 筛选 Approach 起点。
2. 记录每次想象中的 gate 均值、最大值、激活 lane 数和激活步数。
3. 没有有效 Approach gate 时换起点，不计作失败更新。
4. 通过有效 gate 梯度检查后，再进行 Approach 小预算训练。
5. 完成 Approach 后重新进行一次固定128环境正式评估。

本报告没有把 Recovery 训练流 return 写成新的正式成功率，也没有把 Approach 零梯度运行算作有效训练。
