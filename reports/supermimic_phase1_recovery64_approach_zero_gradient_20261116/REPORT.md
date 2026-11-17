# Recovery 64 → Approach 启动失败实验

## 运行来源

- 实验目录：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1`
- 启动代码提交：`38ef1bb7478561933383d8807c31300c525023d5`
- 初始化策略 SHA256：`8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- Recovery checkpoint SHA256：`aaa814534fcb54c59a6d36a19efbbf7648eb56d04fa8bb18d5d66bb92bc6ec26`
- TensorBoard：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/tensorboard`

## 结果

Recovery 阶段完成 **64/64** 次 actor 更新；Approach 阶段启动后，第一个候选被判定为“accepted update”，但实际没有任何可训练参数变化，程序按保护检查主动终止：

```text
RuntimeError: approach accepted update had zero trainable parameter change; no effective gradient
```

因此：

- Recovery 更新：64
- Approach 更新：0
- World Model 更新：69
- Critic 更新：3224
- 完整 native 训练流回合：130
- 训练流 native return 均值：`-12.1638567289`
- 初始化基线 return：`-12.52678435`
- 训练流均值差：约 `+0.36293`
- 这不是固定 128 环境正式评价，不能写成新的成功率或 complete5 成绩。

## Recovery 阶段诊断

Recovery checkpoint 已保存：

- `recovery_candidate_0064.pt`
- `cycle_01_recovery_candidate_0064.pt`
- `recovery_checkpoint_0016.pt`
- `recovery_checkpoint_0032.pt`
- `recovery_checkpoint_0048.pt`
- `recovery_checkpoint_0064.pt`

最终进度中的关键执行变化仍较小：相对原生参考的最终归一化命令变化约 `0.0023557`，关节目标变化约 `0.0024669 rad`，没有显示超出执行预算。基础策略和冻结分支保持不变。

训练流 return 略高于初始化基线，但由于策略在训练过程中不断变化、回合不是冻结策略正式评价，不能据此证明 Recovery 已提升固定 128 面板。

## Approach 失败原因

失败发生在 actor 梯度保护检查，而不是 Isaac 或 GPU 启动错误。Approach 阶段没有产生有效的可训练参数变化，继续接受更新会使“更新计数”与真实策略变化不一致，因此程序正确停止。

这说明当前 Approach 起点/门控/想象链没有给 Approach head 提供有效梯度信号。下一步应先定位：

1. Approach-gated 起点是否真的覆盖有效接近状态；
2. imagination 中 approach gate 是否保持激活；
3. actor 目标对 approach 参数的梯度是否被执行预览、离散事件或状态重建截断；
4. 是否把 Recovery-only 的采样分布错误地带入 Approach 更新。

不能通过关闭零变化检查、强行增加学习率或把一次无变化算作成功更新来解决。

## 结论

这是一次 **Recovery 64 次完成、Approach 首次更新因零有效梯度而停止** 的诊断运行。Recovery 产生了可保存的中间候选和略高的训练流 return，但没有新的正式 128 环境成绩；Approach 尚未真正学习。