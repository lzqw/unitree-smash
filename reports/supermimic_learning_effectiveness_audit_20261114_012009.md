# SuperMimic 当前学习有效性审计

审计日期：2026-11-14（Asia/Shanghai）

## 结论

当前 SuperMimic 训练进程仍在真实 Isaac 原生环境中采样并更新 Actor、Critic 和世界模型，但现有证据只能证明“学习链路在运行”，不能证明羽毛球任务成功率已经提高。

当前没有新的冻结 128 环境评估结果。最近可审计的正式评估仍为：

- 合格空中拍线触球：0/128
- complete5：0/128

因此本轮不能宣称击球能力提升。

## 当前运行

输出目录：

`reports/supermimic_formal_w0p32_i0p96_20261113_194828/`

启动配置：

- 512 个并行环境
- `world_window_s=0.32` 秒，32 个控制步
- `imagination_s=0.96` 秒，96 个控制步
- 100 Hz 控制周期
- GPU `cuda:0`

截至本次审计，训练历史约到第 2809 次迭代：

- 原生交互步数：约 1,438,208
- Actor 更新：约 2,778 次
- Critic 更新：约 55,560 次
- 世界模型更新：约 8,334 次
- Replay：约 249,856 条转移

训练进程仍在运行，未停止或修改。

## 学习信号检查

### 支持学习链路正在更新的证据

- checkpoint 500 到 checkpoint 2500 的 Actor 参数相对变化约 4.9%。
- 世界模型 loss 从早期约 32 降至近期约 2.5–3.0。
- Replay 中存在真实原生奖励，说明环境交互和奖励写入正在发生。

### 尚不能证明任务学习有效的证据

- imagined return 最近仍在约 -3 到 -14 之间波动，没有稳定上升趋势。
- Actor 梯度存在明显尖峰，历史最大约 `2.2e7`，最近窗口仍有数百量级波动。
- Critic loss 仍有明显波动。
- Replay 奖励均值约 -0.07，正奖励比例约 0.2%，高奖励事件极少。
- 当前训练历史没有记录 `native_qualified_string_contact`、`native_complete5` 等任务事件，无法从训练曲线判断是否出现成功击球。

## 证据边界

世界模型 loss 下降不等于机器人学会击球；Actor 参数变化也不等于任务能力提高。必须冻结 checkpoint，使用原 128 环境、原来球和完整 8 秒协议进行评估，才能得到任务成功率。

当前应将本运行标记为：

> 原生交互和模型更新正常；任务级有效性尚未验证，策略尚未证明收敛。

## 建议的下一项检查

冻结一个已保存 checkpoint 做正式 128 环境评估，并在后续训练日志中增加：

```text
native_qualified_string_contact
native_ball_side_complete
native_recovery_success
native_complete5
```

在这些指标出现之前，不应只依据 imagined return、world loss 或 Actor loss 选择最佳策略。
