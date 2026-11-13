# SuperMimic actor-anchor stability run

日期：2026-11-14

## 触发原因

结构化历史模型第一轮在约第81--85轮出现原生奖励从约 `-0.012` 短时下降到约 `-0.95`，而 imagined return 仍为正，说明世界模型乐观且 Actor 开始偏离原生可行行为。该轮在第87轮停止，未进入完整击球评估。

## 修正

提交：`a7d0eab`

- 固定初始化 PPO actor 作为 reference actor；
- imagined rollout 增加动作锚定损失；
- Actor 学习率 `5e-5`；
- world model/Critic 学习率保持 `2e-4`；
- 锚定系数 `0.05`；
- 保留结构化历史重建和原生事件日志。

CPU 测试：`35 passed`。

## 新运行

输出目录：

`reports/supermimic_structured_anchor_w0p32_i0p96_20261114_0228/`

启动参数为 512 环境、0.32 秒世界模型窗口、0.96 秒想象窗口、100 Hz 控制。

截至第64轮：

- 原生步数：32,768；
- world loss：约 `0.044`；
- imagined return：约 `6.89`，仍需原生验证；
- Actor grad norm：约 `3.99`；
- 原生奖励均值：约 `-0.0116`，尚未出现前一轮的早期大幅恶化；
- 原生事件：尚未到达完整击球时刻，事件计数仍为0。

## 结论边界

锚定版本目前只显示早期稳定性改善，尚无击球或 complete5 结果。必须继续运行到完整8秒回合并冻结评估，才能判断是否提升任务成功率。
