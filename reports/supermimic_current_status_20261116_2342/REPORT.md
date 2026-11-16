# SuperMimic 当前训练状态与数据闭环审计

- 记录时间：2026-11-16（北京时间）
- 代码提交：`38ef1bb7478561933383d8807c31300c525023d5`
- 分支：`feat/supermimic-native-physical-loop-20261114`
- 当前运行目录：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452`
- 当前周期：`cycle_1`
- 运行状态：进程仍在运行，未完成正式评估

## 当前进度

本次运行使用 128 个 Isaac/PhysX 环境，控制周期为 0.01 秒。当前仍处于 Recovery block：

| 指标 | 当前值 |
|---|---:|
| Recovery actor updates | 22 / 64 |
| accepted updates | 22 |
| rejected / backtracked | 0 / 0 |
| native ticks | 433 |
| native transitions | 55,424 |
| completed native episodes | 0 |
| world-model updates | 27 |
| critic updates | 1,504 |
| imagination steps（最近一次） | 64 |
| imagination window | 0.64 s |
| world-model window | 0.16 s |
| native reward mean（最近 tick） | -0.03297 |
| imagined return mean（最近批次） | -15.52 |
| world loss（最近批次） | 85.31 |
| critic loss（最近批次） | 0.0528 |
| peak GPU memory recorded | 1.39 GB（训练进程记录值） |

`native_reward_mean` 是最近控制 tick 的即时奖励均值，不是完整 8 秒 episode return。由于当前只有 433/800 个控制 tick，还没有完整回合，因此 `native_return_mean` 仍为空，不能据此判断任务成功率或原生回报是否提高。

## 当前环境实际在做什么

训练器处于 Recovery 参数块，但 128 个环境仍从地面执行完整任务。前半段准备、起跳、接近、挥拍和击球由冻结的初始化策略及冻结的 Approach/base 分支完成；恢复门控激活后，Recovery head 才能改变最终命令。当前不是只运行落地测试，也没有在线 MPC、教师接管或 BC 示范。

不同 lane 的物理阶段可以不同。当前日志没有保存每个 lane 的实时阶段直方图，因此不能从现有记录精确报告有多少环境正在准备、起跳、击球或恢复。

## world model 数据来源与匹配性

world model 当前接收两类真实数据：

1. 初始化策略 `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399` 生成的原生 archive；
2. 当前训练策略在线采集的原生 transition。

当前 imagination 日志仍显示 `imagined_start_source=initialization_archive`，说明想象起点目前主要来自初始化 archive。在线数据正在写入 `cycle_1/native_transitions`，但当前尚无完成的 8 秒回合，实时数据主要还是未完成 episode 的中间片段。

因此，world model 对初始化策略附近的动力学覆盖较好，但对新的 Recovery 增量动作仅部分匹配。它学习的是最终执行命令到下一状态的转移，不能读取未来真实动作或未来事件标签。

## 为什么必须加入 Recovery 执行后的失败数据

Recovery 更新会改变球拍分离后的足端、身体和机械臂动作。若 world model 只看到初始化策略成功或失败前缀，它就无法准确学习新 Recovery 动作导致的：

- 首次触地未建立承载；
- 触地后支撑丢失；
- 头部、躯干或髋部碰撞；
- 侧翻、滑动和恢复超时；
- 稳定窗口未达到。

这些失败 transition 应作为真实动力学和事件预测数据，而不是作为 actor 的正确动作示范。每条失败轨迹应在 reset 前保存失败前状态、最终执行命令、下一状态、事件类型、terminal 标志、episode_id、policy_version 和执行记忆，不能跨 reset 拼接。

Replay 应同时保留成功与失败，按阶段和失败类型分桶采样。建议先使用约 50% 普通窗口、25% 成功恢复窗口、25% 恢复失败/碰撞窗口，并监控各类失败是否被单一头部碰撞样本淹没。当前代码已经有阶段桶和事件窗口机制，但本次运行尚未积累足够的新 Recovery 失败回合来验证覆盖。

## 现阶段结论

- 训练程序和 GPU 原生交互仍在运行；没有异常退出。
- Recovery 参数确实发生变化，基础策略保持冻结。
- world model、critic 和 imagination 都在更新，但 world loss 与 imagined return 仍有较大波动。
- 尚未产生完整 native episode return，因此没有新的任务性能证据。
- 0.16 秒 world-model 窗口适合短期执行响应预测，但无法覆盖完整 8 秒羽毛球任务；必须依赖持续真实 transition 更新，并在后续增加新 Recovery 成功/失败状态的覆盖。
- 下一步优先级应是确认实时 replay 中真实 Recovery 阶段和失败事件的数量，再判断是否提高 world-model 更新比，而不是直接延长 imagination 时域或盲目增加 actor 更新。

## 证据位置

- 训练进度：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1/progress.json`
- 训练日志：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1.log`
- 原生 transition：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1/native_transitions/`
- TensorBoard：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/tensorboard/`
- 初始化 archive manifest：`/home/autodrive/LZQW/supermimic_phase1_resume_20261116_224452/cycle_1/native_transitions/manifest.json`

本报告只记录当前状态和数据闭环分析，没有修改训练算法、没有停止进程、没有启动新的评估。
