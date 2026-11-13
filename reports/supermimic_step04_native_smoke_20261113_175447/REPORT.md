# SuperMimic v2 原生 smoke 调试报告

本轮只执行了一次极小规模原生 smoke：2 个环境、20 个控制迭代、GPU PhysX。没有启动正式评估，也没有进行大规模训练。

## 结果

- 目标提交：`a6294990ce99444975b1820050d3a4b024b1d4e7`
- 临时 worktree：`/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290`
- Isaac/PhysX：完成 AppLauncher 启动和场景创建
- 训练进程退出码：0
- 实际训练迭代：0
- replay 写入：0
- world model 更新：0
- actor 更新：0
- training manifest/status：未生成
- 正式成功率：未测量，不应填零

## 首个问题

日志在场景创建后出现：

```text
SimulationApp._wait_for_viewport: Exception during viewport wait: No module named 'omni.kit.viewport'
```

因此当前阻塞点在 Isaac AppLauncher/viewport 初始化阶段，训练脚本没有进入 `main()` 中创建输出目录和环境循环的后续步骤。这个问题不能归因于策略不更新、replay 没有有效数据或任务成功率低，因为这些阶段尚未执行。

完整日志见 `native_smoke.log`，退出码见 `exit_code.txt`。本轮没有重试、没有修改代码、没有调整时域、没有启动其他训练或评估。
