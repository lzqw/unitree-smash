# 已通过 G04 的自主策略接口

最佳原生权重：`artifacts/G04/policy.pt`，SHA256
`cd9119ad85d4ac45a76d9c0014058e4edfbcc4ded47317893e55419781c8d980`。
方法为教师搜索、BC 与 DAgger；正式双面板成功率 99.5% / 95.5%。
评测期间完全由网络从地面控制，没有教师前缀。训练摘要见
[本轮报告](../reports/G04/autonomous_learning_20261022/REPORT.md)。

`artifacts/G04/autonomous_actor/actor.pt` 是可直接载入的 TorchScript 网络：

```python
import torch
actor = torch.jit.load("artifacts/G04/autonomous_actor/actor.pt", map_location="cuda:0").eval()
with torch.inference_mode():
    actions = actor(observations)  # float32 [N, 794] -> [N, 18]
```

网络内含冻结的观测归一化，不要额外归一化输入。输入来自
`quadsmash.tasks.stand.observation_contract.observe_native`：11 × 69 维历史帧、
11 维目标命令、5 维实际事件阶段、1 维回合时钟、18 维上次动作，共 794 维。
当前帧在历史最前端，10 ms 一帧；新回合按 `PolicyHistory.reset` 清空历史，
回合时钟重置为零。策略只读可观测状态，不访问未来事件或教师参数。

动作顺序：FL/FR/RL/RR hip，FL/FR/RL/RR thigh，FL/FR/RL/RR calf，PiPER joint1–joint6。
动作先限制到 [-1,1]。腿关节目标相对默认站姿的尺度为回合 0.8 s 前 0.25 rad、
之后 0.55 rad；臂关节为 0.12 rad，再执行原关节位置限制。保持原资产、执行器、
2 ms 物理步和五倍控制降采样。完整映射由 `JumpEnv._pre_physics_step` 定义，
没有增加力矩、物理阻尼或隐形辅助力。

导出校验使用 4096 条真实采集观测，TorchScript 与原生 CPU 策略最大绝对误差为零。
这只是导出一致性；95.5% 最差面板成绩来自独立 Isaac Lab 进程中的原生权重。
不把 CPU 导出前向校验称为另一次 GPU 评测。

原生权重载入方式保持 `NativeRunner.load(..., load_optimizer=False)`。
BC 的 Adam 状态未保存到原生 PPO 优化器字段；后续 BC/PPO 应显式做权重初始化，
不能将此文件解释为 BC 中途精确优化器恢复。保留的训练参数、种子、数据与源码快照
用于重新运行或继续采集。

后续空中击球可以沿用统一 794/18 接口和此 actor 初始化；目标命令与球信号此前没有
被训练为击球任务，需要后续授权和独立评测。当前只验证 4 秒跳跃恢复、名义完整载荷
和固定开发初态范围，没有验证真实硬件或大范围动力学变化。
