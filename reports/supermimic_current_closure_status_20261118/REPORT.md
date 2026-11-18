# SuperMimic 当前训练闭环状态

日期：2026-11-18
基线提交：`aeb8ebf`

## 已完成的代码闭环

- Recovery 与 Approach 共用真实 Replay、共享 world model 和 critic，并保留独立 actor/Adam 状态。
- 原生 transition 的来源、策略版本、episode/lane/stream、事件阶段和执行命令可以追溯。
- world model 候选支持留出集验证，训练过程中保存最佳验证状态，而不是无条件采用最后一步权重。
- 原生响应校准要求显式执行合同、实际 checkpoint 身份和 held-out ball-response 支持。
- action-conditioned ball model 已接入候选拟合、想象和质量检查路径。未通过原生证据时不会放行 actor 训练。
- 历史冻结原生回合已恢复为显式 anchor archive，不能冒充初始 `-12.52678435` 基线。
- 新增事件对齐探针规格生成器 `scripts/build_supermimic_event_aligned_probe_specs.py`，按接触/拍框事件边缘覆盖留出 lane，生成器只读原生 archive，不运行 Isaac、不制造反事实状态。

## 当前证据

- 代码测试：`17 passed`（native response data、action-conditioned ball、candidate gate）。
- 最新多来源候选：`/home/autodrive/LZQW/experiments/supermimic_ball_response_fit_multi_best_20261118_203802413079814`，world SHA256 `a2e6609ded2d4819ba180f4f02897d0f8e1cf225e0ce5f14237fe47f78560a7f`，action-ball SHA256 `bd1bad50f3f9cabeefd66ab0dca7eb7cedd4c954dbeea9f832f0d59d8cab478d`。候选状态为 `FIT_COMPLETED_NOT_ADMITTED`。
- 留出事件对齐规格已从冻结 archive `03463185ebcb5a211e0ee458edee10a1e577fbdd5f17c59934f7fae6a89a4295` 生成，选择 ticks 267、270、266、271、272，覆盖 12 个留出 lane。规格目录在 `/home/autodrive/LZQW/experiments/supermimic_event_aligned_probe_specs_20261118_2210`，其 `manifest.json` 已随报告保存。
- 现有旧探针在留出集仅有 4 条非零球响应行、2 个独立 lane。新的 `--min-native-ball-validation-rows` 默认值为 8，因此该候选会被拒绝，不能进入 actor/critic 训练。
- 旧候选验证显示接触前 offset -1 速度 RMSE 约 5.17 到 5.35 m/s，符号错误率约 47.8% 到 60.9%；接触时刻 offset 0 RMSE 约 1.77 到 2.17 m/s，符号错误率 26.1% 到 21.7%；offset +1 RMSE 约 0.0818，符号错误率为 0。该结果仍不足以证明 PhysX 动作梯度方向正确。

## 尚未完成

- 事件对齐 probe specs 只生成，尚未在 Isaac/PhysX 中执行。
- 尚未完成同一起点的原生正负动作扰动与 world model Jacobian 对照。
- 尚未启动 Recovery→Approach→Recovery 的真实参数更新闭环。
- 尚未运行大规模训练、128 环境正式评价或性能晋升。
- 当前服务器 NVIDIA 驱动不可用，`torch.cuda.is_available()` 为 false，未声称 GPU 训练已经开始。

## 运行规则

本轮没有启动训练进程，也没有修改物理、奖励、动作合同或正式评分。下一步应先使用事件对齐规格执行原生探针，确认留出球响应支持和动作方向，再决定是否允许新候选进入 actor 更新。

## 本次新增修复（提交 `b2c26e9`）

- `scripts/fit_supermimic_action_ball.py` 新增接触事件三时刻联合选择。候选只有在 `-1` 和 `+1` 邻接时刻不恶化时才有资格进入候选比较，再按三时刻相对 RMSE 选取最佳保存点。若所有优化保存点都引入邻接退化，则恢复零初始化 sweep，并标记为未接受。
- `scripts/fit_supermimic_native_response.py` 将 GPU 同步从每个优化步移到每16步验证边界，避免校准内循环被无必要的 CPU/GPU 同步串行化。验证数值和模型合同不变。
- 新增 `tests/test_supermimic_action_ball_fit.py`，覆盖接触后退化拒绝、联合指标选择和缺失事件时刻拒绝。

新增针对性测试结果：`20 passed`，语法检查退出码为0。

本次仍未启动 Isaac、Actor/Critic 更新或大规模训练。服务器当前没有可用 NVIDIA 驱动，GPU 原生训练需在驱动恢复后执行。

## 最新提交（`c5d84d3`）

候选元数据现在会记录 `accepted_update`、`accepted_parameter_update` 和完整的事件安全选择结果。若所有训练保存点都使接触前或接触后邻接误差恶化，最终 checkpoint 明确恢复为零初始化 sweep，而不是把最后一次退化权重当作候选。

## GPU启动保护（提交 `0f44f7b`）

训练入口现在在 `AppLauncher`/Isaac 启动前检查 CUDA 设备。当前服务器不可见 GPU 时会立即报出明确的非零错误，不再先启动仿真再在 `cuda:0` 张量分配处失败，也不会静默退回 CPU。

## 事件邻接准入收紧（当前后续提交）

World Model 候选准入现在对接触前和接触后一个控制 tick使用严格不退化约束。接触当步误差改善不能抵消邻接时刻新增的冲量。新增合同测试覆盖该拒绝路径。

## 原生响应校准同步事件准入（后续提交）

原生配对响应校准现在与离线动作球拟合使用同一事件三时刻选择规则。每16次校准更新同时检查留出集的 `-1/0/+1` 事件误差，邻接时刻退化的校准权重不会被保存为最佳 checkpoint。若没有事件安全 checkpoint，校准入口会恢复初始状态并报错。
