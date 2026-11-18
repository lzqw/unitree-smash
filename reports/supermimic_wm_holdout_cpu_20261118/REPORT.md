# WM 留出误差复核

本目录保存同一 32 个事件对齐留出窗口上的 legacy 与动作条件化候选对照。数据来自冻结原生 archive 的 packed supervision，运行设备为 CPU。该报告不包含 PhysX 反事实响应证据。

## 候选

- legacy：`legacy_retry.json`
- action-conditioned candidate：`candidate_retry.json`
- 候选来源：`supermimic_action_ball_mode_causal_v3_recalibrated_20261118`
- 候选 gate：`candidate_gate.json`

## 结果摘要

候选在事件边缘的单步速度 RMSE 从 `4.1869` 降到 `2.5895`，前向速度符号错误率从 `0.7391` 降到 `0.3043`。接触前单步 RMSE 从 `4.0122` 降到 `3.2628`。

但是多步回归没有整体满足保护条件。候选在接触前偏移的 4 步 RMSE 从 `2.4607` 升到 `2.8352`，因此严格自回归筛选拒绝候选，拒绝原因是 `adjacent_autoregressive_degradation_-1_4`。候选未被晋升。

两个模型在本次非接触命令敏感度窗口上均报告球响应导数范数为零。该窗口不能验证击球动作梯度，不能替代原生 PhysX probe。

## 判定

- 单步事件筛选：通过
- 完整 4/16/32/64 步相邻事件保护：拒绝
- 候选安全筛选：拒绝，不复制 checkpoint
- 原生事件对齐探针：未执行
- 原生动作响应方向：未验证
- Actor 更新：0

CPU 留出结果只用于定位误差来源。当前候选仍不能进入 Contact-aware Actor BPTT 或大规模训练。
