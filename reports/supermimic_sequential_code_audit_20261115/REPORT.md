# 顺序 Recovery → Approach 代码审计

本轮只修改代码、运行 CPU 测试和读取已有轨迹；没有启动 Isaac、训练或新的128环境评估。

## 已完成的代码修复

- 新增 `--update-mode sequential|joint`，默认 `sequential`；
- sequential 路径明确执行 Recovery → 保存中间检查点 → Approach → 保存最终候选；
- joint 保留为独立对照模式，不再默认运行；
- 两个 head 使用显式白名单，切换时冻结另一 head；
- 学习率由 trainer 的 `actor_lr` 保存并复用，不再在切换时硬编码5e-5；
- 新增 `native_critic_stage_audit`，支持起跳、接近、分离、着地、稳定、末端阶段的 RTG 标签审计，不更新 critic；
- 新增只读 `ball_side_shadow`，缺失事件保持未知，默认不执行拒绝/接受；
- 累计预算报告新增 `relative_native_accepted`，与单步、块起点、最后接受变化分开。

## CPU 测试

命令：

```text
CUDA_VISIBLE_DEVICES="" QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
  bash scripts/python.sh -m pytest \
  tests/test_supermimic_sequential_audit.py \
  tests/test_supermimic_horizon_v2.py \
  tests/test_supermimic_physical_loop_components.py \
  tests/test_supermimic_native_metrics.py -q -ra --maxfail=1
```

结果：**49 passed in 1.75s**。

覆盖：shadow 缺项不当成功、分阶段 RTG 标签、预算向量差异、原有时域/物理循环组件。

## 已有顺序候选的累计偏移离线审计

数据来自已完成的 `calibrated_blocks_20261115_1415`，未重跑仿真：

- recovery/approach 的训练 history 已分块读取；
- 分别汇总 block 起点、原生接受参考和最后接受参考；
- 详细结果见 `sequential_cumulative_drift.json`。

该旧运行没有记录旧版本的 `relative_native_accepted` 字段，因此该字段在旧记录中为缺失；新代码会从固定 native reference 开始记录。不能把旧单步 `reference_final_command_delta` 当成累计原生偏移。

## 球侧 shadow 离线审计

对已有 pi0、piR、piRA 的 `episodes_summary.json` 运行只读 shadow：

- pi0：128条；
- piR：128条；
- piRA：128条；
- 不修改奖励、策略或正式判定；
- 缺少分离速度、拍面、越网或落点证据的字段保持 `missing_fields`，不默认成功。

详细输出见 `ball_side_shadow.json`。

## 未运行项目

- 未启动 Isaac；
- 未启动 Recovery/Approach 训练；
- 未运行新的128环境正式评价；
- 未更新或覆盖任何真实策略 checkpoint；
- 未启用 ball-side enforce；当前只提供 shadow 模式。
