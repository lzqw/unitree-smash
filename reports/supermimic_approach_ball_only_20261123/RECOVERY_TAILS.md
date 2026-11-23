# 阶段二恢复轨迹

E1 修订 A8 和 A16 已导出 29 段实测碰后轨迹，来自 15 个不同发球案例，合计 4,474 个控制周期。每段由首次触拍开始，到复位前的终止状态结束。文件保留 18 维请求、基础动作、最终动作和执行关节目标，以及原生关节、机身、足端、羽毛球状态和 G07 混合器限速历史。完整执行器快照可通过输出中的原生分块和偏移引用恢复。

已重新读取导出的张量。全部轨迹的后继关节状态与下个控制周期的初始关节状态逐项一致。终止标记只出现在最后一帧，每个执行目标均为有限数。原生档案、配置、轨迹索引和导出文件均有 SHA-256。 [导出校验证据](RECOVERY_TAIL_EVIDENCE.json)

| 数据覆盖 | E1 A8 | E1 A16 |
|---|---:|---:|
| 完整恢复尾段 | 14 | 15 |
| WM 留出 lane | 8 | 9 |
| 非留出且有机械臂动作扰动 | 6 | 6 |
| 非留出且无扰动 | 0 | 0 |
| 非留出且无扰动的合格触球回合 | 28 | 31 |

当前尾段采样每八个 lane 取一个。首回合的机械臂辨识也按同一 lane 余数选取，并排除 WM 留出。实测保留下来的非留出尾段因此全部来自扰动回合。该批数据可用于执行器和碰后动力学诊断，正常策略分布下的恢复训练仍缺完整无扰动尾段。后续自然采集需要覆盖非留出且未施加辨识扰动的 lane，现有留出标签继续保留。

本次仅从已完成的原生归档导出，没有追加仿真预算。[轨迹导出程序](../../scripts/export_supermimic_recovery_tails.py)

## 自然回合补采入口

在冻结策略的原始 TRAIN128 上使用已有的原生状态采集。该入口使用球侧评分合同，保留原有尾段规则，并由冻结策略正常执行。采集后索引检查 checkpoint Actor 指纹、零优化器更新、执行合同和真实步数，再从序列中排除 WM 留出 lane。固定开发面板及独立留出面板不可用作此训练资料的来源。

```bash
export PYTHONPATH=.:scripts
export QUADSMASH_ZSTD_ARCHIVE=1
"$PYTHON" scripts/evaluate_supermimic_v1.py --checkpoint "$CHECKPOINT" \
  --g07-execution-root "$EXECUTION_ROOT" --approach-ball-only --record-physical-transitions \
  --headless --output "$CAPTURE"
"$PYTHON" scripts/index_supermimic_recovery_tails.py --frozen-run "$CAPTURE" \
  --scratch "$SCRATCH" --output "$CAPTURE/recovery_index.json"
"$PYTHON" scripts/export_supermimic_recovery_tails.py --index "$CAPTURE/recovery_index.json" \
  --scratch "$SCRATCH" --output "$CAPTURE/recovery_trajectories.pt"
```

兼容性和数据分区已通过 CPU 回归测试。冻结自然采集尚未运行。若后续现有多块采集已经得到无扰动尾段，可直接使用训练归档。
