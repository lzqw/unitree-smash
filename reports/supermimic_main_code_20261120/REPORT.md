# SuperMimic main code report

日期：2026-11-20

分支：`feat/supermimic-hybrid-toi-v13-20261119`

本报告记录当前主线代码和并行支线 B 的可复现入口。支线 A 的训练合同、评估脚本和正在进行的训练作业保持原状。

## 当前代码交付

- `supermimic_native_substep_trace_v1.py` 保存原始接触矩阵、接触位置矩阵、球拍力、地面力和球网传感器流。
- `audit_supermimic_oracle_a.py` 对已有 native PhysX 子步执行 TOI、接触对象、出球方向、法向/切向冲量和有效质量审计。
- `supermimic_contact_pilot_v1.py` 从原生子步归档构建动态接触和 near-miss pilot，CPU trace reader 支持并行 worker。接触矩阵不再丢弃 manifold slot，near-miss 在含接触 lane 中保持采样，并生成 gap/approach bucket、阈值敏感性、来源 hash 和无泄漏切分审计。
- `build_supermimic_contact_pilot.py` 和 `plan_supermimic_contact_pilot.py` 提供 B1 构建与剩余配额计划。
- `run_supermimic_event_aligned_native_probes.py` 支持从仓库目录直接启动原生子步采集。

## B0 结果

两组已有子步归档共 16 个接触事件。出球方向余弦均值为 `0.99784` 和 `0.99921`，法向冲量 MAE 为 `0.00532–0.00597 Ns`，切向冲量 MAE 为 `0.00181–0.00200 Ns`，推断有效质量为 `0.00456–0.00498 kg`。nominal 重复轨迹最大误差为 `0`。多次接触仍缺少原生证据，因此没有开放 V13 contact BPTT。

详见 [`reports/supermimic_oracle_a_20261120/REPORT.md`](../supermimic_oracle_a_20261120/REPORT.md)。

## B1 当前数据状态

现有 204 个 native probe 归档得到 1077 个动态接触候选和 706039 个 near-miss 候选。当前 pilot 已选 1077 个动态 contact 和 1000 个 near-miss，距离第一轮 `2000+1000` 目标还缺 923 个动态 contact。覆盖审计仍缺中心、边缘、二次接触及地面/球网干扰桶。旧归档没有 PhysX manifold normal、racket part 和完整多接触顺序，这些字段在数据中保留为显式缺失或质量标记。

详见 [`reports/supermimic_contact_pilot_20261120/REPORT.md`](../supermimic_contact_pilot_20261120/REPORT.md)。

## 验证

支线 B 定向测试通过，共 `9 passed`。完整工作区测试仍有 full-racket 相关失败和错误，未通过放宽验收阈值处理。

GPU 原生采集 smoke 已启动并在第 149 个控制 tick 停止，没有把未完成采集计入 pilot 数量。14 个带子步捕获的采集规格已保存在 `reports/supermimic_contact_pilot_20261120/specs/`，可继续补足剩余 923 条。

## 上传范围

本次 GitHub 提交包含支线 B 的源码、测试、B0/B1 报告和采集规格。工作区中与支线 A 相关的未提交修改没有纳入本次提交，也没有上传权重、机器人资产或未完成 GPU 采集目录。
