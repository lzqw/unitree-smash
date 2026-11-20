# SuperMimic main code report

日期：2026-11-20

分支：`feat/supermimic-hybrid-toi-v13-20261119`

本报告记录当前主线代码和并行支线 B 的可复现入口。支线 A 的训练合同、评估脚本和正在进行的训练作业保持原状。

## 当前代码交付

- `supermimic_native_substep_trace_v1.py` 保存原始接触矩阵、接触位置矩阵、球拍力、地面力和球网传感器流，并记录 racket body/filter 元数据。
- `audit_supermimic_oracle_a.py` 对已有 native PhysX 子步执行 TOI、接触对象、出球方向、法向/切向冲量和有效质量审计。
- `supermimic_contact_pilot_v1.py` 从原生子步归档构建动态接触和 near-miss pilot，CPU trace reader 支持并行 worker。接触矩阵不再丢弃 manifold slot，near-miss 在含接触 lane 中保持采样，并生成 gap/approach bucket、阈值敏感性、来源 hash 和无泄漏切分审计。
- `build_supermimic_contact_pilot.py` 和 `plan_supermimic_contact_pilot.py` 提供 B1 构建与剩余配额计划。
- `run_supermimic_event_aligned_native_probes.py` 支持从仓库目录直接启动原生子步采集，并支持保留完整 probe、归档未完成分支后续采集。

## B0 结果

两组已有子步归档共 16 个接触事件。出球方向余弦均值为 `0.99784` 和 `0.99921`，法向冲量 MAE 为 `0.00532–0.00597 Ns`，切向冲量 MAE 为 `0.00181–0.00200 Ns`，推断有效质量为 `0.00456–0.00498 kg`。nominal 重复轨迹最大误差为 `0`。多次接触仍缺少原生证据，因此没有开放 V13 contact BPTT。

详见 [`reports/supermimic_oracle_a_20261120/REPORT.md`](../supermimic_oracle_a_20261120/REPORT.md)。

## B1 当前数据状态

2026-11-21 受控原生采集结束，正式 pilot 已有 2000 条动态接触、1000 条负例，达到数据覆盖门槛。20 批原生数据支持 25 个击球网格、125 个后续接触区间及地面/球网干扰分层。首触解析校准的独立留出集 TOI P95 为 1 个子步，误报率为 1/69（1.45%），策略训练门槛仍未通过。[测量和准入明细](../supermimic_contact_pilot_20261120/REPORT.md)

## 验证

本轮支线 B 的 4 组定向测试共 17 项通过。已核对逐点原生法向、摩擦、扣除重力/阻力后的冲量与动量变化、源文件哈希和 split 重叠。[启动前审查记录](../supermimic_contact_pilot_20261120/CODE_REVIEW.md)

## 上传范围

本次 GitHub 提交包含支线 B 的源码、测试、B0/B1 报告和采集规格。工作区中与支线 A 相关的未提交修改没有纳入本次提交，也没有上传权重、机器人资产或未完成 GPU 采集目录。
