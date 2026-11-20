# Branch B0 Oracle A diagnostic

日期：2026-11-20

入口：`scripts/audit_supermimic_oracle_a.py`

该审计只读取已有 `native_probe.pt` 的 PhysX 子步轨迹，不更新策略或世界模型。解析模型使用现有 `BallImpulsePriorV1` 的移动拍面、TOI 插值、法向恢复和切向摩擦项，并将原生球拍状态作为输入。

## 结果

| 归档 | 接触事件 | 出球方向余弦均值 | 法向冲量 MAE | 切向冲量 MAE | 推断有效质量 | TOI 相位 |
|---|---:|---:|---:|---:|---:|---:|
| `native_substep_contact_8zmiqosh` | 9 | 0.99784 | 0.00532 Ns | 0.00181 Ns | 0.00459–0.00498 kg | 0.06994 |
| `gpu_probe_current_c0722fc_run225943` | 7 | 0.99921 | 0.00597 Ns | 0.00200 Ns | 0.00456–0.00497 kg | 0.00777 |

两组 `nominal_a`/`nominal_b` 的重复轨迹最大绝对误差均为 0。两组均未出现连续多子步接触，当前证据不能覆盖多次接触模型。

## 判定

原生球拍状态下，TOI、接触对象、接触时刻和出球方向的解析结果在这 16 个已有接触事件上保持一致方向，支持继续做 B1 pilot。法向冲量仍有约 5–6 mNs 的平均误差，切向冲量存在约 1.8–2.0 mNs 误差，有限拍面有效质量和摩擦需要在 pilot 中继续测量。此结果不开放 V13 或 Actor contact BPTT。

缺失项由原始归档限制。接触法向由解析几何计算，原生归档只保存接触力与接触点，未保存 PhysX manifold 法向。`racket_part`、多接触顺序和地面/球网同子步干扰仍需 B1 采集补齐。
