# Branch B1 contact pilot status

日期：2026-11-20

入口：`scripts/build_supermimic_contact_pilot.py`

该构建器读取已有 `native-one-control-tick-substeps-v1` 归档，在 CPU 上并行扫描每个子步，保留原始球/球拍状态、接触点、接触力积分、相对速度和 TOI 前后快照。接触矩阵保留所有 manifold slot，行中同时保存 slot 级冲量和有效 slot 索引。缺失的 PhysX manifold 字段显式写入 NaN 并附 `quality_flags`。

## 当前归档结果

- 扫描 native probe：204 个
- 动态 racket contact 候选：1077 个
- near-miss 候选：706039 个
- 已选动态 contact：1077/2000
- 已选 negative/near-miss：1000/1000
- 状态：`INSUFFICIENT_CANDIDATES`
- 仍需新的原生 GPU 采集：923 个动态 contact

现有 1077 条不能填充成 2000 条。解析后的 `racket_part`、PhysX manifold normal、接触法向的真实引擎标签和多接触顺序在旧 trace 中没有保存，因此这些字段保留为 unknown 或显式缺失，不能伪造。当前覆盖审计还缺中心区 100 条、边缘 56 条、二次接触 100 条，以及地面/球网干扰各 25 条。

near-miss 现在会在包含其他接触的 lane 中采样，并按解析几何 gap、接近速度和干扰对象分桶。对 `0.0001/0.001/0.005/0.01 N` 的阈值敏感性审计在现有归档上均得到 1077 个动态区间，说明当前力信号远高于阈值，后续仍需补采低冲量边界样本。

新采集的 native trace 还会保存 racket body 名称、接触 filter 路径和 filter index。旧归档缺少这些静态元数据，因此旧行仍保留显式 unknown，不回填猜测值。

pilot 行按 source archive 做原子切分，并保留 spec、episode、capture 和 lane hash。当前切分为 train/validation/test `1479/364/234`，source overlap 为零，`spec_04_tick279` 作为未见 spec holdout。状态仍为 `INSUFFICIENT_CANDIDATES`，不能进入训练。

下一批采集需要沿中心、5×5、拍框/边缘、入射法向速度、切向速度、球拍线/角速度、TOI 相位、二次接触及地面/球网干扰补足剩余 923 条。negative 已有足够数量，可与动态采集并行运行，negative 采集不再等待动态 contact 完成。

已启动过一次 2-spec GPU smoke。当前原生评估每 100 个控制 tick 约需 6 分钟，试运行在第 149 个 tick 主动停止，没有写入新的 native probe，也没有计入 pilot 数量。后续使用 `collection_plan.json` 和 `scripts/run_supermimic_event_aligned_native_probes.py` 继续采集即可。

已保存 14 个带子步捕获的事件规格到 `specs/`。完整续采命令需要使用原生策略 checkpoint 和 `/tmp/supermimic_exec_overlay_20261119` 执行根，并将输出写入新的实验目录。
