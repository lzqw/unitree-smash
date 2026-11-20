# Branch B1 contact pilot status

日期：2026-11-20

入口：`scripts/build_supermimic_contact_pilot.py`

该构建器读取已有 `native-one-control-tick-substeps-v1` 归档，在 CPU 上并行扫描每个子步，保留原始球/球拍状态、接触点、接触力积分、相对速度和 TOI 前后快照。原始对象 filter 分开处理，支持逐个保存 PhysX manifold 接触点和对应的法向冲量。缺失的 PhysX manifold 字段显式写入 NaN 并附 `quality_flags`。

## 11 月 21 日正式 pilot

20 批受控的原生铰接球拍试验得到 2226 个击球候选。根据覆盖配额选取 2000 个击球区间和 1000 个负例，每个负例来自单独的 capture/lane。三集合分别有 2069、447、484 行，按独立 teacher lane family 分组，已知 spec、原始采集文件和 episode 跨集合重叠均为零。所有正例都有 PhysX 逐点法向和分离距离。拍面部件是按原生接触点和配置几何得到的分类。样本来自一个 teacher 前缀的 16 组 lane family；跨策略或跨实验室的泛化仍需独立证据。

125 个正例属于第二次及之后的接触区间。每次区间以连续球拍力定义，保留两次区间之间的无力步数和几何间隙，有助于区分实际分离与短暂断力。覆盖全部 25 个网格，另有 47 个地面、42 个球网干扰窗口。404148 个无球拍 manifold 的子步中，球拍力 P50/P95/P99 都是 0 N。

先仅用训练集合校准解析首触时间、法向冲量尺度和摩擦。独立测试的 315 次首触全部检出，TOI 中位误差 0 子步，P95 为 1 子步。出球方向 P10 为 0.9873，速度幅值比中位数 0.9977。69 个整段无击球试验中有一个误报，FPR 1.45%，超过 1% 门槛。11/315 条出球法向分量符号有误。当前有数据准入，策略训练继续关闭；64 个控制步稳定性、Body WM 球拍输入与 V13 均未在本轮验收。

正式数据位于本机 `/home/autodrive/LZQW/experiments/supermimic_b1_pilot_20261121/pilot.pt`，其完整原始数据来源保存在同目录 manifest 及 pilot 的 source_files 中。归档摘要、测量审计和首触校准分别见 [pilot 摘要](pilot_20261121_summary.json)、[子步测量](measurement_audit.json) 和 [独立集合结果](analytic_calibration.json)。

## 11 月 21 日复核

旧归档经文件级去重后有 953 条正例记录。进一步逐项比较 TOI 前后子步状态、冲量、接触点、步数和时钟后，其中 680 条与其他记录完全相同，只剩 273 个不同观测窗口。这个数仍可能包含同一动作前缀的近似重复。下列 1077 与缺口 923 是此前的原始计数，不能作为当前独立事件数或工期外推依据。按 273 条作基数，距离第一轮 2000 条还差 1727 个不同击球窗口，新的受控采集将独立核验。

矩阵的对象和 manifold 语义以 CODE_REVIEW.md 的 11 月 21 日复核为准。机器人刚体名 Racket 与拍框、拍线的分类分别记录。

## 历史归档结果

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
