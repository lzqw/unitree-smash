# 状态驱动长时域落地 MPC 结果（2026-11-09）

本轮暂停 RL、BC 和教师蒸馏，使用同一套 Go2+PiPER 原生物理环境，从实际球拍与球分离后的全身状态接管恢复。控制器不读取 TRAIN 编号、失败名单或历史轨迹库；历史 nominal 只作为统一软姿态目标，当前状态、速度、接触和上一条实际命令每次参与重规划。

## 时间规划

控制器采用两层时间尺度：

- **密集接触规划窗：** 32 个 10 ms 节点，共 0.32 s，覆盖空中落脚、首次触地和初始加载的主要过程。
- **滚动重规划：** 每 16 个控制 tick（0.16 s）重算一次；阶段变化或计划耗尽时立即重算。每次只执行当前序列的下一条命令，随后继续读取原生状态。
- **完整任务验收：** 原生评估仍执行完整 8 s 回合。0.32 s 是局部接触模型的密集预测窗，不被宣称为 8 s 全程动力学证书；8 s 任务结果由 PhysX 原生执行判定。

## 固定 TRAIN128 结果

| 指标 | 长时域状态驱动 MPC |
|---|---:|
| complete5（5 m/s 主指标） | **37/128（28.91%）** |
| 合格空中触球 | 74/128 |
| 条件恢复 | **49/74（66.22%）** |
| 实际球拍接触 | 110/128 |
| 越网 | 73/128 |
| 目标落点 | 60/128 |
| 数值故障 | 0 |
| GPU 用量 | 0.696 GPU 小时 |

旧纯控制对照 `pure_control_state_settling_v23` 为 7/128 complete5、7 条条件恢复；同一长时域控制器保留其中 6 个，并新增 31 个 complete5 案例，丢失 1 个。短时单步状态驱动版本为 0/128，说明仅检查 10 ms 局部动作不足以完成恢复。

报告中的 legacy `complete=0` 是 8 m/s 字段；正式比较使用 `complete5=37`。本轮仍是固定开发面板、保留基础控制器和持久恢复辅助的控制实验，`formal_acceptance=false`，不能直接宣称泛化或完全独立 RL。

## 诊断解释

长时域规划把恢复从“下一步姿态是否可行”推进为“当前命令经过一段触地/加载序列后是否仍有足够净空和低动量”。结果中条件恢复从 7 提升到 49，说明接触建立与加载规划是当前成功率的关键变量。complete5 仍低于条件恢复，剩余失败主要来自球侧出球、落点和速度条件的交集；后续需要逐回合保留球侧与恢复交集，不能把 49 条恢复全部当成完整成功。

该控制器仍不是实时全身非线性 MPC 证书：预测使用近似的隐式 PD/接触模型，执行时由 PhysX 原生验证；预测失败会拒绝候选并保持上一条实际命令。此次没有改变关节、速度、力矩、摩擦、碰撞或成功判定。

## 下一步

1. 固定当前 37/128 版本，补做同面板的逐案例保留/丢失/新增和首次足端接触、承重建立、非足部接触时间统计。
2. 对 49 条条件恢复中未 complete5 的回合，区分出球速度、越网、落点和事件顺序失败，避免继续盲目调落地参数。
3. 将 0.32 s 接触窗扩展为分阶段非均匀窗口：触地前密集、加载后较疏，并在真实支撑建立或计划失效时立即重规划。
4. 在固定控制器确认收益后，再考虑 JAX/GPU 批量化；当前瓶颈是 CPU QP/接触预测吞吐，不应以缩短 8 s 评估窗口换取速度。

## 复现证据

- 控制器：`scripts/pure_control_state_driven_long_mpc_v1.py`
- 原生适配：`scripts/native_pure_control_state_driven_long_mpc_v1.py`
- 计划生成：`scripts/prepare_native_state_driven_long_mpc_v1.py`
- 静态检查：`scripts/check_native_state_driven_long_mpc_v1.py`
- 结果：`reports/pure_control_state_driven_long_mpc_all128_20261109_03/metrics.json`
- Git 提交：`ce4c690`

## 自适应阶段时域对照（2026-11-09）

随后完成了 `pure_control_state_driven_adaptive_mpc_v1` 的完整 TRAIN128、8 秒原生评估。该版本按阶段使用不同局部窗口：`AIR_PREPARE=0.40 s`、`BUFFER_LOAD=0.24 s`、`SUSTAIN=0.12 s`，每 8 个控制 tick 重规划。

| 指标 | 固定 0.32 s 长时域 v1 | 自适应阶段时域 v1 |
|---|---:|---:|
| complete5（5 m/s 主指标） | **37/128（28.91%）** | **3/128（2.34%）** |
| 合格空中触球 | 74/128 | 74/128 |
| 条件恢复 | **49/74（66.22%）** | **4/74（5.41%）** |
| 实际球拍接触 | 110/128 | 110/128 |
| 越网 | 73/128 | 71/128 |
| 目标落点 | 60/128 | 58/128 |
| 数值故障 | 0 | 0 |
| GPU 用量 | 0.696 GPU 小时 | 0.897 GPU 小时 |

该对照仍然从球拍与球分离后的实际状态接管，未读取 TRAIN 编号、失败名单或轨迹库，且完整执行 8 秒回合。自适应版本的接触和求解流程可以运行，但在当前近似接触/加载模型下，阶段性缩短窗口没有保留固定 0.32 s 版本形成的恢复序列，导致条件恢复和 complete5 大幅下降。它不能证明更长时域原则无效，说明当前控制器更依赖一个覆盖触地与初始加载的稳定局部窗口。

因此当前控制结论是：**固定 0.32 s 长时域 v1（37/128）仍为最佳版本；0.64 s 固定窗口（2/128）和本次自适应窗口（3/128）均作为负面对照保留。** 后续应先分析 v1 的逐案例接触建立、承重建立和球侧交集，再设计非均匀窗口；不能仅按阶段改变 horizon 或 replan stride。

自适应结果证据：

- 控制器：`scripts/pure_control_state_driven_adaptive_mpc_v1.py`
- 原生适配：`scripts/native_pure_control_state_driven_adaptive_mpc_v1.py`
- 计划生成：`scripts/prepare_native_state_driven_adaptive_mpc_v1.py`
- 静态检查：`scripts/check_native_state_driven_adaptive_mpc_v1.py`
- 结果：`reports/pure_control_state_driven_adaptive_mpc_all128_20261109_01/metrics.json`
- 作业：SUCCEEDED，约 3229 秒，0.897 GPU 小时

## 固定 0.32 s 窗口、每 8 tick 重规划对照（2026-11-09）

为隔离重规划频率的影响，完成了与固定 v1 完全相同的 0.32 s 局部预测窗口，只将重规划间隔从 16 个控制 tick（0.16 s）改为 8 个 tick（0.08 s）。本轮仍执行完整 8 s 的 TRAIN128 原生评估，未改变关节、速度、力矩、摩擦、碰撞或成功判定。

| 指标 | 固定窗口 v1：16 tick | stride8 对照：8 tick |
|---|---:|---:|
| complete5（5 m/s 主指标） | **37/128（28.91%）** | **19/128（14.84%）** |
| 合格空中触球 | 74/128 | 74/128 |
| 条件恢复 | **49/74（66.22%）** | **24/74（32.43%）** |
| 实际球拍接触 | 110/128 | 110/128 |
| 越网 | 73/128 | 72/128 |
| 目标落点 | 60/128 | 59/128 |
| 数值故障 | 0 | 0 |

stride8 共执行 102,400 个环境步，作业成功，耗时约 3356 秒、0.932 GPU 小时。相对固定 v1 的 37 个成功案例，它保留 14 个、丢失 23 个并新增 5 个。虽然计算量更大、重规划更频繁，完整成功和条件恢复都明显下降。

这说明当前近似接触／加载模型下，更频繁地重规划会破坏已经形成的 0.32 s 恢复序列；不能把“更新更快”直接等同于“反馈更好”。因此当前应继续保留固定 0.32 s、每 16 tick 重规划的 v1（37/128）作为最佳版本，stride8 作为负面对照。后续若要提高反馈频率，应先解决计划交接、命令连续性和预测模型一致性，再重新评估，而不是继续单独缩短 stride。

本轮证据：

- 控制器：`scripts/pure_control_state_driven_long_mpc_v1_stride8.py`
- 原生适配：`scripts/native_pure_control_state_driven_long_mpc_v1_stride8.py`
- 计划生成：`scripts/prepare_native_pure_control_state_driven_long_mpc_v1_stride8.py`
- 静态检查：`scripts/check_native_state_driven_long_mpc_v1_stride8.py`
- 结果：`reports/pure_control_state_driven_long_mpc_stride8_all128_20261109_01/native/metrics.json`
- 作业记录：`reports/pure_control_state_driven_long_mpc_stride8_all128_20261109_01/native/job.json`

## 计划交接连续性对照（2026-11-09）

在固定 v1 的 0.32 s 预测窗口和每 16 个 tick 重规划基础上，增加了 4 步新旧计划交接过渡，测试重新规划时的命令连续性是否能避免恢复动作被打断。其余物理限制、执行器、碰撞和 8 s 完整任务判定均保持不变。

| 指标 | 固定 v1 | 4 步交接过渡 |
|---|---:|---:|
| complete5（5 m/s 主指标） | **37/128（28.91%）** | **30/128（23.44%）** |
| 合格空中触球 | 74/128 | 74/128 |
| 条件恢复 | **49/74（66.22%）** | **41/74（55.41%）** |
| 实际球拍接触 | 110/128 | 110/128 |
| 越网 | 73/128 | 73/128 |
| 目标落点 | 60/128 | 60/128 |
| 数值故障 | 0 | 0 |

连续交接版本相对 v1 保留 29 个成功、丢失 8 个并新增 1 个（TRAIN27）。因此，平滑交接没有改善当前近似模型下的恢复，反而降低了条件恢复和完整成功。说明旧计划与新计划之间的差异并非单纯的命令跳变问题；将新旧目标短时间平均，可能把两个各自可行的恢复动作混成不可行的中间动作。

当前最佳仍为固定 v1（0.32 s、每 16 tick 重规划、37/128）。stride8（19/128）和连续交接（30/128）均作为负面对照保留。后续不再继续简单增加交接平均步数；若要改进重规划，应在统一动力学／接触预测中重新优化交接序列，而不是对关节目标做逐点线性混合。

本轮证据：

- 控制器：`scripts/pure_control_state_driven_long_mpc_v1_continuous.py`
- 原生适配：`scripts/native_pure_control_state_driven_long_mpc_v1_continuous.py`
- 结果：`reports/pure_control_state_driven_long_mpc_continuous_all128_20261109_01/native/metrics.json`
- 作业记录：`reports/pure_control_state_driven_long_mpc_continuous_all128_20261109_01/native/job.json`

## QP 内命令连续性对照（2026-11-09）

上一版把新旧计划在求解后做线性平均，结果下降到 30/128。为了避免这种后处理改变可行性，本轮将最终绝对关节目标与上一条实际命令之间的二次代价直接加入 interval QP，保持固定 0.32 s 预测窗口、每 16 tick 重规划和原生 8 s 评估。默认权重为零，只有本控制器变体启用，物理限制和任务判定未改变。

| 指标 | 固定 v1 | QP 内连续性 |
|---|---:|---:|
| complete5（5 m/s 主指标） | **37/128（28.91%）** | **40/128（31.25%）** |
| 合格空中触球 | 74/128 | 74/128 |
| 条件恢复 | **49/74（66.22%）** | **52/74（70.27%）** |
| 实际球拍接触 | 110/128 | 110/128 |
| 越网 | 73/128 | 73/128 |
| 目标落点 | 60/128 | 60/128 |
| 数值故障 | 0 | 0 |

相对固定 v1，QP 内连续性版本保留 35 个成功、丢失 2 个并新增 5 个（TRAIN27、45、49、84、121）。这说明连续性代价放在求解目标中比事后线性混合有效：它在满足动力学、接触和执行器约束的同时抑制命令突变，而不会把两个动作平均成未验证的中间动作。

该结果是固定 TRAIN128 开发面板上的控制辅助成绩，仍保留基础控制器和持久恢复辅助；不能直接称为独立 RL 已学会。当前应保留 QP 内连续性 40/128 为新的最佳控制候选，并继续用逐案例旧成功回归和未见状态评估确认其稳定性。

本轮证据：

- 控制器：`scripts/pure_control_state_driven_long_mpc_v1_qp_continuity.py`
- 原生适配：`scripts/native_pure_control_state_driven_long_mpc_v1_qp_continuity.py`
- QP 修改：`scripts/pure_control_interval_qp_v1.py`、`scripts/whole_body_support_qp_v1.py`
- 结果：`reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native/metrics.json`
- 作业记录：`reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native/job.json`

## QP 内命令连续性独立种子复测（2026-11-09）

为检验 40/128 是否只来自单个评估种子，使用相同控制器、相同冻结策略、相同 TRAIN128 面板和完整 8 s 原生执行，仅将随机种子改为 `882073`。本轮没有改变物理参数、成功判定或控制器配置，也没有把中途统计当作最终结果。

| 指标 | 首次 QP 内连续性评估 | 独立种子 882073 |
|---|---:|---:|
| complete5（5 m/s 主指标） | **40/128（31.25%）** | **40/128（31.25%）** |
| 合格空中触球 | 74/128 | 74/128 |
| 条件恢复 | 52/74（70.27%） | 52/74（70.27%） |
| 实际球拍接触 | 110/128 | 110/128 |
| 越网 | 73/128 | 73/128 |
| 目标落点 | 60/128 | 60/128 |
| 数值故障 | 0 | 0 |

独立种子完整执行了 102,400 个环境步，作业成功，记录了全部 128 个 TRAIN 案例。两个种子在汇总指标上完全一致，说明该控制器在这个固定开发面板上具有较好的复现性；这仍然不是未见来球、初态扰动或正式泛化评测，也不能把控制辅助成绩称为纯 RL 已学会的能力。

本轮证据：

- 配置：`reports/pure_control_state_driven_long_mpc_qp_continuity_native_preparation_20261109_01/controller.json`
- 独立种子结果：`reports/pure_control_state_driven_long_mpc_qp_continuity_all128_seed882073_20261109_01/native/metrics.json`
- 完成标记：`reports/pure_control_state_driven_long_mpc_qp_continuity_all128_seed882073_20261109_01/native/completed.json`
