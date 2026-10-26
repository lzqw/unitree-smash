# 075阶段中间策略冻结比较：已登记，等待当前流程退出

登记于2026-10-26：已看到本轮075训练中的3次完整回合，以及075冻结8的原全TRAIN128结果；尚未运行075冻结16/24/32或最终40。因此这是训练开发选择，不能视为盲测。

当前1734658观察进程、1734667共享guard及1734680原生训练仍在运行。新观察进程只等待；必须等原流程完成新增32次训练、最终40的原全128评测、轨迹审计和视频，实际退出且达到READY_FOR_VISUAL_REVIEW，验证全部原生收据、审计SHA和空闲共享GPU账本后，才会顺序评测16、24、32。每项仍用原GPU0 guard，单项2700秒/102400控制环境步上限，不新增云资源或并行GPU任务。

CPU实测：从实际075 canonical8重建的平均动作在1366条已保存原生公共输入上与其公开策略逐位一致；实际原生16诊断权重的导出也为零误差。保留同一个固定base846、原腿0.03/臂0.15尺度。16的actor RMS为14680064，critic RMS为4194304，分别相对源8前进2097152；保留旧actor的10485760偏移。全部17个Adam状态真实前进160步。诊断文件仅用于冻结推理，绝不作为严格恢复检查点。

第一次CPU准备发现RSL中间诊断的infos实际为None，而canonical才附带初始化信息；当时尚未创建PLAN、策略或启动GPU。实现已改为验证真实作业INPUTS、严格源SHA、training_signature、RMS、Adam和阶段计数，并明确要求诊断infos=None。修正后准备与完整原075冻结8独立重算均通过。

全部比较使用原TRAIN128、noise=none、seed724201。开发排序预先登记为完整成功数→条件恢复数→空中接触数→攻击出球数→更早阶段。这优先保留完成任务和可恢复接触；不改写旧025比较的攻击优先排序。任一实测策略达到原50%/70%/85%及NF0开发门槛，则停止多余比较，转入原A/B各200回合的正式评测准备；开发门槛不是正式通过。

当前269份原持有源码/输入未改；本队列额外持有父PLAN和CPU导出来源，共284份，另在启动时持有自身PLAN字节。所有持有文件需保持到本队列真实退出。已有leg06七来源候选仍只是CPU准备，没有启动；如本比较得到新增来源，需要后续重新登记它的来源集合和选择，不能直接沿用旧七来源计划。

主要证据：PREDECLARED_PLAN.json、CPU_PREPARATION_CHECK.json、ACTUAL_CPU_PIPELINE_CHECK.json、exports/iter16/EXPORT_CHECK.json。全目标仍未通过；正式A/B未读、未评测，训练中的稀少成功不计作冻结成绩。

准备命令（已运行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/autodrive/Isaac/env_isaaclab/bin/python reports/G07_full_racket_v4/centered_placement075_intermediate_selection_queue01/run.py --prepare-only
```

启动观察进程后不可重复运行或因观察超时重启。状态见state.json及DETACHED_LAUNCH_RECEIPT.json；真实失败保留error.txt和全部原始输出，由后续证据决定下一步。
