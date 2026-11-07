# 完整区间QP耗时分析与批量约束构造候选

2026-11-07；性能工程进展，不是落地成功率进展。原生 GPU0 全场对照继续运行，未修改其固定源文件，也未更改后续已排队的实验输入。

## 实际测量

输入为已经完成的 `landing_mpc_command_history_cohort26_repeat_20261107_01` 真实状态与最终命令记录。按照原审计入口回放全部观测记忆，仅在首次活动时刻和每 100 tick 求解，得到 111 个状态；每个状态比较原目标及最小命令变化目标，总计 222 次 MPC 调用。

CPU profile 总耗时 67.77 s，其中区间约束构造累计约 9.97 s，隐式预测约 24.04 s，模型 snapshot 约 15.86 s。累计时间存在嵌套，不能相加；profile 本身会增加运行开销。这说明除了 QP 后端，重复 Python 数组构造与动力学预测也值得优化。

新候选批量构造所有隐式子步的约束，保留每一条力矩、速度、关节位置、单边法向力与摩擦多边形约束及原行顺序，没有减少子步或时域，也没有改代价和容差。独立子进程通过替换内存中的函数引用进行 CPU 比较，不修改磁盘上的原控制器。

| 比较项 | 结果 |
|---|---:|
| 0—4 足接触、1／16 子步等价性及异常维度测试 | 11 项通过 |
| 真实状态数／目标设置数 | 111／2 |
| 首步、完整时域接受／拒绝及失败阶段差异 | 0 |
| 首步关节目标最大差异 | 0 rad |
| 无 profile 原版完整回放耗时 | 42.1446 s |
| 无 profile 批量版完整回放耗时 | 37.6107 s |
| 单次顺序比较耗时减少 | 10.76% |

两种实现各自均得到原目标 75 次首步通过、32 次完整时域通过；命令变化目标 75 次首步通过、20 次完整时域通过。后者是没有可行性回退的 CPU 目标对照，不能混同当前正在运行的带回退原生候选。

这是一次顺序计时，调度与缓存仍影响耗时；没有宣称原生仿真也提升相同比例，更没有证明闭环轨迹逐位相同。物理结果保持需要后续原生运行验证。

## 代码与接入状态

- `scripts/batched_interval_constraints_v1.py`：独立批量构造候选。
- `scripts/audit_batched_interval_runtime_v1.py`：两个隔离子进程的真实记录比较。
- `tests/test_batched_interval_constraints_v1.py`：全部约束、上下界、行标签与命令残差等价性。
- `scripts/patches/batched_interval_constraints_entry_v1.patch`：仅替换约束构造函数导入的待应用补丁，**尚未安装**。

不在正在运行的命令变化代价对照或原样透传对照中插入该改动。后续版本可使用此候选缩短教师计算时间，但必须继续保留原物理残差检查及原生结果核对。

复现命令（输出目录必须不存在，以保留旧证据）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/python.sh \
  scripts/audit_batched_interval_runtime_v1.py \
  --run reports/landing_mpc_command_history_cohort26_repeat_20261107_01 \
  --output reports/landing_batched_interval_runtime_20261107_01
```

紧凑证据及代码哈希见 [JSON](landing_interval_runtime_20261107.json)。大 profile、原始回放记录与权重留在本机。正式 G07 验证仍失败；此项性能测试不替代有效恢复、学生学习或正式任务验收。
