# 将真实救援转换成可表达的恢复监督

2026-11-07。本轮在不修改运行中控制器的同时，完成了已验证恢复轨迹的筛选、命令历史保存及新动作坐标的还原检查。没有开展新的学生训练，不能将这些工作计作关闭教师后的能力提升。

## 已完成的数据

来源为已复现29/128的 `landing_mpc_command_history_repeat3_20261107_01`。先重新计算原TRAIN128完整评分，再核对已经通过的命令审计及逐轨迹哈希。

| 数据 | 数量 |
|---|---:|
| 完整成功回合 | 29 |
| 仅恢复合格、非完整任务成功 | 7 |
| 恢复阶段监督行 | 17,837 |
| 实际执行且最终成功的控制纠正行 | 947 |
| 含成功纠正的完整回合 | TRAIN67、TRAIN95 |

TRAIN1中未成功恢复的纠正没有进入正向监督。其他失败也只保留在本地分析清单中。文件同时区分未修改的成功命令、教师实际执行命令，以及成功混合控制回合中的学生回退命令；不把回退命令称为QP解。

每个正向文件保留完整回合的无损压缩公开观测及历史、最终归一化动作、最终关节目标、上一条实际关节目标、因果恢复资格、恢复已持续时间，以及可用时的学生位置请求、教师获准目标和实际介入标记。只在恢复资格成立且上一条命令已知的行上提供恢复监督。

资格门控仅使用此前已观察到的空中触球阶段，随后观察足接触恢复阶段；触发后保持至本回合结束，不随足接触短暂丢失而关闭。未来完整成功与恢复标签用于事后筛选训练数据，不用于该门控。准备阶段短暂离地后的接触不会自行触发这一门控。它是数据/学习接口候选，尚未作为新的原生自主接管逻辑验证。

源运行是确定性评估，缺少学生原始网络残差和采样log_prob。数据明确声明为监督数据，不能当作on-policy PPO rollout。学生位置请求也不冒充原始网络动作。

## 解决动作范围不匹配

此前已确认，大部分新增成功救援命令超出旧学生“基础目标±0.4 rad”的范围。此处保留总关节命令，另定义可供后续恢复模块使用的新软件坐标：

\[
a^{candidate}=\operatorname{clip}(a^{base}+2\delta,-1,1),\quad \delta\in[-1,1]^{18}。
\]

这里的动作是按原关节上下界归一化的**总位置目标**。随后按上一条实际下发命令施加机械臂原10 rad/s目标变化率，再经过原关节范围。它没有扩大硬件关节范围、物理速度、力矩或修改质量；也不是给旧残差标签换名字。

在恢复门控关闭时，参考组合函数逐元素返回原基础命令，不改写已有击球前缀。后续网络仍需要从观测、动作历史和当前基础命令预测修正，不能把案例编号或未来结果作为输入。

对全部17,837行，利用保存的真实基准位置请求/未修改基准命令求出的坐标，经这套映射重新生成目标，最大目标误差为0（本次数值审计）。原生日志与导出总目标的最大核对误差约3.42e-7 rad。**可表达和可还原不等于神经网络已学会，更不等于闭环成功。**

证据：[数据清单](verified_recovery_commands_dataset_20261107.json)、[动作坐标审计](verified_recovery_command_coordinates_20261107.json)。约19 MiB的NPZ数据仅保存在服务器，GitHub只上传程序与紧凑证据。

## 程序与验证

- [筛选导出](../scripts/export_verified_recovery_commands_v1.py)：独立重新计算回合结果，检查观测及命令哈希，拒绝覆盖旧数据。
- [门控与动作坐标](../quadsmash/rl/recovery_command_dataset_v1.py)：缺失的初始命令历史明确标记未知；不以实测关节角替代。
- [独立还原审计](../scripts/audit_recovery_command_coordinates_v1.py)：核对角色、资格、实际命令历史和动作表达范围。
- [7项测试](../tests/test_recovery_command_dataset_v1.py)：覆盖因果性、准备阶段接触、接触丢失、回合重置、缺失历史、越界标签拒绝及完整范围映射。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/python.sh scripts/export_verified_recovery_commands_v1.py --run reports/landing_mpc_command_history_repeat3_20261107_01 --command-audit reports/landing_command_history_repeat_20261107/native_command_audit.json --output /tmp/verified_recovery_commands
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/python.sh scripts/audit_recovery_command_coordinates_v1.py --dataset /tmp/verified_recovery_commands --output /tmp/recovery_coordinate_audit.json
```

本轮代码和数据工作没有替换原生基线：学生仍为27/128，已重复验证的持续辅助为29/128。11:54时，旧控制器26例覆盖试验实际进程仍在GPU0运行，已记录600/800 tick；[一致隐式PD候选](landing_implicit_command_qp_20261107.md)在队列中等待。后续先判断控制救援覆盖，再用结果筛选的数据训练可执行的恢复模块，并关闭额外控制评测学生。
