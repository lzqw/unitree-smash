# 零更新复验停止：扫掠审计参数被默认值覆盖

日期：2026-11-14。运行提交65a1597a1c0930d89448af1b48a9731606d25754。
目录：`/tmp/supermimic_source_contract_eval2_20261114_UmizUq/run`。本次没有训练更新。

## 确定的问题

`evaluate_supermimic_v1.py`的`--sweep-audit-iterations`默认0，随后无条件写入`cfg.sweep_audit_iterations`。源`full_racket_cfg()`设置4096；来源合同校验遗漏了这项覆盖。

实际`physics.json`明确为0，部分已结束回合的numerical_failures出现数百计数。`BallEnv._consume_physics_frame()`会将扫掠几何unresolved计入sweep_unresolved，终止记录再将它与sweep_misses相加写成numerical_failures。

迭代数0并非明确“关闭诊断”：保守推进循环不执行，但粗筛候选仍作为未决返回。CPU构造同一球穿过拍线平面：0迭代为hit=False/unresolved=True；4096迭代为hit=True/unresolved=False，详见cpu_repro.json。

这说明这些计数不能被直接称为PhysX数值爆炸或策略权重错误。几何审计不施加碰撞力，实际碰撞响应仍由PhysX完成；本证据也不支持把落地失败因果归结为扫掠参数。但本次已不满足历史相同审计配置，不能报告成完整初始化能力复现。

## 已采取的动作

发现后停止，无调参重跑。SIGINT及SIGTERM发出后同一PID仍存活，核对PID、脚本和输出路径后对本任务评估PID1649118使用SIGKILL；外层真实退出码137。不是自然崩溃，也不是完成或收敛。停止请求原文随附。

没有Python异常堆栈可归因，不伪造failure.json。没有completed.json，没有metrics.json。最后落盘进度500/800 tick，首回合终止记录共5例；进度保存间隔为100tick，**精确中断tick未记录**。不以部分回合推算128环境成功率。

截至最近500tick的独立旧/新最终18关节目标最大差仍为0，五阶段均覆盖；该执行接口检查仍有效，但不能替代整回合、全部审计条件一致的原生复现。原初态相等的证据也保留。

## 对先前报告的更正

此前“恢复来源合同”只覆盖了已核验的布局、地面、资产、控制周期与恢复关联，尚未覆盖扫掠参数。此前运行中快照保留，不能继续解释成所有评估设置均已一致。相关报告加入本次停止说明。

## 最小后续修复（本轮未实施）

评估参数默认应为None并继承源cfg；正式复验拒绝与来源配置不同的覆盖，将有效扫掠参数加入task contract及启动前校验。训练入口的同类覆盖也需核对，旧数据仍保留为诊断。增加CPU配置分支与0迭代几何反例测试，不放宽阈值、不修改PhysX或成功定义。修复并检查后另建独立运行，不能补写本次完成文件。

本轮新提交的物理观测重建和执行预览仍是独立CPU模块，未导入该进程，没有修改运行中的控制链。目标仍未完成，尚无新的正式成绩。
