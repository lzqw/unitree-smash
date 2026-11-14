# 来源扫掠审计合同修复

2026-11-14，基于78953ae的失败证据。

## 修正

- 评估的sweep-audit-iterations默认改为None，保留源factory的4096，不再把0写入环境。显式值只允许与源合同相同。
- 训练入口同样存在写死0的覆盖，已删除。
- task contract升级v2，记录maximum_iterations=4096和原tolerance_m=1e-6，在环境构造前检查实际cfg。旧合同checkpoint/replay不会静默resume。
- 没有改变PhysX碰撞响应、物理步长、100Hz控制、策略权重、奖励、成功判定或时域。
- 补充AppLauncher之后的SIGINT/SIGTERM请求标志，在控制步边界保存错误和已完成tick，避免只依赖Kit信号处理。尚未通过真实运行中的信号退出试验，不能宣称原生退出行为已验证。
- 每100tick保存已终止首回合数值审计计数、当前回合未决与漏检计数，分清“检查未决”与一般非有限数值。

## CPU证据

指定测试：test_supermimic_task_contract、test_native_action_contract、test_supermimic_horizon_v2。

**62 passed in 1.60s，退出码0**。包含源factory默认值核对、默认继承、明确相同值、拒绝不同/零覆盖、有效cfg被污染时拒绝、0迭代扫掠返回未决的几何反例。三文件py_compile与git diff --check通过。

## 原生复验计划与边界

另建唯一目录，使用原source policy、seed882072、原128顺序、origin0、64米地面、0.01秒控制及完整8秒；无world/actor/critic更新。当前仅修复上述审计及诊断处理，不接入新物理观测或执行预览模块。

提交此报告时原生复验尚未完成，不能报告新成功率。旧0迭代评估及全部部分记录保留，不补写completed。新的性能结果只能来自完整新运行；遇错保存并停止，不自动调参重跑。
