# 原生关闭生命周期修正与一次短诊断

2026-11-14。代码提交`7a0a892`；运行目录`/tmp/supermimic_shutdown_scope_20261114_E5Moig/run`。本轮承接想象模型审查，先处理原生作业关闭问题，不进行策略学习或完整任务评分。

## 修改与检查

将原生环境、wrapper、actor、控制器、观测及其临时张量放入独立函数作用域。该函数先关闭env，再返回；外层在应用仍存活时进行GC，然后调用原SimulationApp.close。异常时先保存完整堆栈，再清除已经退出的异常帧局部引用。清理错误不覆盖原始异常。

faulthandler使用一直有效到进程结束的文件描述符，不再在app.close返回时主动禁用；未使用os._exit、skip_cleanup、伪造退出0或修改site-packages。仍为fast_shutdown=False。

增加显式`--shutdown-diagnostic-ticks`，仅用于原128环境的短关闭诊断。默认正式模式仍800tick/8秒，物理参数、动作、源策略、lane、奖励与评分未改。短诊断只生成shutdown_diagnostic.json，不能产生metrics.json或completed.json；监督器将其与正式评估严格区分。

CPU：生命周期与终止评分测试18 passed，0 failed，退出0；包含正常/异常路径的循环引用释放顺序、原始异常保留、短诊断不得冒充正式评分、SIGSEGV不得被完成记录掩盖。语法检查通过。CPU fixture不替代真实Isaac验证。

## 唯一一次原生诊断结果

- 原128环境、seed882072、GPU PhysX、原source checkpoint与执行接口。
- 实际10控制tick，0.10秒仿真，1280环境控制步；循环15.386秒。
- 准备阶段新旧最终18关节目标最大误差0，标准仍2e-6。其他四阶段未覆盖，本次不重复声称全阶段复验。
- actor/critic/world更新均0，完整首回合0。本次无成功率，不能将0个完整回合称为0/128失败成绩。
- SHUTDOWN_DIAGNOSTIC_FINISHED → ENV_CLOSE返回 → GC返回（collected=0）→ APP_CLOSE返回 → CLEANUP_FINISHED。
- 随后进程SIGSEGV：child_returncode=-11，监督器退出139。
- 进程1669784已结束，执行会话60662已返回139。没有自动重跑。

**结论：提前释放仿真局部对象并没有修复关闭崩溃。** 这次排除了“只需将局部作用域提前结束即可解决”的简单假设，不能排除框架全局引用、原生析构、插件卸载或退出回调问题。GC=0不证明所有外部原生对象已销毁。

native_fault.log仍为空，即使本次未主动关闭描述符/禁用处理器。因此，之前将空日志归因于脚本提前禁用处理器，只能解释原代码的覆盖缺口，不能当作已经确定的全部原因。本次没有可用C++堆栈，不能指定某个插件为根因。

## 本机源码提供的新线索

读取已安装Isaac Sim 5.1的SimulationApp.close：它在shutdown_and_release_framework前调用`_logging.set_log_enabled(False)`。所以Kit日志停在“shutting down app and releasing framework”本身不能定位崩溃点；它与外部已记录APP_CLOSE返回并不矛盾。源码摘录及SHA256已保存，未修改安装文件。

关闭后原生信号处理/解释器退出路径仍待定位。下一次应在外部原生调试器中捕获SIGSEGV及所有线程回溯，而非继续盲改释放顺序或反复运行完整8秒评分。本机`/usr/bin/gdb`存在，core size为0；本轮只检查可用性，没有另开gdb原生运行、启用大core或安装依赖。

## 范围与保留

只做一次短原生诊断，未启动训练、未修改模型时域/损失/奖励、未追加完整128评分。原27/128初始化复验仍是上一轮记录，本次没有新任务表现。故障仍未解决；本轮到此保存并停止，后续继续从原生栈定位，而不是把空日志或CPU通过当作关闭成功。

本目录保留启动提交和命令、实际配置、原生日志、Kit片段、空fault日志、命令对照、退出码、监督器结果、阶段时间、测试和文件哈希。旧失败记录、训练权重及其他进程未改动。
