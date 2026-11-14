# 原生调试器定位关闭SIGSEGV

2026-11-14。本次使用`3b0a13c01f19a0674aed9db0af3b9d53e605cb4b`代码，不改算法/参数/环境，不安装依赖。一次GDB下的原128环境10 tick关闭诊断，运行目录`/tmp/supermimic_shutdown_gdb_20261114_inYnUo/run`。源策略、seed882072、GPU PhysX和0.01秒控制周期保持。

## 关键进展

此前只能知道“env.close和app.close都返回后崩溃”。现在捕获了实际原生堆栈：

```text
SIGSEGV, invalid address 0x118
libomni.physx.tensors.plugin.so
_physicsTensors.cpython-311-x86_64-linux-gnu.so
carb/_carb.cpython-311-x86_64-linux-gnu.so
_Py_Dealloc / _PyObject_FreeInstanceAttributes / subtype_dealloc
frame_dealloc
BaseException_clear / OSError_clear
delete_garbage
```

完整当前线程40帧见[crashing_thread.txt](crashing_thread.txt)，全部线程及动态库路径在[gdb_native.log](gdb_native.log)。PhysX插件为本机`omni.physx.tensors-107.3.26+107.3.3.lx64.r.cp311.u353`。部分C++库无调试符号，尚不能给出插件内函数名和源码行号。

**可以确认：故障在关闭后的Python对象回收期间进入了PhysX tensor析构路径。** 调用链包含异常对象与堆栈帧释放，支持检查“被保留的异常/帧导致物理对象晚于框架释放”的方向。不能仅凭这些帧确定是哪条业务异常、哪一个全局引用或哪一个tensor实例负责；也不能直接定性为NVIDIA插件自身缺陷。

这不是world model loss、球拍几何预测、评分逻辑或执行中GPU动力学求解的失败证据。那些想象模型缺口仍单独存在，关闭修复不会自动提高策略质量。

## 实际执行

- 128环境×10控制步=1280环境控制步，0.10秒仿真。
- 准备阶段旧完整执行链与新入口最终18关节目标最大差异0；其余阶段未在本短诊断覆盖。
- actor/critic/world更新0；完整回合0；没有metrics.json或completed.json，不报告0/128或任何新成功率。
- lifecycle再次记录env.close返回、GC返回、app.close返回及CLEANUP_FINISHED，之后GDB捕获SIGSEGV。
- GDB PID1670995与被调试进程1671009均已结束。无自动重启、无第二次原生运行。
- `gdb_exit_code=0`仅表示批处理调试命令完成，**不是被调试程序成功**。本次明确保存`inferior_stop_signal=SIGSEGV`、`process_completed_cleanly=false`，不把调试器0写成仿真进程0，也不虚构本次原生退出139。

## 对上轮假设的修订

把运行局部作用域提前释放并执行GC，仍无法消除全部存活引用。上轮GC collected=0不能解释为没有物理对象；本次堆栈证明退出时仍有相关对象被销毁。

已读本机Kit的GC配置：启动时可能暂时禁用自动GC，app-ready时恢复。但显式gc.collect并不受自动GC开关阻止，因此不能仅用“GC被关闭”解释本次失败。没有据此改全局GC或site-packages。

当前异常路径清理只处理从_run_native向外抛出的异常；本次原生运行正常完成，没有进入该except。若某个依赖内部捕获并保留异常，外层清除传播异常帧不会触及它。这是下一步要检查的引用来源，不是本次已经证实的具体责任模块。

## 下一步及本轮停止点

在框架仍有效时检查存活异常及其traceback：定位仍引用原生env/asset/view的帧及持有者，或在GDB进一步读取对应frame/object类型。只针对确认的持有者做释放，避免粗暴清空所有异常/模块、跳过cleanup或使用os._exit掩盖问题。

本轮找到新的具体故障路径后保存并停止，未盲改析构器、未追加完整8秒评分。后续需验证关闭修复，再继续物理状态数据采集及想象模型一致性工作；当前主目标仍未完成，任务性能没有新提升。

## 保存证据

原启动元数据、任务与GPU配置、10步命令对照、生命周期、GDB命令、完整输出/所有线程/库列表、SIGSEGV摘要、空faulthandler日志及文件哈希均保留。调试器未联网下载符号，未生成大core。旧权重、旧报告及其他GPU进程未动。
