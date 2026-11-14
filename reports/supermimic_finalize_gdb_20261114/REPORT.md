# 缓存修正后的SIGABRT：PhysX绑定静态终结器晚于Python线程状态

2026-11-14，源码`d6d24bb2b2ca272cb0fa738575ee0ae1b5572d01`。保留上一轮Kit stat-cache重置，不修改算法、物理、奖励、时域或控制接口。本轮只运行一次GDB下的原128环境10tick诊断。

## 本次定位

实际栈为：

```text
__run_exit_handlers(status=0)
  → omni.physx/bindings/_physx...so
  → list_dealloc
  → PyThreadState_Get
  → fatal_error: current Python thread state is NULL
  → abort / SIGABRT
```

与此前`OSError_clear → frame_dealloc → physx.tensors`的SIGSEGV不同，本次是**系统C/C++退出处理函数释放Python列表时，Python线程状态已经失效**。问题位于`omni.physx`绑定，不是原生10步控制期间的动力学失败。

完整当前线程栈、全部线程、动态库映射及启动命令已保存。本机绑定缺少完整调试符号，尚未确定该列表的业务用途或变量名。

对本机二进制做只读反汇编核对：崩溃返回点偏移0x112a61来自0x112a40处的引用计数析构函数；0x1107bf通过__cxa_atexit注册该析构函数，静态对象地址偏移0x19ae60。它最终调用_Py_Dealloc。偏移仅适用于debugger_result.json中保存SHA256的这个二进制，不能跨版本复用。

这项证据把定位从泛泛的“GIL有问题”缩小到**绑定库中的静态Python对象在C运行时退出时析构**。没有手动调用私有终结器、修改二进制、dlclose强制卸载、跳过cleanup或用os._exit掩盖错误。

## 公开接口核查与未完成部分

本机PhysxExtension.on_shutdown已经声明调用release_physx_interface与release_physx_interface_scripting；单凭读取此源码不能证明本次每个扩展关闭回调都完整运行，也不能证明该API负责这个静态列表。

绑定还提供复制/克隆相关回调等返回Python列表的接口。当前没有足够证据把这个静态列表指定为某个replicator或接触回调，不据此关闭物理复制、修改128环境布局或手动释放接口。下一步应追踪该静态对象初始化时的调用路径及扩展实际关闭顺序，找到有依据的释放点。

## 执行范围

原128环境、seed882072、GPU PhysX、源策略、完整执行器条件不变。实际10控制tick、0.10秒仿真、1280环境控制步；准备阶段新旧最终18关节命令误差0。actor/critic/world更新0，没有完整8秒任务评分。

生命周期仍记录缓存重置、GC、app.close返回与CLEANUP_FINISHED，然后被调试程序SIGABRT。GDB批处理退出0仅代表调试器命令完成；`process_completed_cleanly=false`，未把它写成原生成功。GDB与被调试进程均已结束，无自动追加运行。

## 本轮停止点

现在有具体的静态析构路径，但尚无已验证的安全修复。继续保存原非零状态，不放宽进程完成标准。下一轮围绕该对象的初始化/清理路径定位，不再反复猜测普通env局部引用或GPU算力。

没有新的任务成功率；先前27/128初始化复验和世界模型不一致的结论均不改变。关闭工程故障与策略性能分开报告。本轮发现新的具体退出路径后停止，未启动长训练或完整评估。
