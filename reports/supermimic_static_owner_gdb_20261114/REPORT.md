# PhysX退出异常：静态Python列表的创建路径

本次运行代码为`afb867570eeccef1573a61510097d43b8a48cea4`，源码diff为空。原Isaac解释器、GPU0、原128环境与seed882072，10控制tick退出诊断；不是完整任务评估。实际14.69秒原生循环、1280环境控制步，actor/critic/world更新均0。准备阶段最终命令误差0，不能据此验证其他阶段。

## 确认的证据

GDB在当前安装的`_physx.cpython-311-x86_64-linux-gnu.so`静态对象初始化处设置断点，读取内存与栈，未修改控制状态或库文件。对象确认为Python list、长度1。创建路径为SimulationManager.initialize_physics、warm_start、SimulationContext.reset、DirectRLEnv构造和项目环境构造。

env.close与app.close返回后仍SIGABRT：`__run_exit_handlers → _physx静态析构 → list_dealloc → PyThreadState_Get`。此时Python正在finalizing且当前线程状态为空。GDB退出码0不等于被调试程序成功，本次记录为原生进程异常。

二进制偏移仅适用于本机安装版本：静态槽0x19ae60，初始化断点0x1107b5，析构0x112a40。完整栈见gdb_native.log；argv、解释器、脚本哈希及提交见startup.json。

## 边界与停止点

安装的Cloner注册replicationAttachFn并返回长度1的排除路径列表，其场景与静态对象吻合，是具体所有权线索。但未直接证明回调身份，也未证明unregister_replicator能清除该静态对象。

没有修改site-packages、物理复制、执行器、碰撞或评分，没有使用强制退出或返回0掩盖异常。本次停止于证据保存，无新性能成绩。这是退出生命周期问题，不能解释模型内球位置漂移；想象任务问题见[对应性审计](../supermimic_imagination_alignment_review_20261114_113503/REPORT.md)。
