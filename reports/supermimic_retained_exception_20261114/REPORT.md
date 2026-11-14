# Kit异常缓存持有物理对象：定位、定向释放与下一处退出错误

2026-11-14。本轮从上一轮GDB捕获的PhysX tensor析构栈继续定位。诊断代码`2839f8a`，定向修正代码`3e52581`。两次均为原128环境、原seed882072、GPU PhysX、原策略及命令接口、10控制tick的关闭诊断；未启动训练或完整评分。

## 找到了具体持有链

只读扫描发现1463个带traceback的存活异常，其中1452个FileNotFoundError。**21个异常的帧/父帧链仍含`_run_native`，其局部env类型为实际EvaluationEnv**，同时保留wrapper、reference等对象类型。见retention_summary.json，未记录张量值或异常消息。

已安装Kit的`omni.ext._impl.stat_cache`会缓存`os.stat`返回值或异常，并在后续查询重复抛出同一个异常。保存的异常traceback及父帧由此可以保留原生运行函数，直到缓存失效或解释器退出。此前GDB的OSError_clear→frame_dealloc→PhysX tensor析构路径与这项证据一致。

第一次只读诊断仍退出139。原始扫描文件约205MB，原因是许多异常重复引用相同长帧链；原文件及gzip留在该次/tmp运行目录，哈希已保存，Git只上传紧凑摘要与原生日志。扫描约有一分钟量级的CPU开销；不是原生10步的采样耗时。扫描中的isinstance检查触发torch弃用警告，未将其误判为原生退出根因。

## 最小修正

仅在原生函数返回、env.close完成、Kit/PhysX仍然存活时调用已安装模块的`reset_stat_cache()`，随后GC，最后保持原app.close路径。

- 不修改site-packages，不重写os.stat，不清除全部Python异常/模块，不跳过cleanup。
- 未导入Kit的CPU进程不额外加载它；未知API明确报错。
- 不改变任何控制步、原生执行器、奖励、评分或学习算法。
- CPU测试由10项增加到11项，新增“负查询异常缓存保留资源→缓存重置→资源在app关闭前释放”的生命周期回归测试。11 passed，退出0；fixture不替代后面的真实原生复测。

## 真实复测：释放生效，但关闭仍未通过

修正运行：`/tmp/supermimic_stat_cache_fix_20261114_KKb1uP/run`。

| 检查 | 结果 |
|---|---|
| 实际原生执行 |128环境×10 tick，0.10秒仿真|
| 准备阶段新旧最终18关节目标误差 |0，标准2e-6|
| 重置前缓存 |6038条，其中1452个异常|
| 重置后缓存 |0条|
| 随后GC |回收30286个对象（上次为0）|
| env.close/app.close |均返回|
| 最终退出 |SIGABRT，child_returncode=-6，监督器134|
| actor/critic/world更新 |均0|
| 任务成功率 |不适用，未完成8秒回合|

新的明确错误为：

```text
Fatal Python error: PyThreadState_Get: the function must be called with the GIL held,
but the GIL is released (the current Python thread state is NULL)
Python runtime state: finalizing
```

**缓存导致的保留链已被定向释放，但整个退出流程仍失败。** 不能把139变成134认定为修好，也不能仅凭一次信号变化证明旧PhysX问题在所有路径永久消失。新错误指向解释器finalizing阶段仍有代码调用需要GIL的Python API；尚未定位到具体C++析构/退出回调。

## 本轮结束与下一步

两个原生进程均已结束，没有自动重试。下一步在保留缓存修正的情况下捕获SIGABRT原生栈，定位哪个退出回调在Python线程状态已经失效后调用API；不继续清空其他全局状态、不升级环境碰运气、不使用os._exit(0)伪装正常结束。

当前关闭问题尚未完全解决，已按“遇到下一问题停止”保存结果。本轮没有新学习性能；初始化27/128仍是此前完整复验，不能将短关闭诊断计入成绩。世界模型物理状态/事件/奖励一致性仍是后续学习重点，这个释放修正不替代它们。

## 保存内容

diagnosis/包含第一次只读诊断日志与退出结果；fixed/包含定向修正复测的启动提交、配置、命令对照、阶段、原生日志及134退出证据。retention_summary.json给出21条关联的计数和3个示例，完整重复扫描数据的本地路径/哈希见diagnosis/raw_audit_provenance.json。安装文件身份、CPU测试及所有上传文件哈希一并保存。旧权重、历史报告和其他GPU进程未改动。
