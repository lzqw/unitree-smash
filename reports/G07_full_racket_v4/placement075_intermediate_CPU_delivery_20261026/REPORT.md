# 075实际中间16/24权重及CPU复核包

[下载57MB证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_native_intermediate16_24_CPU_export_evidence_20261026.zip)。57374471字节，SHA256为18f53e0d36fb8ed8a9d9ea2f42a7952b8908e8631813079ffb07f4f65a3a9770；上传后已重新下载核对SHA、全部成员CRC及151份payload SHA。

这是实际运行中的075严格续训已经执行的中间权重，不是新训练或冻结成绩。第24版在确认原生已完成第25次更新后才导出；没有中断训练或启动额外GPU任务。

| 阶段 | 实际公共输入核验数 | 导出动作最大误差 | actor RMS | critic RMS | 相对源8的17个Adam各前进 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 1366 | 0 | 14680064 | 4194304 | 160步 |
| 24 | 1366 | 0 | 16777216 | 6291456 | 320步 |

源8的actor RMS12582912、critic2097152；旧actor归一化偏移10485760保留。原固定base846、腿0.03/臂0.15、网络及公共观测不变。16/24的model_i是非canonical诊断，供冻结推理与来源检查，不能当成严格恢复检查点。

包内包含两份policy.pt及对应原生model_15/model_23、源canonical8/public8、固定base846、1366条实际已保存输入和元数据、导出脚本、项目Python源文件、实际使用的RSL源码及BSD许可证/本地差异。清单记录软件版本和每份文件的SHA。CPU所需数据完整包含；Python、Torch、numpy、scipy、tensordict仍需外部环境。没有包含全部USD/STL，不能据此宣称新目录PhysX重放通过；完整原生资产仍见已交付的293MB阶段8包。

已在新目录/tmp/g07_placement075_intermediate_cpu_restore_20261026_w_3d4wa5提取，用包内RSL源码重建源8、16、24；151份数据SHA、源8参考输出和两份中间策略独立前向均通过，误差均为零。FRESH_RESTORE_CHECK.json和UPLOAD_RECEIPT.json记录实际核验。

解压后运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python reports/G07_full_racket_v4/placement075_intermediate_CPU_delivery_20261026/check_restored.py --root . --output CPU_RESTORE_CHECK.json
```

包中的progress是当时仍在运行的真实发布快照，不是最终训练收据。冻结16/24/32和最终40的比较仍按原共享GPU0队列执行；当前最佳已测075冻结8仍0/128完整。正式A/B各200及原50/70/85门槛未通过，全目标保持未完成。
