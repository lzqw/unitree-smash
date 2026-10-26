# 实际全身八项比较与独立冷128

本包针对实际完成的全身增益、起跳、走位和恢复交接八项比较，以及随后独立启动的同策略原128例冷评估。`search8/metrics.json`与`cold128/metrics.json`分别给出搜索和独立评估；不把搜索成绩移用为冷评估成绩。全部原始回合、实际Native回合编号、地面初态、接触事件、50Hz机器人与球状态、实际公共输入及动作均保留。

复现需使用原GPU0运行环境。每项的`job.json`保留实际命令、Git版本、种子、时间和资源记录；`source_snapshot.zip`与`source_hashes.json`保存实际生效的400份源码、配置与USD快照。恢复这些快照后，按`job.json`中的原搜索/评估命令使用全新输出目录运行，不覆盖本包证据。外部依赖仍包括已锁定的IsaacLab、IsaacSim、Torch和walking策略。相对或绝对USD引用需要按原路径恢复，或经审计后调整。

完整USD/网格及walking运行依赖沿用已发布的[G07运行资产包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip)，SHA256为`44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb`。当前包不是独立安装的仿真器。

`delivery/ACTUAL_COMPLETED_DATA_CHECK.json`重新核对1024个搜索回合和128个冷评估回合、地面初始化、原协议、实际源权重、CPU对实际GPU动作及400份快照SHA。它也比较当前冷评估和上一轮prepose冷评估中共同有效的实际50Hz采样点；此比较只描述已记录轨迹的一致性，不把旧成绩作为新成绩。

`diagnostics`包含整轮搜索报告、各项实际结果、起跳/恢复时序对照和第一次身体触地附近的几何诊断。几何诊断区分实际传感器汇总触地和视觉网格近地，不声称保存了未记录的逐刚体接触力。历史报告中运行中进度是当时的记录，最终实测以本包两份`metrics.json`和独立检查为准。

正式A/B各200例及50%完整、70%空中触球、85%条件恢复、零数值故障的原门槛保持。本包不读取正式案例，不宣称正式通过；策略仍需继续训练与独立验证。
