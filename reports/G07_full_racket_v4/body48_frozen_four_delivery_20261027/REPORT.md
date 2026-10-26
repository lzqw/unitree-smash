# 四版固定128例资料交付

[下载180MB四版固定128资料包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_body48_frozen13_24_36_48_full_TRAIN128_weights_and_records_20261027.zip)。已实际上传并新下载校验，180377663字节，SHA为cd2316cb69ab404796ab7e641c9ddbd063c1e4984a3c182fdbff647b038d6cb0；134成员CRC、133份内容SHA一致。

新目录按清单先还原两份依赖，再叠加本包，包内RSL、四个公共actor与诊断/最终checkpoint推理一致；5599条有效首回合公共输入对应的CPU/GPU动作最大差分别6.4373e−6、1.66893e−5、2.5034e−6、4.41074e−6，均小于原5e−4回放容差。这里复算命令，没有重新进行物理仿真。见[新目录检查](FRESH_RESTORED_FOUR_DATA_CHECK.json)及[上传凭据](UPLOAD_RECEIPT.json)。

包内报告保留封存时的流程说明，具体完成结果见MANIFEST.actual_four_checks与REPRODUCE。本仓库报告记录后续上传验证。

第13/24/36/48版已实际完成原固定TRAIN128，各版完整成功为1/0/1/0，空中触球58/45/42/7，合格恢复32/24/21/0，数值故障均0。原2×2米目标、全部物理与合法性条件保持，当前继续保留原教师2/128。正式A/B400尚未执行。

本资料包保留每版全128真实50Hz身体/球姿态、前两例100Hz公共输入/动作及有效掩码、原始回合/触球/规划/初态/作业凭据、实际公共权重、原生源码快照和全部成对比较材料。先独立重算回合，核对checkpoint/public actor，并用记录的公共输入检查CPU与实际GPU动作；随后封包、验证内容，再把运行资产伴随包、完整48训练包及本包依次还原到新目录复核。所有恢复检查都从恢复目录运行，步行actor使用包内相对路径。

当前这是实际完成数据的归档流程，封包和新目录检查的真实结果将写入ARCHIVE_CHECK、FRESH_RESTORED_FOUR_DATA_CHECK和READY_FOR_UPLOAD。不预先声称这些检查通过，不把CPU命令回放当作新的物理仿真，也不继承旧教师或旧模型成绩。
