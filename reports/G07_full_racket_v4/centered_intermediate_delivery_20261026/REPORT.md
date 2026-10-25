# 已完成的16/24/32版原TRAIN128评测与五版选择

原GPU0 guard已分别成功完成16、24、32版全部原TRAIN128、零扰动、seed724201评测，每组102400控制环境步。所有128个首次地面回合都保留，没有删除难例。五版比较结果均为0完整成功；未读取新正式面板，原A/B各200及50/70/85、数值故障零的门槛仍未通过。

| 来源更新 | 完整 | 合法空中拍线 | 条件恢复 | 恢复/空中 | 合格攻击 | 数值故障 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0 | 26 | 13 | 50.00% | 3 | 0 |
| 16 | 0 | 25 | 13 | 52.00% | 3 | 0 |
| 24 | 0 | 21 | 9 | 42.86% | 1 | 0 |
| 32 | 0 | 18 | 9 | 50.00% | 0 | 0 |
| 40 | 0 | 22 | 10 | 45.45% | 4 | 0 |

完整→攻击→条件恢复次数→空中→同分较早更新的排序已经在这些中间结果产生前登记。实际选择40版，公共策略SHA256 `2e1bf93056eff22dd309ea52a777ce0648aa50e025f63ec99e5ae8fe662fd84f`；它的空中和恢复弱于8版，不能称整体改善。来源16/24/32是已实际测过的公共推理导出，原非canonical诊断没有被当作可恢复检查点。最终 `RESULT.json` 的SHA256为 `f575de6d0d57c1ae5e1b63029841d278a8207a87fa405e2f1389faba5d704130`。

下载[更正后的完整评测与最终选择包（V2，约139MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_intermediate16_24_32_frozen_TRAIN128_and_final_selection_20261026_v2.zip)：138843404字节，SHA256 `d94790e73bf7c7d2e48126716045ddc3ae90355199506881db673d8c4339fd3f`，99个成员、98份源文件。包含三版全部原始回合/接触/姿态、公共权重与导出检查、实际作业/资源回执、运行时源码快照、逐回合CSV、最终五版选择及物理协议。归档内33项必需原生成员、全部成员CRC与内容SHA均通过；上传后重新下载再次验证，三版首次回合身份和公共权重SHA、实际数组形状均独立核对。见 `ARCHIVE_CHECK_V2.json`、`ARCHIVE_UPLOAD_RECEIPT_V2.json`。

记录覆盖范围：全128例有50Hz的全身姿态与球轨迹，数组为400×128×28×7及400×128×3；完整794维公共观测和18维关节动作仅记录前2例，100Hz，分别800×2×794及800×2×18。各数组均有首次回合有效掩码，回合结束后的数据不能算该回合。没有全128例完整动作记录的主张。

更正记录：初版归档说明误写成全128例也有动作轨迹，原始数据文件本身正确。初版保留供审计，当前交付以V2为准；`CORRECTION_FROM_V1.json`记录旧URL/SHA及准确范围。重新下载核验表明除打包程序外，初版95份既有源内容在V2中逐字节保持。首次CPU打包时读取回执层级错误也保存在 `VALIDATION_NOTES.json`，失败发生在创建归档前，没有修改原生数据或触发训练重启。

运行依赖与其他来源的原始证据采用已上传且新下载SHA回验的附包：

- [原生运行时、标准PiPER/Go2/USD依赖与源8（215MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_residual_probe8_native_training_and_initial_video_20261026_v2.zip)。
- [完整8→40训练、全部32个诊断边界及两段真实训练成功（843MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered40_completed_native_training_and_two_complete_videos_20261026.zip)。
- [固定8/40全部原128例对照、权重和实际40版失败视频（137MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered8_to40_frozen_TRAIN128_video_and_diagnostics_20261026.zip)。

当前新落点反馈0.75试跑已在原主GPU0账本运行，2048个真实地面状态及首个262144步更新已检查；该新训练不在本包中，也没有继承40版成绩。接下来仍由[已启动流程](../centered_placement075_pilot8_queue01/run.py)完成新8更新、独立原全128及同六例视频，再按实际结果推进完整目标。
