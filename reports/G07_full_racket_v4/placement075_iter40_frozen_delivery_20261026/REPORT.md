# 075冻结40：当前实测效果

最终40策略已在原全TRAIN128、零扰动、seed724201下完成原生评测；102400控制环境步，0.301758057 GPU小时，原guard与native进程成功退出。全部128回合由原始episodes独立重算，与原生metrics和父流程结果一致。

| 同一原TRAIN128 | 冻结075第8版 | 冻结075第40版 |
|---|---:|---:|
| 完整扣杀 | 0 | 0 |
| 合法空中触球 | 26 | 22 |
| 空中触球后合格恢复 | 19 | 10 |
| 条件恢复率 | 73.08% | 45.45% |
| 合格攻击出球 | 5 | 0 |
| 数值故障 | 0 | 0 |

40版没有保持第8版的触球和恢复表现。此前17001个变化中TRAIN回合的5次完整执行，不能作为冻结40的成功率。正式A/B各200及原50/70/85门槛仍未通过；没有读取正式面板进行选版。

[独立重算及来源SHA](ROOT_FROZEN40_INDEPENDENT_CHECK.json)记录实际128例、权重和原生作业回执。当前公共权重SHA为bf8ff2a1789f7aba528446b65e90c13d6b219317632dfe667be6b7452662b828，完整球拍和规则保持。

本次六例视频继续使用训练前登记的TRAIN19、23、107、0、8、58，保留真实失败。原父流程先完成已有镜头导出并退出；隔离CPU视频队列随后使用同一原生轨迹生成修正取景版，须独立审阅后再上传。现有中间16/24/32冻结比较仍由同一主GPU0队列顺序执行，不能把未测版本称为改善。

[当前40版实际视频（86.16秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_placement075_iter40_orthographic_v3_20261026.mp4)已上传并新下载SHA回验，4308帧全部解码；15张原生事件图、3张修正落点字幕及8张成品画面已人工检查。侧视/俯视及实际完整球拍保留；字幕明确区分几何落点与任务有效回球，原评测不变。视频SHA256 `7bcbb883d79942dc64ef30a2ed73ecd3a28908666273c98c00d1de32937a8eee`。

[视频代码、取景与画面审阅报告](https://github.com/lzqw/unitree-smash/blob/d4fc16badfb0b35091c4801f41845eeaeca54a0c/reports/G07_full_racket_v4/orthographic075_iter40_landing_labels_20261026/REPORT.md)位于隔离视频分支，原主GPU输入仍保持。
