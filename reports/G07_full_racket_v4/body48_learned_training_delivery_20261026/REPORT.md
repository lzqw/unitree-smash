# 学习阶段实际完整回合与失败视频

[观看37.76秒视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_body48_learned_complete_and_early_failures_20261026.mp4)。保留TRAIN32的一次学习阶段完整8秒回合，以及同轮预热期的TRAIN23拍框误击、TRAIN58漏球身体触地两段失败。阶段不同、含随机探索、经过后选，不据此比较或计算固定成功率。

成功例原生环境1824、episode3，全回合跨采集iteration11至17，击球发生于iteration13，均在actor开始学习之后。实际出球8.152518米/秒，首次落点[3.933655,−0.419493]米，连续稳定恢复1.999037秒；原2×2米目标、攻击与恢复门槛全部保留，独立完整判定通过、全部失效标志为假、NF0。

视频1888帧全部解码；显示采样逐项等于原始100Hz记录每隔一帧取样的结果，7张事件预览及8张成片采样已检查。离线回放曾错误使用后续回合的全局步行相位0；恢复实际控制步1439的相位后，800步命令最大误差3.576279e−6，小于原5e−4容差。原训练程序与物理轨迹均未改动。

[32.97MB精选证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_body48_learned_phase_selected_training_video_evidence_20261026.zip)包含三例完整原始采样、事件、导出几何、源教师公共权重、第13更新非canonical诊断、渲染与核验资料。它不包含整个正在运行的48更新数据集或完整Isaac安装；完整仿真依赖使用此前交付的伴随包。视频与证据包均已从GitHub重新下载，SHA/字节数一致；证据包67成员CRC、66份内容SHA全部通过。

[视频分支的完整制作报告与依赖说明](https://github.com/lzqw/unitree-smash/blob/g07-native-video-framing-v4-20261026/reports/G07_full_racket_v4/body48_learned_training_video_20261026_v3/REPORT.md)保留两次CPU准备失败、修正及全部证据。当前GPU0继续48更新及13/24/36/48固定128比较。新学习器尚无固定评估成绩；已有教师仍是2/128完整、55空中、30合格恢复，原正式A/B及50/70/85未通过。
