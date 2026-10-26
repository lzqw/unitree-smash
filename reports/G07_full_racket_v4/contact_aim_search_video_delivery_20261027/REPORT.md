# 新瞄准点：42.28秒实际搜索视频

[观看视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_aim006_search_complete_and_failures_20261027.mp4)。TRAIN12新完整成功，TRAIN13击球和目标通过但身体触地恢复失败，TRAIN0拍框误击；三段相互独立，均保留原生episode_index2。侧视、俯视、完整回合和触球慢放来自实际记录。

该候选[0,.06]已完成原全TRAIN128：5完整、60空中、28合格恢复、21攻击、NF0。它是搜索阶段成绩，整体7点搜索和单独冷128在快照时仍未结束；此前确认教师仍2/128，不宣称正式A/B或50/70/85通过。

TRAIN12实际接触高1.735米，出球8.22251米/秒、vz=-2.48325米/秒，首落[3.957,-0.589]米，连续稳定4.15秒。TRAIN13原教师曾成功，本候选恢复失败，明确保留退化。

2114帧全部解码，显示身体/球/时钟逐项等于原始50Hz记录；8张原生事件图和8张成片抽样已检查。CPU复算1244条有效公共输入动作最大误差5.96e-6，原容差5e-4。新上传3621158字节视频SHA bbe4560f72336f548858fe304db40f21484d9edf0ac9b41675d79f7870ef1d73，已从GitHub重新下载核对一致。

[46.49MB完整候选原128记录、对应权重和视频证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_contact_aim006_completed_candidate_TRAIN128_weights_and_video_evidence_20261027.zip)：46485655字节，SHA 0dea35c44a8e663b333e5480d1f976e5a8137d3fab50c3e33cd430d0984ac9f9；新下载49成员CRC和48份payload SHA全部通过。包含全128实际50Hz姿态、原始接触/回合/计划/重置、原生源代码快照、对应完整嵌入源策略。它不是最终7点选优或单独冷评估包，外部Isaac/Torch与此前运行时伴随包仍需另外提供。

[完整制作报告、代码与两次准备修正](https://github.com/lzqw/unitree-smash/tree/g07-native-video-framing-v4-20261026/reports/G07_full_racket_v4/contact_aim_search_video_20261027_v2)。GPU0继续当前原生作业；[已登记下一轮引拍/腕部比较](../prepose_branch_after_aim_queue01/REPORT.md)等待当前父1827656真实退出，不更改419项活动输入。

原结构验收命令`python tools/verify_goal.py --root . --goal G07`已实际运行，退出2：新版完整G07正式evidence尚未生成，不能标PASSED。该工具仅查结构和阈值，视频、CPU检查与搜索成绩不替代原A/B各200。
