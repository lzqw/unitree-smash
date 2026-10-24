# G07 当前：最新真实触球视频已交付，GPU 训练批次已结束

继续前读原目标 /home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。仅 GPU0，无累计 GPUh、总工作窗或截止，保留有限单 job 超时、步数和费用记录。完整目标仍未达成，不标完成。

当前没有活跃 GPU 作业。ball_timing_grid01 已于 2026-10-24T04:48:46.431105+00:00 SUCCEEDED 执行结束，guard1329665/child1329677已退出，旧exec54326不可重复启动。实际 391.810751 s / .108836320 GPUh / 10176 控制-env步，账本25600是保守声明上限，不是实际执行量。

本批 16候选 × TRAIN[1,65] =32首回合执行，含重复参考策略。原生4次接触中3次己方合法空中拍线接触，1次拍框；0完整成功，0数值故障。3次拍线接触出球都偏弱且向上，随后因身体触地终止，未完成稳定恢复。拍框样例env20首次落地在己方[1.209758759,-.070777893,.025089461]、t3.177293062。所有失败保留，不能称正式成功率。TRAIN_RESULT_SUMMARY.json、episodes.csv、原始contacts/episodes/权重与源快照均在该目录。

当前用户请求的视频已完成全部1447帧解码核查、无错误，并上传Github核对大小SHA；scripts/render_full_racket_air_contact_v4.py用CPU llvmpipe回放实际28刚体/球轨迹和原生接触子步全身快照，侧视/俯视/慢放，无插值或虚构轨迹。只显示首回合有效姿态，不把早停后reset和弹跳拼成成功。首落点定格单独标明姿态与落地事件时间不同。输出reports/G07_full_racket_v4/video/Current_effect_air_contact_20261024_0450.mp4；28.94秒/1280×720/50fps，已上传同一Github release，旧视频保留。链接：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4。

此前low_base_branch01完成2代共32次实际TRAIN，同一参数策略两次合法空中拍线接触，0完整；第二代reset原生成功。其7,734,045字节证据zip SHA936b113d6d02c3649048fa92d717fd0e068b646d04fc9654d95285964a13125d已上传Github，旧job未保存触球lane全身连续姿态，不伪造其触球视频。当前grid新接触快照和全候选50Hz窗口已实际记录。

接下来持续推进实际全身优化：解决机身快速转动造成接触点朝己方、拍面向上、出球偏弱与身体触地；保持r3完整球拍/实际质量惯量/执行器/真实网场/原来球及门槛不变。必要时联合轨迹或残差学习，不凭静态IK/失败采样宣称不可行。继续扩大TRAIN128与扰动，再验证原生近网整拍/随挥返回边界、补齐外部USD来源锁、冻结源码权重、原A/B各200独立评测。必须完成合法实际空中拍线击球+攻击出球+真实越网首次目标落点+同回合稳定恢复，再交付成功/失败侧视俯视及旧新对照。原门槛worstpanel完整≥.5/空中≥.7/条件恢复≥.85/NF0，不能缩减。

既往上下文保存在reports/G07_full_racket_v4/NEXT_before_video_0450.txt及此前各NEXT快照。不得覆盖preexisting status/G07_RESOURCE_STOP.json，不停止无关GPU/MPS作业。
