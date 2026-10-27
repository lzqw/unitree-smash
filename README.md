# QuadSmash：Codex /goal 实施与训练任务包

本地进展（2026-10-27T07:18:58.530325+00:00）：GPU0继续提前横向第5组，前四组原128共512回合空中4/13/11/7，完整/攻击/恢复均0、NF0。新增[落地后支撑对照](reports/G07_full_racket_v5/landing_posture_diagnosis_20261027/REPORT.md)：恢复参考可从约51度倾斜落地，后续足支撑持续性和减速与失败明显不同；不收紧首次落地姿态。既有恢复/调姿/挥拍队列保持，本轮仅本地分析，无新增GPU作业或GitHub上传。

新实际结果（2026-10-27T06:36:20.494938+00:00）：提前横向七组搜索的[首个修改组](reports/G07_full_racket_v5/early_lateral_first_changed_20261027/REPORT.md)已完成并独立复算，原128空中拍线 **13次**，同轮原样对照4次；完整、合格恢复和攻击均0、NF0。仍是搜索结果，该候选尚未独立冷128，后续组继续，未宣称正式通过。

当前进展（2026-10-27T06:27:44.129301+00:00）：选中原样0的独立原128已实际完成 **0完整/4空中/0恢复/0攻击、NF0**。GPU0已接续7组提前横向拦截，来源实核及CPU重放通过，guard1975624/native1975639运行。[277.33MB九组及独立128完整交付](reports/G07_full_racket_v5/measured_swing9_delivery_20261027/REPORT.md)已上传并新下载核对全部138成员CRC与137份内容SHA；1280回合重算及45381条指令审查完成。**9/10项重放符合原容差，未选中第3组有17条手臂指令超差，已保留失败和投影数值定位，没有放宽容差或改历史权重。** 选中0及独立评估的指令均满足原容差。正式A/B和原50/70/85未通过，G07未完成。以下保留此前观察。

当前进展（2026-10-27T05:48:01.832555+00:00）：[九组挥拍比较](reports/G07_full_racket_v5/measured_swing9_completed_20261027/REPORT.md)已全部完成并独立复算1152回合，空中触球 **4/0/0/1/0/3/2/5/4，完整、合格恢复和攻击均0、NF0**，实际2.805232 GPU小时。原排序选择0原样，GPU0已接续独立冷128，尚无最终新结果。新增[公共观测腿部反向调姿](reports/G07_full_racket_v5/public_air_countermotion_20261027/REPORT.md)完成3×4800条真实输入检查，并增加高度/朝向/足支撑门控避免低位翻倒误触发；[有限后续试验](reports/G07_full_racket_v5/air_countermotion_queue01/REPORT.md)以CPU owner1973878等待恢复9项与其独立评估，1009项持有输入保持，无新增并发GPU作业。**模型预测不是实际恢复改善，正式A/B和原50/70/85未通过，G07未完成。** 以下保留此前观察。

当前进展（2026-10-27T05:29:03.324714+00:00）：GPU0原9组比较已完成前8组，空中触球 **4/0/0/1/0/3/2/5，完整和合格恢复均0、NF0**，第9组继续。新增[28刚体公共状态与动量模型](reports/G07_full_racket_v5/centroidal_recovery_model_20261027/REPORT.md)已用5600条公共观测、3200帧原生姿态核验，位置误差低于2微米；当前恢复参考队列保持。另更正[恢复时长字段](reports/G07_full_racket_v5/centroidal_recovery_model_20261027/RECOVERY_DURATION_ERRATUM.md)：3.729912和历史视频4.063888秒均为最长连续稳定时长，不是恢复完成时刻；原生判定不变。**没有新的闭环恢复成绩，正式A/B与原50/70/85仍未通过，G07未完成。** 以下保留此前观察。

当前进展（2026-10-27T05:09:59.049800+00:00）：GPU0原9组比较已完成前7组，空中触球 **4/0/0/1/0/3/2，完整和合格恢复均0、NF0**，第8组继续。[九项共享恢复参考](reports/G07_full_racket_v5/recovery_prior_native_20261027/REPORT.md)已接入独立原生入口，实际9×4800条指令与已公开参考逐字节一致；[后续恢复试验](reports/G07_full_racket_v5/recovery_prior_queue01/REPORT.md)已以CPU owner1968858等待短窗口试验及其独立评估，991项持有输入保持，无新增并发GPU作业。**恢复候选尚无新原生闭环成绩，原正式A/B和50/70/85仍未达标，G07未完成。** 以下保留此前观察。

当前进展（2026-10-27T04:52:32.598658+00:00）：GPU0原9组比较已完成前6组，空中触球依次 **4/0/0/1/0/3，完整和合格恢复均0、NF0**，第7组进行。新[短窗口挥拍控制](reports/G07_full_racket_v5/windowed_swing_20261027/REPORT.md)已实现，7项各4800条实际公共输入CPU核验通过：准备阶段指令逐位一致，原5rad/s限速保持。独立[后续队列](reports/G07_full_racket_v5/windowed_swing_queue01/REPORT.md)已以CPU owner1966524等待提前横向试验及其独立128评估结束，再审核实际来源并执行原全部128例比较和独立冷128；970项持有输入保持，无新增并发GPU作业。**窗口方案尚无原生闭环成绩，正式A/B与原50/70/85尚未通过，G07未完成。** 以下保留此前观察。

当前进展（2026-10-27T04:36:27.466734+00:00）：GPU0已完成原9组中的前5组，原128的空中触球数依次 **4/0/0/1/0，完整与合格恢复均0、NF0**；后续组继续，实际进度见status/STATUS.json。7组提前横向控制的等待流程仍在，未增加并发GPU作业。

已提取[真实恢复实录的共享动作参考与9项未执行候选](reports/G07_full_racket_v5/recovery_motion_prior_20261027/REPORT.md)，6.77MB权重/实录包和1.71MB六例数据已上传并新下载校验。从恢复的394份源码复算实际800行，以及每项4800行、全部9项的指令均通过，现场Python读取0。**种子仅恢复成功，速度和目标均失败；这些新候选没有原生闭环成绩，未提升为最佳来源。** [单独记忆保护的阴性结果](reports/G07_full_racket_v5/contact_guard_memory_20261027/REPORT.md)没有在4800条参考输入上改变动作，暂不占GPU试验。另已实核恒定挥拍速度会把来球前拍头移位约4–9cm，后续窗口化速度方案尚未实现或执行。原物理、全部来球、目标及正式A/B各200和50/70/85保持，G07未完成。以下为此前观察。

当前进展（2026-10-27T03:37:47.705465+00:00）：GPU0的[原9组挥拍方向比较](reports/G07_full_racket_v5/measured_swing_queue01/REPORT.md)已完成前2组，原样对照为 **0完整/4空中/0恢复**，首个挥拍改动为 **0完整/0空中/0恢复**，均NF0；第三组进行。[原样对照实核](reports/G07_full_racket_v5/measured_swing_reference_20261027/REPORT.md)保留实际4800条公共指令核验和后期恢复轨迹差异，没有声称全程逐位重放。

[全部原128例漏球诊断与提前横向控制](reports/G07_full_racket_v5/early_tangent_intercept_20261027/REPORT.md)发现124个未完成空中拍线的回合都向同一侧漏过，材料击球点横偏中位16.4cm（插值几何诊断，未改判原生结果）。9项设置各4800条真实公共输入CPU检查完成；两项高增益过度修正未登记，保留7项。[后续顺序流程](reports/G07_full_racket_v5/early_lateral_queue01/REPORT.md)已以CPU等待进程1957568启动，待当前9组与独立冷128完整结束，选择实测来源并再次CPU验证后，才通过同一Main GPU0 guard执行，没有新增并发GPU作业。原全部来球、物理、目标及正式50/70/85保持；7项目前没有闭环成绩，G07未完成。以下段落保留此前观察。

当前进展（2026-10-27T03:03:22.638063+00:00）：恢复关联v5的15更新与独立原128例已完成，最终冻结为 **0完整、0合格空中拍线、0合格恢复、NF0**。训练738回合的8空中/1恢复没有在冻结评估中复现，因此保留原教师同协议4次空中触球的来源。GPU0已接上[9组实测挥拍方向比较](reports/G07_full_racket_v5/measured_swing_queue01/REPORT.md)，每组均为全部原128来球，再独立冷128；945项源码与输入保持。原球场/球拍物理、2×2米目标和正式A/B各200、50/70/85门槛保持，G07未完成。

[143.69MB完成训练、15个诊断权重、canonical及冻结128证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_contact_recovery_v5_completed15_training_and_independent128_20261027.zip)已上传并新下载核验120成员CRC与119份内容SHA。[64.12MB完整源码、输入与全部10层USD资产配套包](reports/G07_full_racket_v5/contact_recovery_cold128_delivery_20261027/REPORT.md)已恢复1016项路径；新15策略从恢复源码复算1600条真实指令误差1.096725e−5、现场Python读取0。[完整报告](reports/G07_full_racket_v5/contact_recovery15_completed_delivery_20261027/REPORT.md)。

[正常比例的完整扣杀成功视频（14.82秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Smash_success_TRAIN29_normal_ratio_20261027.mp4)来自**上一布局TRAIN29历史成功**，全部画面明确标注；当前原点0新版仍没有完整成功，不能继承历史成绩。以下段落保留此前观察与历史证据，当前状态以上述及status/STATUS.json为准。

当前进展（2026-10-27T01:25:36.020208+00:00）：[新版触球飞行恢复关联已在GPU0实际评估原128例](reports/G07_full_racket_v5/contact_recovery_queue02/REPORT.md)。owner1939834、guard1939835、Native1939849存活，128真实地面初态、原点0及931输入保持已查；结果待出。修订在[独立程序分支](https://github.com/lzqw/unitree-smash/tree/g07-contact-flight-recovery-v5-20261027)，Main旧运行代码和历史评分保持；24项CPU测试、11360真实采样复算和1600条完整教师动作复算通过，不作新Native成绩。首轮缺URDF启动失败及用量保留，补齐原引用并确认旧进程退出后使用新02目录。后续顺序为新版真实冷源核验、15更新训练和最终冻结128；原物理、目标和正式门槛保持。最新可看的视频仍为下方70.16秒实际残差15失败录像。


当前进展（2026-10-27T01:04:23.750617+00:00）：[新origin0残差15训练与独立原128例均已完成](reports/G07_full_racket_v4/origin0_residual15_completed_delivery_20261027/REPORT.md)。最终冻结策略为**0完整、0合格空中拍线、0合格恢复、NF0**；3次实际球拍接触全为拍框。训练中721回合的7次空中触球不能当作冻结成绩。新策略未超过原教师同协议4次空中触球，保留原教师作为当前实测较好来源，正式A/B各200及原50/70/85门槛未通过。

[最新70.16秒实际视频：三次拍框误击与两个未触球对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_origin0_residual15_fixed_cold128_20261027.mp4)已上传并独立下载SHA回验，3508帧全解码、实际输入一致和视觉审核通过。[155MB完整训练、权重、程序、原128评估与视频包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_origin0_residual15_completed_training_cold128_and_current_video_20261027.zip)保留全部721回合和15个诊断检查点，159成员CRC/158payload SHA已新下载核验。此前73秒四次空中触球视频属于原教师，历史视频与成绩保留。

[只读恢复时序复现与分析](reports/G07_full_racket_v4/recovery_sequence_analysis_20261027/REPORT.md)已完成；原首128行、初始模型和12条共同轨迹一致。实际TRAIN69在同一合格触球飞行落地后，落地资格被后续短暂离地清除，随后稳定3.73秒且无机身触地仍未获得原资格。当前保留原评分，尚未运行修订版；该例速度和落点也不满足完整扣杀。当前本项目三项进程均已退出，下一步修正同飞行关联并原生验证。早期失败诊断保留于[原报告](reports/G07_full_racket_v4/origin0_residual_early_failures_20261027/REPORT.md)，其运行中措辞仅代表历史观察。

以下为历史进展，运行中措辞只代表相应观察时点，当前以以上及status/STATUS.json为准。

当前进展（2026-10-26T18:01UTC）：[最新42.28秒瞄准点实际视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_aim006_search_complete_and_failures_20261027.mp4)保留新成功TRAIN12、原成功例TRAIN13的恢复退化，以及TRAIN0拍框误击。参考点[0,.06]原128搜索为**5完整、60空中、28合格恢复、21攻击、NF0**；独立冷128现已完成为3完整、60空中、26合格恢复，NF0，仍未达标。2114帧全部解码、原生数据逐项一致、事件及成片抽样检查、新下载SHA验证均通过。[视频、46.49MB完整候选原128记录与权重](reports/G07_full_racket_v4/contact_aim_search_video_delivery_20261027/REPORT.md)已交付。此前[37.76秒48更新训练录像](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_body48_learned_complete_and_early_failures_20261026.mp4)继续保留。

当前原16米布局的已确认来源为Aim[0,.06]：**3完整、60空中、26合格恢复、24攻击、NF0**；完整3/128、条件恢复43.33%，原50/70/85未通过。[完整7点搜索＋冷128、程序和最佳权重304.6MB包](reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/REPORT.md)已上传并新下载131成员/130payload校验。原教师grid8冷128的2/55/30与[98秒历史教师视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_body_grid8_cold128_complete_and_failures_20261026.mp4)继续保留。

此前[48更新课程](reports/G07_full_racket_v4/body_residual48_after_grid16_queue01/REPORT.md)和13/24/36/48固定评估均已完成，完整成功1/0/1/0，未超过原教师；[180MB固定比较包](reports/G07_full_racket_v4/body48_frozen_four_delivery_20261027/REPORT.md)与[1.21GB训练包](reports/G07_full_racket_v4/body48_completed_training_delivery_20261026/REPORT.md)已交付。随后原16米布局Aim7和冷128完成，确认3/128。上述历史结果均保留；当前GPU0已切换到通过原生精度准入的origin0布局，重新测量策略，未继承历史成绩。

[原生坐标精度与重合隔离检查已通过](reports/G07_full_racket_v4/origin_precision_after_aim_queue01/REPORT.md)：同一来球128副本的最大自由飞行误差由原16米布局7.396毫米降至origin0布局0.173毫米，后者球/根副本偏差均0、球接触力0、提前done0、NF0。完整物理资产和全部原任务/评分字段保持。新[原生8×128引拍比较](reports/G07_full_racket_v4/prepose_origin0_after_precision_queue02/REPORT.md)已由父1884350/guard1884474/native1884488在GPU0运行，460输入保持，真实128地面初态及原点均已核对；原严格镜像失败后使用原限位内近等效种子，旧成绩不继承。新布局第一原样对照已[独立核验为0/128完整](reports/G07_full_racket_v4/origin0_first_reference_20261027/REPORT.md)、0合格空中拍线、11拍框、123身体触地；其余候选继续，不宣称精度修正已改善策略。

[落点范围离线对比](https://github.com/lzqw/unitree-smash/blob/ca0b3d7/reports/G07_full_racket_v4/target_scope_comparison_20261026/RESULT.json)在同一教师128例上得到原2×2米区2完整、4×4米区12完整、整个合法单打半场13完整，保留其余全部条件。用户范围选择待答复，当前训练与原成绩未改写。

[前三项origin0迁移与准备动态诊断](reports/G07_full_racket_v4/origin0_transfer_diagnostic_20261027/REPORT.md)已独立核对384行，均0完整/0合格空中拍线，NF0；来球释放时原样机身倾角19.14度，腿残差请求很小，不能认定它是主因。[下一轮8项全身起跳/接近/恢复适配](reports/G07_full_racket_v4/gain_body_after_prepose_queue01/REPORT.md)已完成1600真实输入CPU复算和规划后局部reset检查，等待owner1891402实际存活且无子进程，471输入SHA保持。当前1884350/1884474/1884488继续原比较；新作业尚未运行，未声称性能改善或正式达标。

[最新30.22秒origin0真实失败视频与37.01MB两候选原128完整证据包](reports/G07_full_racket_v4/origin0_current_video_delivery_20261027/REPORT.md)已上传并新下载SHA/CRC回验；保留拍框后摔落、站稳漏球、更换引拍触网三例。1511帧全解码，5张事件与8张成片抽样已查，1203有效公共输入CPU复算通过。当前前四候选均0完整/0合格空中拍线，GPU0继续第5项；该片只覆盖已结束的前两个候选，不继承旧成功录像或成绩，原正式门槛仍未通过。

[新origin0全身残差训练/独立评估接口](reports/G07_full_racket_v4/origin0_fresh_residual_interface_20261027/REPORT.md)已实现并上传；658真实公共输入上新零残差对源指令差0，CPU对原GPU最大3.09944e-6，原5e-4容差保持，4项CLI检查通过。尚未安装或启动新Native训练，完整冷源正向准入与真实训练/冻结评估待后续执行。GPU0前五项均0完整/0合格空中拍线、NF0，继续第6项；当前471输入与已登记全身时序/恢复后续队列保持。

[全身比较等待队列已在GPU启动前修正参数拼接问题](reports/G07_full_racket_v4/gain_body_after_prepose_queue02/REPORT.md)，原1891402与原程序/PLAN保留，替代owner1895642实际等待且无子进程，当前原生1884488继续。[新15更新全身残差训练＋冻结原128后续流程](reports/G07_full_racket_v4/origin0_residual_after_gain_body_queue01/REPORT.md)已由owner1896271实际等待；482输入保持，真实源绑定和正向冷源CPU核验通过后才会用GPU0启动，尚无新训练成绩。原物理、所有案例及正式50/70/85门槛保持。

[新origin0正式A/B各200的锁定、评估与汇总入口](reports/G07_full_racket_v4/origin0_formal_interface_20261027/REPORT.md)已实现并上传，保持原较差面板50/70/85和NF0。实际新布局0/128及旧布局冷源均被准入拒绝，正式文件打开0，6项CLI检查通过；正向准备和正式原生400尚未运行。384条实际落地记录的7个资格差异样本均无空中拍线，不据此改写评分。当前482输入和GPU0及两级接续保持。

[新origin0第7项已找到4次真实空中拍线](reports/G07_full_racket_v4/origin0_first_air_contacts_20261027/REPORT.md)：原128例独立核对为0完整、4空中、0条件恢复、0攻击、NF0。两例已首落原目标但身体触地；四例速度4.23至6.10m/s均低于8，另含一例球被网挡。该值仍是已结束单个搜索候选，整轮及冷128未完成，原门槛保持，不称为达标。

[origin0引拍8×128搜索已全部结束并核验](reports/G07_full_racket_v4/origin0_prepose_search_completed_20261027/REPORT.md)：1024首回合、4567有效公共输入复算，最大指令误差5.00679e-6；候选6为4空中/0完整/0恢复/0攻击、NF0，其余0空中。实际566784控制环境步、1.73833GPU小时，guard保守按819200上限记账。GPU0已接上最佳候选独立冷128（1900476/1900490），结果未出，不把搜索值当冷成绩；482输入和后续两级队列保持。

[全身时序比较的原样参照128已核对](reports/G07_full_racket_v4/gain_body_first_reference_20261027/REPORT.md)：4空中、0完整/恢复/攻击、NF0；668实际公共输入复算最大3.516674e-6。四次精确接触与前次冷评估相同，但首次终止3.17秒之后24个未获空中触球案例存在姿态差异，未证明全程等价且不推定原因。GPU0已进入关闭两层旧残差的候选1，整轮和新冷128未结束，原门槛保持。

[关闭两层既有残差的原128例已完成](reports/G07_full_racket_v4/gain_body_residual_off_20261027/REPORT.md)：0完整/空中/恢复/攻击、NF0，仅2拍框；参照为4空中。256条案例与实际回合编号、673条新公共输入复算通过，最大动作误差8.106232e-6。稳定标志2次不能替代接触后的恢复；该项未改善，继续保留残差做时序适配。GPU0已保存提前40ms起跳候选2的128真实地面初态，整轮及新冷128未完成，原门槛保持。

[提前40ms起跳的原128例已完成](reports/G07_full_racket_v4/gain_body_timing_results_20261027/REPORT.md)：0完整/空中/恢复/攻击、NF0；7次接触全为拍框，128身体触地。原128案例和实际回合编号4至7、682条实际公共输入已独立核对，最大动作误差2.861023e-6，400实际源码及482持有输入保持。该项未改善，GPU0继续推迟40ms起跳候选3；整轮、最佳冷128及后续残差课程尚未完成，原门槛保持。

[推迟40ms起跳的原128例也已结束](reports/G07_full_racket_v4/gain_body_timing_results_20261027/REPORT.md)：2空中、0完整/恢复/攻击、NF0；TRAIN69与112均首落原目标，但速度5.878/5.846m/s且身体触地。658条实际公共输入复算最大4.768372e-6，原案例/实际编号和400源码及482持有输入保持。提前/推迟均未超过原样4空中，GPU0继续恢复交接相位+0.15候选4；整轮及冷评估尚未完成，原正式门槛保持。

以下为历史进展记录，其中运行中措辞只代表当时观察；当前状态以上段与status/STATUS.json为准。

2026-10-26 GPU0已启动[保留两层策略的全身拦截与起跳搜索](reports/G07_full_racket_v4/fixed_actor_body_cem_gen2_queue01/REPORT.md)：10候选各跑原128来球、2代，随后单一冻结策略做原128冷评估。1280真实地面初态已核对，380来源文件加PLAN保持；新方法尚无终端成绩。最新85秒视频新下载SHA核验已通过，原正式50/70/85目标仍未通过。

2026-10-26最新视频已更新：[85秒预热13更新实际六例](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_leg06_full_cohort_warmup13_orthographic_v3_20261026.mp4)，原全TRAIN128为0完整、25空中、14条件恢复（56%）、1攻击、NF0。训练13更新和固定评估均已完成，当前本项目GPU作业已退出；[视频与核验报告](reports/G07_full_racket_v4/leg06_warmup13_video_delivery_20261026/REPORT.md)保留失败及上传证据。正式A/B和50/70/85仍未通过。

2026-10-26 当前视频已更新：[81秒leg06新8更新实际六例](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_leg06_pilot8_orthographic_v3_20261026.mp4)，4068帧全部解码、15事件图与8实际解码帧审查、新下载SHA核验通过。原全TRAIN128为0完整、27空中、14条件恢复（51.85%）、6攻击、NF0；来源0758为26空中/19恢复，恢复下降，[逐例对比](reports/G07_full_racket_v4/centered_leg06_after075_intermediates_pilot8_queue01/REPORT.md)保留。GPU0已实际接上[完整首批回合后再学习的13更新对照](reports/G07_full_racket_v4/centered_leg06_full_cohort_warmup13_queue01/REPORT.md)：父1777456、guard1777458/native1777471，2048真实地面状态已核对；302项持有输入保持，原正式A/B及50/70/85未通过。
2026-10-26实际已核对预热7→策略学习8转换：2048首批回合完整闭合，最新11/13更新、2883584步、NF0。[隔离的出球与提前拦截CPU分析](https://github.com/lzqw/unitree-smash/blob/29189cd2afffc801b4b0818e867ae50674528bbb/reports/G07_full_racket_v4/release_target_envelope_20261026_v2/REPORT.md)已对23个实际自由首落核验，5项必要检查通过；方向/时序几何候选尚无机器人可达、冲击或恢复证明，原全128完整仍0。main训练与原302输入保持，后续原128固定评估和真实视频继续。
[253MB leg06新8完整训练、权重和原全128数据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_leg06_pilot8_native_training_and_frozen_TRAIN128_20261026.zip)已上传并新下载核验，541成员CRC/540payload SHA通过；[解包、伴随资产与来源完整性报告](reports/G07_full_racket_v4/leg06_completed8_delivery_20261026/DELIVERY_REPORT.md)保留全部负结果和软件依赖。预热7对照当前实际3/13、786432步，actor权重保持而critic学习60步，NF0；原正式门槛未通过。
2026-10-26当前冻结效果：[第40版实际六例视频，86秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_placement075_iter40_orthographic_v3_20261026.mp4)已上传并新下载SHA回验，4308帧全解码，侧视/俯视、触球慢放和失败恢复均保留。075第40版原全TRAIN128、零扰动、seed724201独立重算为0完整、22空中、10条件恢复（45.45%）、0合格攻击、NF0；第8版为26空中、19恢复、5攻击，续训发生退化。[实测报告与独立核验](reports/G07_full_racket_v4/placement075_iter40_frozen_delivery_20261026/REPORT.md)保留全部128例来源。此前17001个变化中TRAIN回合的5次完整不能当冻结成绩，[29秒真实训练成功与失败视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_placement075_first_complete_and_failures_20261026.mp4)单独保留。GPU0已接上16/24/32比较，完整数据归档通过新目录核验。原A/B各200及50/70/85未通过。
[207.5MB中间16/24/32原全TRAIN128比较包V2](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_intermediate16_24_32_frozen_TRAIN128_and_source_diagnostics_20261026_v2.zip)已上传并新下载核验，490成员CRC及489份内容SHA通过；[新目录与依赖说明](reports/G07_full_racket_v4/placement075_intermediate_delivery_20261026/DELIVERY_REPORT_V2.md)明确原生资产伴随包。三版有效公共输入共4050条，源actor导出误差均0；原32版0完整/30空中/18恢复/4攻击，既定十来源排序仍选0758。

2026-10-26新增进展：[1.213GB完整075第8至40版数据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip)已上传并独立新下载核验，965成员CRC及964份内容SHA通过，[恢复与交付报告](reports/G07_full_racket_v4/placement075_completed40_data_queue01/DELIVERY_REPORT.md)保留全部原始记录、32诊断、权重和完整资产。中间16版原128为0完整/25空中/15恢复/0攻击，24版为0/22/12/2，均NF0；[全部实测报告](reports/G07_full_racket_v4/placement075_intermediate_actual_results_20261026/REPORT.md)已更新，GPU0继续32版。五个真实成功回合的CPU模仿在整回合留一5折均退化，已保存[代码与负结果](https://github.com/lzqw/unitree-smash/blob/db82fef/reports/G07_full_racket_v4/success_residual_bc_candidate_20261026/REPORT.md)并暂不进入原生试跑。原正式门槛仍未通过。
2026-10-26中间24版已完成：原全TRAIN128为0完整、21空中、9条件恢复（42.86%）、NF0；原GPU0队列已接上最后32版。另已准备[0.75条件试跑的完整执行流程](https://github.com/lzqw/unitree-smash/blob/2523bf5df1d3e08d3af532e1bc7d8f0060f17bef/reports/G07_full_racket_v4/centered_placement075_pilot8_queue01/REPORT.md)：真实源持有进程/旧检查点CPU边界检查通过，包括禁止候选本地GPU账本启动、拒绝旧严格续训冒充新075、解析首更新/8更新优化器布局；尚未集成main或原生执行。等待全部比较及实际持有进程退出后绑定赢家，再按原主guard运行新8、首个真实更新自动核验、完整轨迹审计、全128及六例视频。原物理与50/70/85、A/B各200门槛保持，完整目标仍未通过。

2026-10-26中间比较：16版全原TRAIN128实测0完整、25空中、13条件恢复（52%）、NF0；原队列已在GPU0接上24版，随后32版。另在隔离分支[准备了落点反馈距离尺度0.75的条件修订](https://github.com/lzqw/unitree-smash/blob/ab88fec1e0ff57e0349b12e88a4484bbc81b4168/reports/G07_full_racket_v4/centered_placement075_candidate_20261026/REPORT.md)：seed748901、2048环境×8更新、最大权重仍20；冻结40仅作兼容性样本，1890条实际输入的迁移平均动作逐位一致，actor归一化10485760保留、critic/Adam新建。新配置尚未集成main或原生训练；来源仍由实际全部中间结果选择，主196项持有源码不改、原物理与50/70/85及A/B各200门槛保持。

2026-10-26最新冻结结果：新增量学习器40在原全TRAIN128、零扰动、seed724201下为0完整、22空中、10条件恢复（45.45%）、NF0；较新增量8的26空中、13恢复下降，训练中的两次成功未稳定复现。[84.66秒实际六例失败视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_residual_base8_iter40_20261026.mp4)和[137MB两版完整原128例、权重、轨迹及对照包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered8_to40_frozen_TRAIN128_video_and_diagnostics_20261026.zip)均已上传并新下载SHA回验；4233帧全解码、14张事件预览及8张成品画面已检查。四次合格攻击都已越网，但首落x4.40至5.15米，仍超原4米目标上界。见[完整报告](reports/G07_full_racket_v4/centered40_fixed_video_review_20261026/REPORT.md)及[全128配对](reports/G07_full_racket_v4/centered8_to40_fixed_diagnostic_20261026/REPORT.json)。原父流程实际退出后，GPU0已顺序接上预登记16版评测，随后24/32；主源码继续保持，原A/B各200及50/70/85门槛保持，完整目标仍未通过。

2026-10-26实核运行：新增量学习器canonical8→40已成功完成，8388608控制环境步、2.748763 GPU小时；17156个闭合训练回合独立重算1945空中、916条件恢复、2完整、NF0，全部14条选定100Hz轨迹审计通过。[843MB完整原生训练、全部32个边界权重、两段成功实录及失败包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered40_completed_native_training_and_two_complete_videos_20261026.zip)已上传并新下载SHA回验，224成员CRC及223份内容SHA通过；[交付报告](reports/G07_full_racket_v4/centered40_training_delivery_20261026/REPORT.md)说明canonical来源、资源和复现依赖。这是变化中策略训练，不能当最终40冻结成绩；原全TRAIN128零扰动评测已在原GPU0 guard接上，A/B各200及原50/70/85门槛保持，完整目标仍未通过。

2026-10-26新增：[第二个完整训练回合＋身体触地失败对照，24秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_second_complete_and_failure_20261026.mp4)已上传并新下载SHA回验。实际TRAIN25、LANE153、EP5，800条100Hz轨迹；出球8.01017m/s、首次落点[3.929108,−0.430321]、原生稳定恢复3.62692秒。实际拍上接触点速度1.88147m/s，与出球速度分别报告；速度余量仍仅0.01017m/s。1200帧全解码，4张事件预览及7张成品画面已检查；[报告与独立核验](reports/G07_full_racket_v4/centered_second_complete_video_20261026/REPORT.md)保留原生记录和失败，不代表冻结表现。GPU0继续原训练，原A/B各200及50/70/85门槛保持，完整目标尚未通过。

2026-10-26训练新进展：[本轮新增量续训首个完整回合及独立核对](reports/G07_full_racket_v4/centered_resume_training_capture_20261026/FIRST_COMPLETE_NATIVE_EPISODE_CHECK.json)已保存真实800条100Hz全身/动作/公共输入：TRAIN111、LANE1519、EP3，实际收集横跨actor10至16。出球8.00235493m/s、向下2.27961m/s，合法空中拍线、实际越网、首次落点[3.994049,−0.695709]、同回合恢复4.10289秒、无违规。速度余量仅0.00235m/s、目标后沿余量5.95毫米；实际拍上接触点速度0.36269m/s，与出球速度分别报告。这是变化中策略训练的一次可行执行，最近冻结8仍为0/128完整，尚不证明稳定复现或正式达标。

[37秒真实完整训练回合＋两例失败视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_first_complete_and_failures_20261026.mp4)已上传并下载SHA回验：1854帧全解码，6张事件预览及9张实际解码画面已检查。保留真实非零回合编号，显示从地面准备、接触、首次落点及到8秒的恢复；另两例为恢复但漏球和身体触地失败。[161MB原始记录、100Hz动作、实际视频、源码与权重包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_resume_first_complete_training_video_20261026.zip)为161167119字节，SHA256 `cb7bf568d12593059ce7800ddee30c6e7f93ab89b60de7da0290fd28ca8a1457`，71成员CRC及70源文件SHA通过，含当时全部4516闭合回合及canonical源8/actor10至16边界诊断。源训练作业尚未结束，不能把诊断权重当canonical重启或冻结成绩；运行时资产复用下方215MB已验证训练包。见[下载回执](reports/G07_full_racket_v4/centered_resume_training_capture_20261026/ARCHIVE_UPLOAD_RECEIPT.json)。

2026-10-26最新固定评测：[新增量学习器8完整原TRAIN128](reports/G07_full_racket_v4/centered_residual_pilot8_queue02/FIXED_TRAIN128_RESULT.json)为0完整、26空中、13条件恢复（50%）、3合格攻击、6目标命中、NF0。原固定基线8为20空中、12恢复（60%）、1攻击、8目标；[全部128例配对、三次合格攻击及实际运动对照](reports/G07_full_racket_v4/centered_probe8_fixed_diagnostic_20261026/REPORT.json)均保留。三次合格攻击TRAIN6/7/23已经实际越网且同回合恢复，但首次落点x4.52至4.74米，超原4米目标上界，仍全部失败。源初态一致、2.5秒来球位置一致，2.7秒拍线中心差异中位数3.0厘米；这是运动描述，不是单个方法组件的因果证明。

[最新85秒新冻结8实际效果视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_residual_base8_probe8_retry02_20261026.mp4)及[112MB完整冻结128、配对诊断与视频包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_probe8_frozen_TRAIN128_video_and_diagnostics_20261026.zip)已上传并下载SHA回验。视频4276帧、85.52秒、50fps，保留训练前登记19/23/107/0/8/58全部六例；16张事件预览和8张实际解码画面已检查，侧视/俯视显示完整机械臂和球拍、实际接触点与首次落点。包为111769964字节、SHA256 `b08ace400ad9044da80839273b00174ce49d1bdce04d363e2d4d6299364a0533`，110成员CRC及109源文件SHA通过，运行时资产复用下方215MB训练包。[上传回执](reports/G07_full_racket_v4/centered_probe8_fixed_video_review_20261026/ARCHIVE_UPLOAD_RECEIPT.json)。

本轮已按[预登记条件](reports/G07_full_racket_v4/centered_residual_resume32_queue01/PARENT_VERIFIED_DECISION.json)从真实canonical8严格完成32更新至40。[末次独立回执](reports/G07_full_racket_v4/centered_residual_resume32_queue01/TRAINING_RECEIPT.json)核对17项Adam各推进640步、两套归一化增加8388608步、所有2048条源episode ID链精确；本作业17156个闭合回合、2完整、NF0。此前[实际update15与首批地面回合检查](reports/G07_full_racket_v4/centered_residual_resume32_queue01/ROOT_ACTUAL_UPDATE15_FIRST_COHORT_CHECK.json)继续保留；全部14条真实轨迹审计通过。训练成功不是冻结成绩，原完整成功50%、空中70%、条件恢复85%、A/B各200及全部物理/规则保持，完整目标仍未通过。

[第32次实际权重与公共导出检查](reports/G07_full_racket_v4/centered_intermediate_selection_queue01/ACTUAL_UPDATE32_HEALTH_AND_EXPORT_CHECK.json)已完成：17项Adam相对源8推进480步，两套归一化增加6291456步，累计新增量学习器8388608步；1890条真实公共输入的导出误差0。已登记16/24/32公共策略均准备完成，之后由现有队列在原全TRAIN128上实测；目前没有这些中间版本的新冻结成绩。非canonical诊断仍仅用于推理，未作为物理/优化器重启点。

[第24次实际权重与公共导出检查](reports/G07_full_racket_v4/centered_intermediate_selection_queue01/ACTUAL_UPDATE24_HEALTH_AND_EXPORT_CHECK.json)已完成：17项Adam相对canonical源8均推进320步，两套归一化增加4194304步，累计新增量学习器6291456步；1890条真实公共输入上的导出误差0。公共24已准备供现有队列之后评测，非canonical `model_23.pt`没有被当作重启文件。当前没有该24版冻结成绩，也没有启动第二个原生GPU作业。

独立分支[已测增量策略迁移候选接口](https://github.com/lzqw/unitree-smash/blob/a676d58417a960670494d7c2560e211d29bd5d91/reports/G07_full_racket_v4/centered_policy_transfer_candidate_20261026/REPORT.md)已准备并上传，a676d58，50项CPU检查通过：在1890条真实公共输入上，迁移后的外层平均动作与来源逐位一致，actor归一化保留，新critic/探索/Adam/计数独立。它支持已经实际测过的中间公共策略，不把非canonical诊断当重启点，也拒绝借用其他策略的评估回执。当前GPU0仍执行原作业；该候选未集成、未原生运行。待当前训练和中间比较持有进程退出，再按实际结果决定是否用于新的TRAIN奖励修订；CPU兼容性探针0.75不是已选配置或性能成绩。

[中间冻结策略比较队列](reports/G07_full_racket_v4/centered_intermediate_selection_queue01/PREDECLARED_PLAN.json)已登记并实际启动，PID1697096/session52011，当前仅等待父流程1678986实际退出；GPU0仍只有原1687066作业。若40版已经达到原50/70/85门槛则跳过额外开发评估，否则依次在原全TRAIN128、零扰动、seed724201下比较实际16/24/32边界策略，并与已测8/待测40按完整、攻击、条件恢复、空中、同分较早更新排序。选择时点发生在看到本轮罕见训练成功之后、但在这些新冻结结果之前，不声称盲测。1890条实际公共输入上的[源8重建与16导出检查](reports/G07_full_racket_v4/centered_intermediate_selection_queue01/CPU_PREPARATION_CHECK.json)误差均0；这是CPU导出一致性，不是新成绩。中间诊断权重仅作推理导出，不冒充canonical重启；196项源码与输入须保持到该队列退出，全部评估继续使用原主GPU0 guard/ledger。

[新落点/网余量诊断](reports/G07_full_racket_v4/centered_attack_target_geometry_20261026/REPORT.json)保留全部128例事件链；三次真实合格攻击都已越网和恢复，但深出目标0.52–0.74米。使用同一已声明气动参数的CPU自由飞行，原首落误差0.35–2.06毫米；同速瞄准x3.9米的假设需要出球再向下7.7–9.6度，其中TRAIN6/7保守网余量约2.6/4.6厘米、TRAIN23约−2.2厘米。不是机器人已执行或接触/恢复可行性证明。原窄落点奖励核在三次实际落点上计算仅约0.003–0.252，提供后续方法诊断，不修改当前续训契约或原目标区。

此前新02原生试跑实际完成2097152步、0.670711 GPU小时，独立重算3703回合311空中、112条件恢复、0完整、NF0；canonical8和公共策略已保存，[全部12条完整100Hz轨迹](reports/G07_full_racket_v4/centered_residual_pilot8_queue02/TRAINING_TRACE_AUDIT.json)通过实际原始回合/接触/公共输入SHA/姿态审计。

[新方法22秒首批训练实录](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_initial_exploration_20261026.mp4)已上传并下载SHA回验，1109帧全解码，9张实际解码画面及3张事件预览已检查。TRAIN107起跳后稳定恢复3.96秒但漏球，TRAIN8身体触地终止；两例均失败，来自初始探索、不是最终新8冻结策略。两条lane事先登记，这次诊断选例发生在看过轨迹之后。[完整训练/权重/原始记录/视频/程序和运行依赖包（215MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_residual_probe8_native_training_and_initial_video_20261026_v2.zip)为214767506字节、SHA256 `d672edf51ec90649b1b0381fe9ff861e0c2433c98de944e0a07937e52954c4e4`；165成员CRC、163源文件SHA及新下载整包SHA已核对。包内包括原Go2步行策略、标准PiPER URDF和10层USD依赖，不包含随后完成的冻结评测，冻结数据见上方112MB附包；[交付回执](reports/G07_full_racket_v4/centered_training_capture_video_20261026/ARCHIVE_UPLOAD_RECEIPT.json)可查。完整G07目标仍未通过。

2026-10-26实核运行：新02试跑的[2048个实际地面状态](reports/G07_full_racket_v4/centered_residual_pilot8_queue02/ACTUAL_GROUND_INITIALIZATION_CHECK.json)及[首个原生更新](reports/G07_full_racket_v4/centered_residual_pilot8_queue02/ACTUAL_UPDATE_0001_HEALTH_CHECK.json)已核对：262144步、模型/Adam有限、actor仍在两次价值网络预热阶段、NF0，完整轨迹记录器已核对同数公共输入，尚无闭合回合。另已在新冻结结果产生前登记并启动[条件续训队列](reports/G07_full_racket_v4/centered_residual_resume32_queue01/PREDECLARED_PLAN.json)：等待父流程实际退出并独立复算全TRAIN128；若原指标达标则保留冻结策略进入正式面板评估，否则要求至少16次空中、8次条件恢复及完整轨迹审计通过，才同契约从canonical8继续32更新。原18关节网络、Adam、归一化、RNG和计数保持，不继承旧运动状态；未宣称改善或正式通过。

2026-10-26更新：新固定8基线增量学习的首次原生试跑在第1次PPO更新后，因包装控制器缺少日志清空接口而退出；实际262144步、actor仍预热、NF0，没有完成回合或新视频。[失败记录及费用审查](reports/G07_full_racket_v4/centered_residual_pilot8_queue02/REVIEWED_PREVIOUS_FAILURE.json)保留原FAILED、0.095479 GPU小时与账本保守计步。修复增加日志setter及未来完整更新后的异常检查点保护，主工作区85项CPU检查通过；随后用不同02目录、相同配置和seed从全新增量学习器重新运行，不从非canonical诊断权重续训。先看下方已审核的原8版86秒视频及退化40版80秒视频；它们均非新增量学习器的成绩。完整G07目标仍未通过。

2026-10-26下一轮：[固定已测基线上的增量学习计划](reports/G07_full_racket_v4/centered_residual_pilot8_queue01/PREDECLARED_PLAN_V2.json)已完成准备并集成。第二组gain0.70实测为22空中、13条件恢复、0合格攻击、0完整、NF0；按结果产生前登记的完整/攻击/恢复/空中顺序，选择仍有1次合格攻击的原8版作为固定基线。[源码退出和基线选择证据](reports/G07_full_racket_v4/centered_residual_pilot8_queue01/BASELINE_DECISION_AND_SOURCE_EXIT.json)确认旧原生/视频持有进程已结束，才修改主源码。新18关节网络从零均值增量、全新价值网络/归一化/Adam/计数起步，腿增量先限±0.03rad、臂±0.15rad，旧公共策略固定，原硬件、球拍、球场、规则和验收保持。主工作区83项检查通过，3200条实际公共输入（含一次真实重置）上零增量逐步复现原指令；这仍是CPU一致性证据，不继承原策略成绩。接下来仅通过原GPU0 guard/ledger执行2048环境×8更新，并记录完整训练轨迹，再独立冻结TRAIN128和生成真实视频；本段发布时新方法尚未完成原生训练，完整目标未通过。

[实时拍面修正第一组gain0.35原生结果](reports/G07_full_racket_v4/live_tangent_trial_launch_v2_20261026/gain035_RESULT.json)已完成并独立重算全部原TRAIN128：71次球拍接触、21次合法空中拍线、11次条件恢复、0次合格攻击出球、0完整、NF0；19次越网、16次合法场内、8次目标命中仍不能组成完整扣杀。真实消耗0.311335 GPU小时、102400控制环境步，全部128个公共观测/动作流保留，原未执行提案不改写；另存的新执行权重才取得`native_execution=true`。第二组gain0.70已经由同一主GPU0 guard顺序接上，尚无完整结果。后续训练还需同时处理攻击出球和全身动作漂移。

[冻结40失败复测、视频与诊断完整包（108MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint40_frozen_TRAIN128_failed_videos_and_diagnostics_20261026.zip)已上传并下载回验：107664472字节，SHA256 `c06ea95fc519d8cdeff0f2332c110c8cae322b19ff1b9f750ce3ef59c5c8281f`，81成员全部CRC/SHA验证。包含8/40两版全部128例记录与实际全身轨迹、canonical40与导出、两段真实失败视频、完整对照表及导出/运动诊断程序；完整球拍USD等运行时依赖复用下方已验证483MB训练包。此包明确记录40版失败，不是正式A/B或完整目标通过证据，见[下载回执](reports/G07_full_racket_v4/joint40_fixed_delivery_20261026/UPLOAD_RECEIPT.json)。

2026-10-26最新结果：[冻结40实际失败视频，80秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_joint_placement_iter40_20261025.mp4)和[TRAIN65/6重新执行，约30秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_joint_placement_iter40_TRAIN65_6_20261026.mp4)已审核并下载SHA回验，全部5483帧解码，28张实际预览/解码画面检查。固定原TRAIN128为0触球、0空中、0条件恢复、0完整、NF0，明显退化；训练中的两次完整成功未能稳定复现。所有128例及[8→40完整对照](reports/G07_full_racket_v4/joint40_fixed_diagnostic_20261026/REPORT.json)保留，不能继续把40版当作更优策略。[导出核对](reports/G07_full_racket_v4/joint40_export_diagnostic_20261026/REPORT.json)中1600条实际输入网络误差0、800步完整控制器与原生指令最大误差1.32e-5；[运动对照](reports/G07_full_racket_v4/joint40_export_diagnostic_20261026/ACTUAL_MOTION_COMPARISON.json)确认初态、原解析控制器与击球前来球一致，但2.7秒拍线中心差异中位数37.5厘米、机身姿态差异28.8度。此为实测耦合运动差异，不是单一原因的因果证明。两种基于有用8版的实时修正继续顺序实测；原入口`queue.py`遮蔽标准库导致的启动失败已[单独保留和修复](reports/G07_full_racket_v4/live_tangent_trial_launch_v2_20261026/STARTUP_FAILURE_REVIEW.json)，新目录第一组已进入真实PhysX。后续方案还需约束全身动作漂移；候选v3奖励代码仍未进行原生训练，完整原目标仍未通过。

[新学习器40原始训练与接触证据包（483MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint40_training_raw_and_contact_evidence_20261026.zip)已上传并独立下载回验：483123295字节，SHA256 `a5b3c8bc661e80efb2ba649a96a095e94a4fb81698a03b7823cd5de9860fe19f`，112成员全部CRC/SHA一致。包含18591个原始训练回合、实际接触状态、源canonical8与末canonical40、27至35更新的诊断权重、10层真实USD依赖、CSV和源码。变化中策略两次完整成功不归因于某个冻结权重；原训练完整动作未记录。打包时仍在运行的冻结40评测与新视频均未混入，见[下载回执](reports/G07_full_racket_v4/joint40_training_delivery_20261026/UPLOAD_RECEIPT.json)。

2026-10-26进展：[新学习器40训练与评测恢复报告](reports/G07_full_racket_v4/joint_placement_iter40_recovery_20261026/PROGRESS_REPORT.json)。续训实际完成8388608控制环境步、2.753516 GPU小时；18591个闭合训练回合独立重算为1659空中、690条件恢复、2完整、NF0。[两次完整训练回合的原生拍线接触记录已逐一匹配](reports/G07_full_racket_v4/joint_placement_first_training_complete_20261026/FINAL_CONTACT_CONFIRMATION.json)，但没有保存原训练全程动作，不能伪造成功视频，也不能把变化中策略成绩算成冻结表现。原后处理错误要求续训回合编号从0开始，实际canonical8保留1至3；新校验改为逐lane严格核对源检查点至末检查点的全部编号连续性，其余检查保持。历史失败/跳过状态未改写、未重跑训练；原seed724201全128冻结40评测已启动，随后CPU导出原六例与TRAIN65/6复测视频。两种原登记实时拍面修正也已恢复条件队列。另一个[奖励/轨迹记录候选分支](https://github.com/lzqw/unitree-smash/tree/g07-live-string-attack-reward-v4-20261025)有73项CPU检查通过，尚未进行v3原生训练；主源码保持至现有全部持有进程退出。完整G07目标仍未通过。

2026-10-25 后续准备：[真实轨迹上的预测误差](reports/G07_full_racket_v4/joint8_prediction_error_20261025/REPORT.json)包含两例35条精确100Hz公共观测，以及71个接触回合的610条50Hz向后差分估算，两类不混称。精确子集中，提前80至120毫秒的局部拍面残差中位数约21.9厘米，40毫秒内约1.3厘米；局部坐标随球拍转动，不能把它当作孤立球预测误差或新策略成绩。原200毫秒两组原生修正试验继续保留，等待实测。

[第40次冻结评测视频队列](reports/G07_full_racket_v4/joint_placement_iter40_video_queue01/PREDECLARED_PLAN.json)已启动：在当前续训期间、固定评测产生前登记沿用19/23/107/0/8/58六例；待实际全128评测完成，再以Mesa/llvmpipe CPU渲染侧视/俯视并逐帧解码，最后留待真实画面审核。现在还没有第40版视频。

[独立的新奖励候选分支](https://github.com/lzqw/unitree-smash/tree/g07-live-string-attack-reward-v4-20261025)已上传：66项CPU检查通过，实际canonical8的13个公共actor/归一化/探索张量迁移在200条真实观测上与原导出输出误差0，价值网络和Adam保持新建。新模式用当前实测拍线中心势函数替代旧上缘位置引导、提供有界实际出球进度反馈，并令20分目标奖励同时要求原攻击门槛；回放显示原4039训练记录的21个软目标、固定128的8个软目标均不再满足该奖励门槛。这是终端公式回放，不是历史支付重建或改善证据。该分支尚未合入正在运行的源码、未进行v3原生训练；下一步需等待现有源码持有进程结束并审查真实结果，仍只使用主工作区原GPU0 guard/ledger。

2026-10-25 续训实核：[新学习器第21次更新](reports/G07_full_racket_v4/joint_resume_health_20261025/REPORT.json)已完成本作业3407872控制环境步，累计新学习器5505024步；独立重算7001个闭合训练回合为606空中、231条件恢复、0完整、NF0。17项Adam状态均比canonical8增加260步，两套归一化计数各增加3407872，actor权重确实更新；这些是变化中策略的训练数据。GPU0继续原32次更新任务与随后第40次固定TRAIN128评测。

[71次实际接触诊断](reports/G07_full_racket_v4/joint8_contact_centering_20261025/REPORT.json)及[接触分布图](reports/G07_full_racket_v4/joint8_contact_centering_20261025/contact_distribution.png)显示20次拍线、51次拍框集中在上缘；当前教师瞄准点为拍头中心上方0.10米，运动学拍头位置误差最大0.0163毫米。历史0.07米静态重定向原生搜索仍0完整，已保留失败结果。新的[实时来球/拍头切向修正试验](reports/G07_full_racket_v4/live_tangent_trial_20261025/PREPARATION.json)在1600条实际观测和800步完整控制器回放中通过有限性、导出一致性、腿部指令一致性和原5rad/s目标限速检查；两种强度均保持`native_execution=false`。观察队列已启动，等待当前原生训练及冻结评测真正结束，若尚未达到原正式门槛，再逐个执行全部原TRAIN128；尚无新试验的PhysX成绩。

[联合奖励学习器8完整证据包及下一轮程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint_reward_probe8_native_evidence_and_next_trial_20261025.zip)已上传并下载回验：190699651字节，SHA256 `3f90e2796dac59fd00d45a0d0f3acef0be9f36401b5a063d275296840347955c`，523成员逐一CRC/SHA核对。包含新8训练/冻结128原始记录、权重、86秒视频、完整球拍依赖、接触诊断；变化中策略的第21次快照与尚未原生执行的两个新提案单独标明，不算冻结性能或正式通过。[下载回执](reports/G07_full_racket_v4/joint8_delivery_20261025/UPLOAD_RECEIPT.json)。

[最新实际视频：联合落点奖励新学习器8（86秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_joint_placement_probe8_20261025.mp4)已上传并下载SHA回验，4308帧全解码，侧视/俯视保留训练前选定的TRAIN19/23/107/0/8/58全部六例。TRAIN8实际空中拍线出球8.006米/秒、向下2.095米/秒并合格恢复4.069秒，但首落x4.218米，超原目标约22厘米，仍判失败。完整固定TRAIN128：0/128完整、20/128空中、12/20条件恢复、NF0；8个目标命中全部低于8米/秒，不能据目标数宣称扣杀改善。GPU0已按预登记条件从新canonical8严格续训32更新至新学习器40，完整原目标仍未通过。

[此前原PPO第96次权重视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_iter96_20261025.mp4)已上传并下载验SHA，约69秒，包含侧视/俯视和全部预选失败；[TRAIN58单例复测](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_iter96_TRAIN58_20261025.mp4)约17秒。完整固定TRAIN128结果为0/128完整、24/128空中、8/24条件恢复、NF0，未达原门槛，条件队列已跳过原A/B，未产生新正式成绩。[32与96版完整TRAIN对照](reports/G07_full_racket_v4/residual_iter96_fixed_diagnostic_20261025/REPORT.json)保留全部128例：两版完整均0；32版恢复14/22、攻击3次，96版恢复8/24、攻击1次，空中分别22/24次。下一轮选择32版actor初始化新的联合落点奖励实验，记录该TRAIN取舍，不作统计或因果优势主张。完整资料包已构建并通过287成员CRC/SHA，1171962010字节、SHA703dd32706e9f82498623147b1f0b4bbe9ff6a51a87e1d20558fa92e53c9382c，[完整资料包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_PPO_iter96_fixed_TRAIN128_training_videos_and_raw_20261025.zip)已上传，下载字节数及SHA256回验一致，见[交付回执](reports/G07_full_racket_v4/residual_iter96_delivery_20261025/UPLOAD_RECEIPT.json)。完整G07目标仍未通过，继续训练改进。

已从候选分支集成可选联合落点奖励与显式 --initialize-actor 入口，供新的真实训练实验使用。它在合法空中攻击、实际越网后的首次落点提供平滑反馈（weight20、distance scale0.25米），低速目标命中不获得该项奖励；原物理、完整球拍、执行器限制、规则、全部来球及验收门槛保持。新奖励使用新建价值网络、其归一化、Adam、RNG和学习计数，只从已完成canonical32复制公共actor、actor归一化与探索尺度，不恢复旧物理状态。旧严格续训和扩批仍拒绝跨奖励恢复。

[新的联合落点奖励学习器已完成8次更新](reports/G07_full_racket_v4/joint_placement_probe8_queue01/TRAINING_RECEIPT.json)：实际2097152控制环境步、0.661211 GPU小时，独立重算全部4039个闭合训练回合为303空中、82条件恢复、0完整、NF0。新canonical权重与公共actor均已保存；这是新的学习器计数，从原canonical32只迁移公共actor，训练回合经历参数/归一化变化和探索，仍需冻结评估。

[完整首批2048路地面回合](reports/G07_full_racket_v4/joint_placement_first_ground_cohort_20261025/REPORT.json)已独立重算并保留全部原始失败：209空中、82条件恢复、12攻击、18目标、0完整，攻击与目标无同回合重叠。另有1510条后续回合在单独标识的3558行快照中。209次空中触球均发生于2.659至2.856秒，早于3.84秒时首次actor参数更新，[时间范围说明](reports/G07_full_racket_v4/joint_placement_first_ground_cohort_20261025/TEMPORAL_SCOPE.json)因此不把首批第一拍表现归因为新奖励学习改善。

[新公共actor导出核验](reports/G07_full_racket_v4/joint_placement_export_check_20261025/REPORT.json)通过：200条真实保存的794维公开观测，原生权重与导出actor的CPU输出误差0，12个actor/归一化张量逐项一致，拆批误差2.98e-8；不将CPU保真检查当作物理成绩。[完整新固定TRAIN128结果](reports/G07_full_racket_v4/joint_placement_probe8_queue01/FIXED_TRAIN128_RESULT.json)已核验：0完整、20空中、12条件恢复、1攻击、8目标、NF0。[逐回合CSV](reports/G07_full_racket_v4/joint_placement_probe8_queue01/joint_iter8_fixed_episodes.csv)保留全部128例；[攻击与落点检查](reports/G07_full_racket_v4/joint_placement_probe8_queue01/ROOT_FIXED_ATTACK_TARGET_CHECK.json)确认8个目标全部低于原攻击速度，唯一攻击TRAIN8首落超界0.218米。预选六案例真实视频现已审核交付，见顶部链接。

合并后的奖励、actor迁移和兼容契约共42项CPU检查通过，原[候选检查](reports/G07_full_racket_v4/joint_placement_candidate_20261025/ACTOR_TRANSFER_READINESS.json)、[真实iter32迁移检查](reports/G07_full_racket_v4/actor_transfer_native_source_check_20261025/REPORT.json)和[历史依赖恢复记录](reports/G07_full_racket_v4/joint_placement_prerequisites_20261025/READINESS.json)保留。

[固定评估后的同奖励续训队列](reports/G07_full_racket_v4/joint_placement_resume32_queue01/PREDECLARED_PLAN.json)已启动等待。若新固定TRAIN128达到原50%完整、70%空中、85%条件恢复且NF0的联合门槛，则保留冻结策略进入正式准备；其余情况下，仅在内部防崩检查（空中至少10、条件恢复至少3、NF0）通过后，按同一配置在2048路再学32更新至新学习器iter40，新增8388608控制环境步，然后完整固定复测128例。critic/Adam/归一化/RNG与计数严格续用，回到真实地面边界；正式验收仍要求原A/B各200。后继GPU作业已实际启动；[原生续训核验](reports/G07_full_racket_v4/joint_placement_resume32_queue01/ROOT_NATIVE_RESUME_CHECK.json)在global11确认2048真实地面状态、累计2883584步、全部17项Adam状态及两套归一化计数延续，模型/Adam有限；不预先宣称新策略改善。

[新学习器第5次健康检查](reports/G07_full_racket_v4/joint_placement_probe8_queue01/ROOT_UPDATE05_HEALTH.json)：实际1310720控制环境步，模型/Adam有限。[本次4096条定步长动作抽样](reports/G07_full_racket_v4/joint_placement_probe8_queue01/UPDATE05_TELEMETRY_SCOPE.json)未检出截断，不能据此断言全部动作均未截断。运行时累计2313条闭合回合中，1585条属于初始2048路，另728条来自后续回合；初始仍有463路未闭合。该健康检查没有另行独立重算全部回合，不以当前133空中/0恢复/0完整的截尾运行时计数作冻结策略成绩。完整原始数据与独立评分由训练结束后的队列核验。

[新8次训练后的奖励信号回放](reports/G07_full_racket_v4/joint_placement_probe8_signal_replay_20261025/REPORT.json)重算全部4039条原始记录：17次攻击、21次目标仍零重叠。终态字段回放产生16个正反馈、其中5个超过1；没有逐控制步奖励日志，不能将该回放称为实际发放金额。完整任务仍需原A/B各200；[既有G07注册器检查](reports/G07_full_racket_v4/joint_placement_probe8_queue01/G07_REGISTRY_CHECK.json)也未通过，其旧证据入口缺失，不把CPU保真或TRAIN进展当作正式验收。

[本轮事件短板诊断](reports/G07_full_racket_v4/residual_n2048_event_diagnostic_20261025/REPORT.json)已重算第56次更新的原14048条训练记录：129次空中攻击仅5次同时命中目标，127次目标中122次出球速度低于8米/秒；其中2次攻击与目标均满足的回合因实际机身着地、恢复0而失败。前三批各2048回合均已闭合，空中197/192/200；后续尚未闭合的批次单独标明，不能将截尾统计当作冻结策略退化。报告含每TRAIN案例计数和所选原始回合，可用最终原始数据的相同前缀SHA复算；该诊断保留原PPO第56次快照的创建时范围；新联合奖励效果仍待本轮固定评估。

第96次更新后的正式条件队列已按原计划实际核对固定TRAIN128，因未达原50%完整、70%空中、85%条件恢复且NF0的联合要求，状态为SKIPPED_TRAIN_BELOW_ORIGINAL_THRESHOLDS；未读取或执行新正式面板。其预登记和CPU检查保留，后续策略仍需在原门槛下重新验证。

[第96次更新的最终训练证据](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/FINAL_TRAINING_REVIEW.json)已保存：64次续训实际新增16777216个控制环境步，训练守护正常退出；39077个闭合回合已独立重算为3332空中/1244条件恢复/8完整/NF0，仍是变动策略训练数据。全部8例现已匹配[真实原生拍线接触](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/FINAL_CONTACT_CONFIRMATION.json)、28体姿态和刚体点速度，独立重算最大误差0.00002024米/秒，低于原0.0001容差；这是post-PhysX状态一致性，不是独立测得的入射拍速。最终canonical96与导出权重已核对；固定TRAIN128已完成但未达原门槛，两个新版视频已审核上传，原A/B由条件队列跳过。旧快照的pending字段保持创建历史，以最终确认为准。TRAIN61的负net_margin诊断值及限制仍在[原案例说明](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/CASE_NOTES_UPDATE85.json)，不声称有正的最小净空，也没有完整训练动作录像可补作。

[续训中点独立核验](reports/G07_full_racket_v4/residual_n2048_midpoint_20261025/REPORT.json)：全局update64、本轮新增32/64更新，8388608新增控制环境步；全部19164训练回合为1516空中/599条件恢复/4完整/NF0，模型与Adam有限。首五批各2048均已闭合，后续未闭合批次单独标记。原GPU0继续至iter96，原门槛、模型和规则保持；这些计数仍非冻结策略或正式A/B成绩。

[新增2048环境完整训练回合](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/REPORT.json)：TRAIN58_LANE1338_EP2从地面执行，在1.753米处合法空中拍线击球，出球8.003米/秒、向下2.818米/秒，实际越网后首次落在[3.794,.139]米，合格恢复2.94秒，无失效违规。原7766回合记录及actor39..45权重谱系已保存；该回合经历权重/归一化变化，不是一个冻结策略成绩。GPU0继续原训练，已在固定评估前另行登记TRAIN58真实轨迹复测视频；原训练没有保存完整运动轨迹，不能补造成原回合视频。

[最新69.38秒实测视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_iter32_20261025_v3.mp4)已上传并下载SHA回验：同一iter32固定策略，保留评估前预选的TRAIN19/23/107/0四个失败回合，另加明确标为评估后选择的TRAIN8。TRAIN8在己方1.748米处真实空中拍线攻击，出球8.04米/秒、向下1.55米/秒，实际越网并合格恢复4.329秒；首次落点x4.537米超过原目标上界0.537米，完整仍失败。侧视/俯视、慢放、真实接触与首次落点，全部3469帧解码；v3补全远落点并保持恢复近景，前两版及拒绝原因保留。

[iter32完整训练、固定128回合原始轨迹、源码、权重与视频包（123.9 MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_PPO_iter32_fixed_TRAIN128_video_training_and_raw_20261025.zip)已上传回验：151成员全部CRC/SHA通过，ZIP SHA256为`db25f79073218916780c766bc783ba613ee0e73d8f62c8d8a955f8a3da338c45`。其中的2048信息仅为另行标注的启动检查摘要，不含其进行中的原始训练数据或策略成绩。

[固定iter32诊断报告](reports/G07_full_racket_v4/residual_iter32_fixed_diagnostic_20261025/REPORT.json)与[128行固定评估CSV](reports/G07_full_racket_v4/residual_csv_queue01/iter32_fixed_episodes.csv)可逐回合核对；训练中3171回合的417空中/210恢复/1完整不作为固定actor32成绩。

当前进度（2026-10-25）：新联合落点奖励学习器8次更新与完整固定TRAIN128均已完成，固定完整0/128、空中20/128、条件恢复12/20、NF0；唯一攻击TRAIN8首落超目标约22厘米，完整仍失败。86秒六案例真实视频已审核上传并下载验SHA。GPU0现从新canonical8严格续训32更新至新学习器40，原物理、全部来球、规则和正式50%/70%/85%门槛保持，继续验证完整目标。

[联合落点奖励候选分支](https://github.com/lzqw/unitree-smash/tree/g07-joint-placement-reward-v4-20261025)已通过42项CPU功能检查和真实iter32 actor迁移检查。[隔离运行依赖恢复](https://github.com/lzqw/unitree-smash/blob/119b461010bee619905ad8736eb982b1fcf55160/reports/G07_full_racket_v4/joint_placement_prerequisites_20261025/READINESS.json)现已完成：14个历史文件按原SHA复制，另补固定步态权重；实际候选200路CPU控制器加载、200×794公开TRAIN观测推理及10层USD文件SHA检查通过。它提供实际空中攻击、越网后的首次落点反馈，原门槛不变；仍未原生训练、未证明策略改进。该段保留集成前候选检查的来源；原96训练、固定评估、视频和条件正式队列现均已结束，新实验已在[运行前计划](reports/G07_full_racket_v4/joint_placement_probe8_queue01/PREDECLARED_PLAN.json)登记，2048真实地面环境、8次更新、新学习器由canonical32的公共actor初始化。

[2048环境首批地面训练回合](reports/G07_full_racket_v4/residual_n2048_first_ground_cohort_20261025/REPORT.json)现已全部闭合并独立重算：197空中、97条件恢复、16空中攻击、26目标、0完整、数值故障0；攻击和目标仍无同回合重叠。它们经历了探索与权重更新，不是固定策略成绩，也不是2048种来球。原训练继续，下一版五案例视频已在第96次固定评估开始前登记。

[训练完成报告](reports/G07_full_racket_v4/residual_iter32_training_completion_20261025/REPORT.json)与[3171行训练CSV](reports/G07_full_racket_v4/residual_iter32_training_completion_20261025/training_episodes.csv)保留完整训练计数、更新日志、实际资源用量和来源；[原生接触确认](reports/G07_full_racket_v4/residual_n512_first_complete_20261025/FINAL_CONTACT_CONFIRMATION.json)已与训练结束后的contacts.json匹配。固定评估的128个地面初态与新权重已核验，最新成绩与视频见上方；旧iter8视频保留为历史对照。

[完整训练回合与全部同批原始记录](reports/G07_full_racket_v4/residual_n512_first_complete_20261025/REPORT.json)保留1894回合快照、原生索引、同lane前一失败及权重谱系。PPO训练没有保存这条回合的完整运动轨迹，不能将事件/终态记录补成真实动作视频。第32次更新后的固定TRAIN128评估与实际视频现已完成；TRAIN19/23/107/0四例在该次固定评估开始前已选定，追加TRAIN8的选择时间与依据已单独说明。此前首批512地面训练回合为70合法空中、33条件恢复、0完整；原PPO iter8固定TRAIN128为18/128空中、9/18条件恢复、完整0。

已登记[有条件后续训练](reports/G07_full_racket_v4/residual_resize2048_queue01/PREDECLARED_PLAN.json)：第32次更新的固定评估成功核验且未明显崩溃后，将在2048个真实地面环境（128来球各16副本）再学64次更新至iter96，再次固定评估TRAIN128。本次后继作业现已实际启动，2048原生地面初态、学习器继承及首个更新已核验；冻结策略效果仍待后续复测。原模型、执行器、规则和正式验收条件保持不变。

[此前PPO第8次更新视频（41.36秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_iter8_20261025_v2.mp4)：同一iter8固定策略、预选TRAIN23/107/0。第一例8.42米/秒空中攻击并稳定恢复4.27秒，但首落x=4.793米超出目标；另两例为漏球和拍框失败。侧视/俯视包含完整机器人、机械臂与球拍；2068帧解码和GitHub附件下载SHA回验通过。相同编号可与此前基线视频比较，该历史视频不代表iter32或之后的权重。

[固定TRAIN128全部原始轨迹、源码、权重与视频（69.7 MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_PPO_iter8_fixed_TRAIN128_video_and_raw_20261025.zip)已上传回验，75成员均核验SHA256。

逐回合表另附：[固定策略128行CSV](reports/G07_full_racket_v4/residual_csv_export_20261025/iter8_fixed_episodes.csv)与[变动权重训练215行CSV](reports/G07_full_racket_v4/residual_csv_export_20261025/iter8_training_episodes.csv)。它们分别重算为空中18/恢复9/完整0和22/13/0，不混作一种策略的成功率；原始JSON与物理轨迹保持不变。

[原生PPO训练包：检查点、策略与原始记录](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_native_residual_PPO_iter8_checkpoint_and_records_20261025.zip)已上传并下载校验（19.57 MB、38成员）。它记录8次更新的真实训练，不代表固定策略或正式A/B验收通过。

此前进度（2026-10-25）：航向修正CEM34已完成，最佳候选仅1/32合法空中拍线，0攻击/目标/完整，数值故障0；两个相同参考均0/32，没有带来任务改善。拍线内移随后完成，结果见上方当前进度。

此前进度（2026-10-25）：高度规划CEM33已完成，最佳候选合法空中接触5/32、条件恢复2/5、攻击2、目标0、完整0、数值0；另一同参数执行为空中3/32，未显示分布改善。航向训练随后完成，失败结果见上方当前进度。

此前进度（2026-10-25）：横向修正candidate12的TRAIN128复测完成，空中接触19/128、接触后恢复12/19、攻击7、完整0、数值故障0，原始回合重算一致。该轮高度规划训练随后完成，结果见上方当前进度。

[此前横向修正策略视频（42.66秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_lateral_train128_20261025.mp4)展示同一candidate12在TRAIN128中的攻击恢复但落点偏远、目标命中但速度不足，以及拍框失败；全部2133帧解码通过，GitHub下载SHA256一致。三个独立回合不代表正式A/B通过。

此前进度（2026-10-25）：CEM31 candidate5完成固定A/B各200回合，完整成功0/400；A空中接触12.0%、接触后恢复41.7%，B为3.0%和16.7%，数值故障0。

[此前正式评测视频（31秒）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_formal_evaluation_cem31_20261025.mp4)展示同策略的攻击越网恢复、目标落点失败及强扰动触网失败；全部1550帧解码通过。CEM32的TRAIN128结果见上方当前进度。

此前视频：[41秒真实空中攻击回球与同回合恢复](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261024_1400.mp4)（41.44秒、1280×720、50fps）。CEM20候选50从地面自主起跳，在己方1.763米处实际拍线击球，出球8.10米/秒、向下1.27米/秒，越网后在对方场内首落，并完成3.68秒合格恢复，身体触地记录为0；首落x=4.702米，超过原目标上界约70厘米，完整扣杀仍失败。同策略另一来球漏球；另一参数虽命中目标但速度不足且身体触地，三例保持独立。含侧视、俯视、慢放、原生接触与首次落点。[轨迹、原始事件、精确权重和源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261024_1400_evidence.zip)一并上传；2072帧解码与5附件SHA256核对通过。

最新[两代攻击恢复训练记录与38坐标新方案程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_attacking_recovery_two_generations_and_normal38_native_20261024_1425.zip)已上传并核对SHA256（192,690,446字节，437个归档成员）：保留两次同参数攻击恢复、完整失败落点、第三代339步中断、真实资源回执，以及新方案精确源码/输入/124项CPU检查/原生前100步。恢复时检出源码d1e7a5以获取原冻结来球文件，并解压此包和运行依赖。完整成功仍0，正式400未运行。

此前视频：[24秒真实攻击回球与恢复失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_20261024_1216.mp4)（24.42秒，1280×720、50fps）。CEM16候选17在一次己方空中拍线击球中出球8.19米/秒、向下1.65米/秒，并实际越网；随后身体触地，未观察到首次落点，完整扣杀仍失败。同参数另一独立回合是拍框误击后合格恢复3.92秒，没有拍线或合法回球。含侧视、俯视、四分之一慢放和原生接触定格；[原始轨迹、事件、权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_20261024_1216_evidence.zip)同时上传。全部1221帧解码通过，5附件GitHub大小/SHA256核对。当前GPU0运行CEM21拍面/球路搜索，历史视频保持原执行来源。

此前视频：[35秒最新实际效果：空中击球、触网失败与恢复对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_1048.mp4)（34.72秒，1280×720、50fps）。最新CEM12独立回合实际空中拍线出球7.30米/秒并越网，随后机器人触网；同方案另一回合合格恢复4.45秒但漏球；此前CEM07回球7.52米/秒并在对方场内首落，但目标和恢复失败。含侧视、俯视、四分之一慢放、真实接触点与首落点；三回合均未完成扣杀。[原始轨迹、事件、候选权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_1048_evidence.zip)已上传，全部1736帧解码通过，5附件GitHub大小/SHA256核对。视频所示CEM12已保存停止，当前CEM21拍面/球路训练仍在运行。

此前视频：[33秒当前真实效果：7.52米/秒回球、恢复与最新触网失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0931.mp4)（33.12秒，1280×720、50fps）。三个独立原生训练回合分别展示：CEM07空中拍线出球7.52米/秒并越网、场内首落但目标和恢复失败；另一回合击球后合格恢复2.92秒但球被网挡住；CEM08前移方案击球前触网失败。含侧视、俯视、四分之一慢放及真实接触/落点，完整扣杀仍0。[轨迹、事件、权重与精确程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0931_evidence.zip)一并上传，全部1656帧解码无误，5附件GitHub大小与SHA256已核对。当前CEM09新控制器不在视频内。

此前视频：[26秒最新实际效果：回球恢复与拍框失败对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0819.mp4)（26.04秒，1280×720、50fps）。来自两个独立原生训练回合：CEM03第二代候选16从地面起跳，在己方1.667米处拍线触球、越网、对方场内落地并合格恢复3.94秒；出球4.85米/秒且向上、横向偏出指定目标0.75米，完整扣杀仍失败。后一方案CEM04候选11拍框误击后在3.04秒身体触地，失败按原记录保留。含侧视、俯视、四分之一慢放、实际接触点和首次落点；[姿态、事件、候选权重和精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0819_evidence.zip)一并上传，全部1302帧解码无误，5个附件的GitHub大小和SHA256已核对。当前32参数新作业仍在运行，视频没有将它算作完成。

此前视频：[42秒当前实际效果：空中回球、目标命中与失败对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_three_cases_20261024_0713.mp4)（41.86秒，1280×720、50fps）。侧视/俯视/慢放展示三个独立原生训练回合：上一轮候选6的一次合法空中回球并合格恢复；同参数另一次执行命中目标但合格恢复失败；当前新策略候选0空中触球后在2.88秒身体触地终止。三个回合均未完成扣杀，未将它们拼成同一成功回合。保留当时的训练状态标注。[对应姿态、事件、权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_three_cases_20261024_0713_evidence.zip)已随视频上传；全部2093帧解码无错误，正式A/B各200回合尚未运行。

此前视频：[32秒同回合合法空中回球与合格恢复](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_return_recovery_20261024_0640.mp4)（2026-10-24，31.56秒；侧视/俯视/慢放/原生接触快照/真实首次落点）。TRAIN1候选6从地面自主起跳，在己方1.664米处真实拍线触球，球触网后合法越网、首次落在对方场内，同回合连续稳定恢复3.62秒、身体触地0。完整扣杀仍失败：实际出球4.827米/秒、向下0.250米/秒，均低于原攻击要求；首落点[3.004,-1.067]米，横向偏出原目标6.7厘米。同候选TRAIN65未触球且身体触地，失败对照也在视频里。[完整原始数据、候选权重与精确源码证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_return_recovery_20261024_0640_evidence.zip)随视频发布，全部1578帧解码通过。单例不代表正式面板成绩。

此前[高位触球与独立恢复视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_recovery_20261024_0555.mp4)及[证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_recovery_20261024_0555_evidence.zip)、[低位触球视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4)及[证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450_evidence.zip)保留。

此前完成训练：`ball_contact_recovery_cem02` 两代128次候选首回合共32次接触、15次空中拍线、24次合格恢复，其中1次拍线与合格恢复同回合；9次满足任务触球条件的实际越网、1次目标命中、0攻击达标、0完整扣杀、0数值故障。实际102400控制环境步、0.599666 GPU小时。第二代同候选6命中了目标并稳定3.83秒，但主落地后出现约1毫秒再次离地，原判据要求的合格落足为否，恢复仍按失败保留。没有通过修改判据重算成功。

此前完整训练 `ball_pitch_blend_cem03_resume01` 已完成两代256次候选首回合：28次接触、19次空中拍线、11次合格恢复，其中1次拍线与恢复同回合；8次符合任务触球条件的实际越网、1次目标命中、0攻击达标、0完整扣杀、0数值故障。实际204800控制环境步、0.601900 GPU小时。[第二代原始数据、完整统计及程序附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_pitch_blend_gen1_and_full_summary_20261024_0752.zip)已上传并核对摘要。原中断作业保留，恢复是从本代地面起点重放。

后一方案 `ball_momentum_velocity_cem04` 第一代128次首回合仅有2次拍框接触、0拍线、6次合格恢复、0完整扣杀和0数值故障。该固定速度方向方案已主动停止，第二代部分记录保留；守护回执仍为FAILED，不伪称整轮成功。实际0.355604 GPU小时、122368控制环境步；计费步数上限仍保留204800。

最新完整训练 `ball_native_reference_interception32_cem07` 已正常完成两代256次候选首回合：37接触、25空中拍线、10合格恢复、1次拍线与恢复同回合；8次几何越网中3次符合任务触球/规则条件，2次对方场内首次落地，0目标命中、0攻击达标、0完整扣杀、0数值故障。实际0.600544 GPU小时、204800控制环境步。较快候选32在两次同参数TRAIN1执行中出球7.468/7.524米/秒，实际越网且落在对方场内，但目标和身体恢复失败；另一候选完成拍线与2.92秒合格恢复，球被网挡回。没有把不同回合拼成完整成功。

上一轮32参数提前引拍 `ball_prepose_velocity32_cem06` 第0代128次执行为1拍框、0拍线、8合格恢复、0攻击/完整、0数值故障。实际正向法向拍头速度在部分采样达到2–3米/秒，但拍头高度和横向仍未对准来球。该方案在本代完成后主动停止，第二代31步保留，guard回执仍为FAILED；实际0.316229 GPU小时、106368控制环境步。

上一方案 `ball_power_body_bank_cem08` 已在完成两代后主动保存停止：256次首回合12接触、4空中拍线、4合格恢复、0同回合拍线与恢复、2任务有效越网、2场内首落、0目标/攻击/完整、数值0；140触网、53提前越网。第0代120次全身提案中113次规则终止、7次身体触地。FAILED回执及第三代21步保留；实际0.439862 GPU小时、149760控制环境步，原声明cap409600不改。

任务空间准备 `ball_cartesian_preparation_cem09` 完整第0代128回合为0接触/0恢复/0完整，124身体触地，其中41回合在来球前已结束，数值0。该准备动作使行走失稳，已保存停止；第1代104步与FAILED回执保留，实际0.151311 GPU小时、52608控制环境步。

关节准备修正 `ball_joint_preparation_cem10` 已在完整第0代后保存停止：128执行3接触/1空中拍线/6合格恢复/0同回合拍线恢复/0有效拍线/0攻击/完整、数值0；88触网、14提前越网。第1代131步和FAILED回执保留，实际0.345378 GPU小时、119168控制环境步。其余合格恢复均漏球，没有用漏球恢复的最佳参数替代快击球初始化。

工具运动预测 `ball_tool_motion_anticipation_cem11` 完整第0代128回合为3接触/1空中拍线/5合格恢复/0同回合/0有效拍线/0攻击完整、数值0；85触网、15提前越网、2身体工程越界。已保存停止，下一代31步与真实FAILED回执保留；实际0.307542 GPU小时/106368控制环境步。

本轮新进展：CEM16候选17的一次真实己方空中拍线出球达到8.1947米/秒（向下1.6518米/秒），实际越网余量7.45厘米，首次满足原研究攻击阈值；但3秒处身体触地，首落点未观察到，完整扣杀仍失败。此为前300步的一次已闭合训练回合，整代和正式400尚未完成。[原始事件、300步全身姿态与精确权重](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_attack_velocity_crossing_native_partial300_20261024_1208.zip)已上传并核对SHA，下一步重点把攻击和合格恢复接到同一回合。

当前GPU0运行 `ball_slew50_near_attack_positive38_cem24`：从真实接近8m/s且合格恢复的候选出发，重新采样正方向拍面、挥拍时机和速度。实际前400步77接触/15拍线，但14个拍线回合仍待结局，尚无新完整成功；不能把部分接触统计当整代成绩。所有物理、执行器与原目标保持。上一轮CEM23两代256次候选执行（仅两种不同来球）66拍线、28次拍线与恢复同回合、30目标，0攻击达标/完整、数值0；已保存第2代73步中断并停止。其最快近攻击案例仍为7.9665m/s且首落x4.508，原速度和落点门槛都未通过。正式A/B各200回合尚未运行。

拍面方向38坐标接口已在CEM21原生启用。新增坐标只旋转期望拍面方向，实际出球仍由碰撞产生；没有新增物理关节或改变执行器限制、球拍、球网、原目标区与成功标准。默认38与实际37控制器在相同公开输入上的指令逐位一致，当前38训练效果须由新执行验证。

[两轮失败的完整第0代、后续中断记录、本轮源码/权重和前100步原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_arm32_failures_and_interception_program_20261024_0833.zip)已上传（185,838,293字节），125个归档文件与GitHub附件大小/SHA256均已核对。各作业精确启动源码、实际资源回执和新旧权重分开保留。没有后续排队作业，正式A/B各200回合尚未运行；原8米/秒、向下分量、目标落点、合格跳跃恢复和全部物理条件保持。

此前上部拍线课程 `ball_upper_strings_cem01` 的128次候选首回合全部未触球、未完成合格恢复，完整成功0、数值故障0；实际77120步、0.464638 GPU小时。[完整失败轨迹、权重、精确源码及接续程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_upper_strings_failed_and_contact_update_20261024_0617.zip)保留，包含全部64环境50Hz原生全身姿态、运行依赖和CPU攻击窗口提案。

更早恢复课程128次执行中，13次合格跳跃恢复均漏球，4次接触中1次为空中拍线，0完整成功、0数值故障。这批的第一代[原始数据/权重](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_gen0_and_rule_code_20261024_0528.zip)、第二代触球的[早期部分记录与当时程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_contact_and_curriculum_v3_20261024_0532.zip)保留；不同批次没有混算成功率。

[完整恢复训练、近网修复与程序附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_native_rule_fix_and_recovery_20261024_0605.zip)包含128次原始回合、权重、精确启动源码、10层USD、3个通过的原生近网边界回归，以及此前2个失败尝试，固定于提交 `dc0a4b2`。修复使用实际GPU刚体姿态计算球拍实体与网平面的关系，避免DirectGPU下CPU场景查询的旧姿态。目标回归通过不等于策略验收。当前分支新增[接触保留与实测恢复腿参数交叉课程](reports/G07_full_racket_v4/CONTACT_RECOVERY_CROSSOVER_PLAN_0612.json)，[51项CPU测试](reports/G07_full_racket_v4/CPU_TESTS_0612.txt)通过；新课程原生训练已实际启动，之前的失败按原始结果保留。

复现时同时解压[运行依赖附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_runtime_dependencies_20261024_0608.zip)，补齐配置读取的冻结Go2步态/站立权重、标准PiPER无夹爪源URDF及其引用网格。清单逐项列明SHA；与最新程序包解压到同一仓库根目录。外部Go2 USD层的原始路径映射见 `configs/full_racket_usd_dependencies_v4.json`。


日期：2026-10-21。对象：Go2 + 标准 AgileX PiPER + 固定安装球拍；Isaac Lab；PPO。

**交接阅读入口（2026-10-22）：[给 GPT 的当前情况与代码导读](docs/GPT_HANDOFF_20261022.md)。**
包含真实结果、失败经过、关键程序、尚未验证的假设和可直接复制给 GPT 的问题。
代码与文本证据保存在本仓库；权重、完整运行记录及资产见
[交接附件与恢复说明](docs/HANDOFF_ARTIFACTS.md)。上传不代表训练验收通过。

**当前状态（2026-10-24北京时间）：新版完整球拍、真实球场和击球规则正在适配训练。用户新授权仅GPU0、无累计GPU小时上限和截止；旧窗口与费用保留。当前资产full-racket-court-v4-r3，675 mm完整球拍、6.1 m宽悬空网、20.709991 kg总质量。原生触球、网体/网柱、网下/网上通行及直立持拍自碰撞回归已运行；新版400案例已冻结，正式成绩尚无。当前在进行真实动力学全身动作搜索。**

G00–G06的通过属于旧短拍板资产，不能继承为新版通过。旧G07正式400未达标（A完整50%/恢复89%，B完整35.5%/条件恢复74.4898%，数值0），旧权重、数据和失败保留。新版模型来源、规则映射与假设见[MODEL_RULES.json](reports/G07_full_racket_v4/MODEL_RULES.json)，[新旧球拍与当前实测对照](reports/G07_full_racket_v4/model_task_comparison_20261025/REPORT.json)补充了其创建后的动态证据和正式失败结果，评测配置见[新版协议](configs/g07_full_racket_eval_v4_r3.json)，实际下一步见[运行状态](status/NEXT.md)。[新版完整程序、资产、失败记录和无来球学习检查点](https://github.com/lzqw/unitree-smash/releases/tag/g07-full-racket-v4-progress-20261024)已上传并核对大小及SHA256。最近完成材质一致、62.5us的256次无球候选执行（含精英重测），其中8次无触网/无身体越面并完成跳跃恢复；8环境原生扰动写入/公开观测/局部reset烟测通过，真实来球联合训练正在运行。[增量权重与原始证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_full_racket_adaptation_20261024_0400.zip)已上传。上述烟测不等于独立正式400通过。

视频入口：[38秒当前完整球拍r3动作与失败记录](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_r3_20261024.mp4)（侧视、俯视、慢放；无球恢复、真实来球漏球、触网失败，尚无合法完整扣杀）、[对应原始轨迹证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_r3_20261024_evidence.zip)。历史视频：[r1机械测试与旧策略对照](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/Current_effect_v4_20261024.mp4)、[旧短拍板效果](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/G07_current_effect_comparison_20261024.mp4)。历史报告：[旧G07实测报告](reports/G07/REPORT.md)、[旧视频、权重与完整轨迹附件](https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023)。G04结果与接口见[训练总结](reports/G04/autonomous_learning_20261022/REPORT.md)、[策略接口](docs/G04_POLICY_INTERFACE.md)。旧交接文档保留历史状态。
旧资产为约20.647 kg浮动系统；新版完整球拍资产为约20.710 kg，仍保留标准PiPER、Go2和原执行器限制。
统一 794 维观测、18 维动作、原生 PPO、独立评测及可恢复检查点已实现。
最新状态与累计资源记录分别见 `status/STATUS.json`、`status/NEXT.md` 和 `status/GPU_LEDGER.json`。
工程通过或存在权重文件均不表示策略达到成功率要求；G04 的两个固定面板必须各达到 90%。

## 开始
1. 将本目录内容放入新的研究工作区；已有项目先建立分支，不覆盖原有 AGENTS.md。
2. 阅读 `configs/budget.json` 和 `status/NEXT.md`。当前授权窗口为`g07-full-racket-unlimited-20261024`，仅GPU0、无累计GPU小时或截止限制；保留所有旧费用、单任务有限超时和单GPU作业保护。
3. 在同一工作区打开 Codex CLI/IDE。先输入 `CODEX_GOAL_PROMPTS.md` 的“第一轮启动目标”。
4. 搜索、诊断、失败重试和评测均计入对应授权账本；使用`scripts/run_gpu_job.py`，不可因恢复会话重复启动作业。G08以后尚未授权。
5. 不需要每轮重贴完整方案；Goal 引用这些持久文件。换会话时先读状态和最近报告。

官方 Goal 用法为 `/goal <目标>`；`/goal` 查看，`/goal pause` 暂停，`/goal resume` 恢复，
`/goal clear` 清除。见 `docs/SOURCES.md` S1–S3。若命令未启用，可按官方文档检查
`codex features enable goals`，不要编造 `codex --goal` 或 `codex goal run` 命令。

暂停 Codex 并不意味着其启动的训练进程/调度器作业必然自动停止。
本项目必须另行记录作业 PID/Job ID，支持保存并停止项目自己的作业。

## 阅读顺序
- `AGENTS.md`：禁止事项、证据、预算与执行规则。
- `PLAN.md`：阶段依赖和任务清单。
- `TRAINING_SPEC.md`：算法、观测动作、课程与数值要求。
- `ACCEPTANCE.md`：通过标准及研究结论边界。
- `goals/Gxx_*.md`：本阶段实施细节、命令契约、失败处理。
- `configs/acceptance.json`：机器可读的验收条件。
- `status/STATUS.json`：初始全部 TODO，不得预填 PASSED。

## 已可运行的检查
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/verify_goal.py --root . --goal G00
```
G00–G03 的真实证据已保存在对应 reports 目录；缺少证据或未达到指标的阶段应当校验失败。
校验器检查证据文件、摘要、依赖、指标和必需产物；**不独立证明日志来自真实仿真**，
仍需保留原始逐回合数据、执行日志和可复现命令以接受审计。

## 最小可交付边界
G00–G07 = 能审计的跳跃扣杀仿真基线。
G08–G09 = 可恢复状态初始化与恢复约束事件选择研究分支。
G10 = 冻结测试、消融、多种子及结论审计；结果可以是不支持方法优势。

本次[完整CEM07统计/第二代原始数据及当前四代训练程序、权重和前100步原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_interception_complete_and_body_learning_20261024_0912.zip)已上传（78,543,764字节，SHA256和49个归档文件均已核对）；第0代完整姿态见此前71MB附件。正式A/B各200回合仍未运行，完整扣杀仍为0。

最新[前移准备失败与任务空间准备控制的原生证据附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_body_preparation_failures_and_cartesian_native_20261024_0938.zip)（120,166,563字节，84个文件SHA及GitHub摘要/大小已核对）包含CEM08两代完整轨迹、第三代21步中断、真实FAILED回执，以及CEM09精确源码/输入、81项CPU检查、120路回放和实际前100步。首100步只证明初始化与执行，后续触网规避与扣杀效果仍待实测。

最新[任务空间准备失败与关节准备修正的程序/原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_cartesian_failed_and_joint_preparation_native_20261024_0952.zip)（54,384,135字节，78文件与GitHub SHA核对）包含CEM09完整第0代失败和第1代104步中断、CEM10精确源代码/输入/86CPU检查/真实前100步。CEM10随后已实测前200步128回合均未提前结束；后续触网、击球与恢复仍待运行结果。

当前[关节准备修正第0代前400步原始证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint_preparation_native_partial400_20261024_1000.zip)（49,772,106字节，26成员和GitHub大小/SHA核对）已发布：121首回合结束、7待恢复，1空中拍线+2拍框；拍线出球7.235米/秒并越网，但随后规则故障且恢复失败。3秒截点机器人规则终止由116减至92，触网仍未解决，完整扣杀仍0。四代作业继续运行，正式400未运行。

当前跟踪诊断：4.5软件方案前400步的原始目标变化率最高78.97rad/s，实际受限目标4.500012rad/s，79次接触的目标记录均与真实动作一致；实际100Hz臂速峰值5.000992rad/s的超调保持。actual-to-limited拍面夹角中位10.51度，但actual-to-desired仍13.36度，不能以指令跟踪改善代替合格出球与完整成功。

[最新38坐标第0代完整原始数据与跟踪修正程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_normal38_generation0_and_rate_tracking_update_20261024_1448.zip)已上传，72,744,177字节、52归档成员及GitHub大小/SHA256核对通过。包含完整128次候选执行与0完整成功、目标回球恢复的弱攻击案例，以及尚未原生执行的f40cf20目标速率方案；此前[实际前400步快照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_normal38_actual_partial400_and_train128_entry_20261024_1435.zip)提供逐事件拍面/目标速率诊断的冻结输入。

5rad/s软件方案第0代已完整运行：128次候选执行（两种不同来球），24空中拍线、80合格恢复、13次拍线与恢复同回合、12目标命中、0攻击达标/完整、数值0。较快的实际候选54出球7.9665米/秒并恢复3.184秒，但速度仍低于8、首落x4.508米超过原目标，保留失败。同参数另一来球为拍框。当前下一代继续；已准备围绕该实际候选的正方向新采样，尚未原生执行，不改变任何原验收门槛。

[完整试验记录、当前程序和初始化证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_normal_slew_results_and_current_program_20261024_1525.zip)已上传，292,239,298字节、520归档成员与GitHub大小/SHA核对通过，附原冻结来球文件；当前5方案前400步另有[原生遥测附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_slew50_actual_partial400_20261024_1530.zip)，15结束113待定，不能当完整成绩。

[最新5软件目标方案第0代完整轨迹、原始权重与采样诊断](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_slew50_generation0_and_power_sampling_20261024_1545.zip)已上传，79,110,726字节、32归档成员及GitHub大小/SHA核对通过，严格保留7.9665m/s未达到8和落点超界的失败。当前第1代继续；新正方向输入未原生运行，120路506步CPU预检已完成。

[软件5方案完整两代与新正方向搜索程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_slew50_two_generations_and_positive_search_20261024_1605.zip)已上传：167,207,887字节、448成员、全部数据SHA核验；CEM23两代原生失败/73步中断与CEM24前100步分开标识。后续400步快照不在此归档内，CPU反馈增益方案尚未原生运行。


最新 CEM24 四代原生结果（2026-10-25）：512 次候选执行、409600 控制环境步，83 次空中拍线、16 次攻击门槛、54 次拍线与合格恢复同回合、6 次目标落点、完整成功 0、数值故障 0。第3代候选11在TRAIN65中实际出球8.0968米/秒、向下2.0671米/秒、恢复3.02秒，首落x=4.2200米，目标后沿仍超22.0厘米；候选48出球8.1752米/秒并恢复但首落x=4.4305米。

最新[真实攻击恢复视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261025_1710.mp4)（31.08秒）展示CEM24候选11的同回合攻击、越网和恢复，以及同参数另一来球失败。完整任务仍0。四代原始数据、权重、视频和终端核验已上传至[G07_CEM24_four_generations_and_latest_attack_recovery_20261025_1710.zip](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_CEM24_four_generations_and_latest_attack_recovery_20261025_1710.zip)。

当前GPU0运行CEM25b相对公共来球预测引拍方案：以真实CEM23 TRAIN65接触恢复回合为参考，gain=.35；默认路径逐位等价和142项CPU测试已验证，CEM25b前100步原生初始化已验证，尚无新的接触成绩。


最新 CEM26b（2026-10-25）完成4代、512候选/409600控制步：84次空中拍线、11次攻击门槛、30次拍线+恢复、5次目标落点、完整0、NF0。第3代候选43在TRAIN65同回合出球8.3196米/秒、vz=-2.2454、恢复3.873秒、身体触地0，首落x=4.1707米，目标后沿仍超17.1厘米。

[最新真实攻击恢复视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261025_1900.mp4)展示候选43及独立失败回合；完整扣杀仍0。相对引拍gain=.10的代码、权重和CEM26原始数据随后上传。


完整成功里程碑（2026-10-25）：CEM28原生TRAIN搜索在两个不同候选/代际中各得到一条同回合完整扣杀。候选30：TRAIN65出球norm8.1013米/秒，首次落点[3.9766,-.8932]米，恢复3.6859秒，身体触地0；候选39：norm8.0116米/秒，首次落点[3.8529,-.7712]米，恢复4.0119秒，身体触地0。两者均为独立TRAIN回合，正式128分布与A/B400尚未通过。

[完整扣杀视频（候选39）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_complete_smash_20261025_2030.mp4)与[候选30视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_complete_smash_20261025_2005.mp4)均来自实际全身原生轨迹，失败回合也保留。当前GPU0运行TRAIN[0..7]分布适配搜索。

## 2026-10-25 独立评测与分布适配

CEM28 的两个原生 TRAIN65 同回合完整扣杀里程碑已保留并上传，但它们不代表分布通过。candidate39 在冻结策略锁下完成正式 A/B 各 200 回合：A 为完整 0/200、空中接触 21/200、接触后恢复 7/21、数值故障 0；B 为完整 0/200、空中接触 9/200、接触后恢复 1/9、数值故障 0。按较差面板聚合，形式验收为 false。

随后完成 candidate8 的 TRAIN128 诊断（完整 0/128、空中接触 16/128、接触后恢复 8/16、数值故障 0），并将固定地面重置扰动 A/B 支持加入 TRAIN 搜索，默认无噪声路径保持不变。CEM29 无噪声恢复两代和 CEM30 B 级扰动两代均保存了负结果；291 项测试通过。详细锁、面板、原始事件哈希和搜索收据见 [`formal_and_robust_20261025.json`](reports/G07_full_racket_v4/formal_and_robust_20261025.json)。

交付包：[G07_formal_and_robust_delivery_20261025.zip](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_formal_and_robust_delivery_20261025.zip)，包含策略锁、A/B 400 回合事件、TRAIN128 诊断、分布搜索终端结果和扰动训练代码快照；SHA256 为 `736ea5f05e3128986f07c63c37c5b53fba5461547f1eb0a0b31755aa4475f46d`。

[Candidate5完整评测与原生轨迹证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_CEM31_formal400_native_trajectories_20261025.zip)（272,629,446字节，SHA256 `85b9b6cb1b59948e96332a29c2ba6052f00c1b9b4c2f679ee25ed9e95962697a`）。包含A/B全部400回合全身姿态、事件、锁定源码、TRAIN128、CEM31群体与新评测视频；GitHub下载回验一致。
