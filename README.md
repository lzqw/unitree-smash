# QuadSmash：Codex /goal 实施与训练任务包

[本轮事件短板诊断](reports/G07_full_racket_v4/residual_n2048_event_diagnostic_20261025/REPORT.json)已重算第56次更新的原14048条训练记录：129次空中攻击仅5次同时命中目标，127次目标中122次出球速度低于8米/秒；其中2次攻击与目标均满足的回合因实际机身着地、恢复0而失败。前三批各2048回合均已闭合，空中197/192/200；后续尚未闭合的批次单独标明，不能将截尾统计当作冻结策略退化。报告含每TRAIN案例计数和所选原始回合，可用最终原始数据的相同前缀SHA复算；当前原生训练继续，未证明新奖励有效或正式通过。

第96次更新后的[正式A/B后续队列](reports/G07_full_racket_v4/residual_iter96_formal_queue01/PREDECLARED_PLAN.json)已实际登记等待，28项CPU门槛检查通过。它只在真实新固定TRAIN128达到原全部比例且数值故障0后，锁定同一权重与源码、依次执行原A/B各200回合；未达则跳过并继续改进策略。队列登记时第56次更新的14048回合独立重算为1142空中/460条件恢复/3完整/NF0；后续进度见下方中点记录，均不是冻结策略或正式成绩。

[第96次更新的最终训练证据](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/FINAL_TRAINING_REVIEW.json)已保存：64次续训实际新增16777216个控制环境步，训练守护正常退出；39077个闭合回合已独立重算为3332空中/1244条件恢复/8完整/NF0，仍是变动策略训练数据。全部8例现已匹配[真实原生拍线接触](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/FINAL_CONTACT_CONFIRMATION.json)、28体姿态和刚体点速度，独立重算最大误差0.00002024米/秒，低于原0.0001容差；这是post-PhysX状态一致性，不是独立测得的入射拍速。最终canonical96与导出权重已核对；固定TRAIN128正在GPU0执行，两个新版视频和原A/B仍待。旧快照的pending字段保持创建历史，以最终确认为准。TRAIN61的负net_margin诊断值及限制仍在[原案例说明](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/CASE_NOTES_UPDATE85.json)，不声称有正的最小净空，也没有完整训练动作录像可补作。

[续训中点独立核验](reports/G07_full_racket_v4/residual_n2048_midpoint_20261025/REPORT.json)：全局update64、本轮新增32/64更新，8388608新增控制环境步；全部19164训练回合为1516空中/599条件恢复/4完整/NF0，模型与Adam有限。首五批各2048均已闭合，后续未闭合批次单独标记。原GPU0继续至iter96，原门槛、模型和规则保持；这些计数仍非冻结策略或正式A/B成绩。

[新增2048环境完整训练回合](reports/G07_full_racket_v4/residual_n2048_first_complete_20261025/REPORT.json)：TRAIN58_LANE1338_EP2从地面执行，在1.753米处合法空中拍线击球，出球8.003米/秒、向下2.818米/秒，实际越网后首次落在[3.794,.139]米，合格恢复2.94秒，无失效违规。原7766回合记录及actor39..45权重谱系已保存；该回合经历权重/归一化变化，不是一个冻结策略成绩。GPU0继续原训练，已在固定评估前另行登记TRAIN58真实轨迹复测视频；原训练没有保存完整运动轨迹，不能补造成原回合视频。

[最新69.38秒实测视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_residual_PPO_iter32_20261025_v3.mp4)已上传并下载SHA回验：同一iter32固定策略，保留评估前预选的TRAIN19/23/107/0四个失败回合，另加明确标为评估后选择的TRAIN8。TRAIN8在己方1.748米处真实空中拍线攻击，出球8.04米/秒、向下1.55米/秒，实际越网并合格恢复4.329秒；首次落点x4.537米超过原目标上界0.537米，完整仍失败。侧视/俯视、慢放、真实接触与首次落点，全部3469帧解码；v3补全远落点并保持恢复近景，前两版及拒绝原因保留。

[iter32完整训练、固定128回合原始轨迹、源码、权重与视频包（123.9 MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_PPO_iter32_fixed_TRAIN128_video_training_and_raw_20261025.zip)已上传回验：151成员全部CRC/SHA通过，ZIP SHA256为`db25f79073218916780c766bc783ba613ee0e73d8f62c8d8a955f8a3da338c45`。其中的2048信息仅为另行标注的启动检查摘要，不含其进行中的原始训练数据或策略成绩。

[固定iter32诊断报告](reports/G07_full_racket_v4/residual_iter32_fixed_diagnostic_20261025/REPORT.json)与[128行固定评估CSV](reports/G07_full_racket_v4/residual_csv_queue01/iter32_fixed_episodes.csv)可逐回合核对；训练中3171回合的417空中/210恢复/1完整不作为固定actor32成绩。

当前进度（2026-10-25）：PPO iter32固定TRAIN128已完成并独立重算：合法空中22/128（17.19%）、条件恢复14/22（63.64%）、攻击3、目标3、完整0、数值故障0。第8次更新为18空中/9恢复/0完整，横向基线为19/12/0；这是不同原生执行的描述性对照，不作统计或因果结论。本轮3次空中攻击与3次目标命中没有同回合重叠，完整任务仍未解决。GPU0现已在2048个真实地面环境（128来球各16副本）从iter32继续学习64更新至iter96；正式G07仍未通过。

[联合落点奖励候选分支](https://github.com/lzqw/unitree-smash/tree/g07-joint-placement-reward-v4-20261025)已通过42项CPU功能检查和真实iter32 actor迁移检查。[隔离运行依赖恢复](https://github.com/lzqw/unitree-smash/blob/119b461010bee619905ad8736eb982b1fcf55160/reports/G07_full_racket_v4/joint_placement_prerequisites_20261025/READINESS.json)现已完成：14个历史文件按原SHA复制，另补固定步态权重；实际候选200路CPU控制器加载、200×794公开TRAIN观测推理及10层USD文件SHA检查通过。它提供实际空中攻击、越网后的首次落点反馈，原门槛不变；仍未原生训练、未证明策略改进。当前2048主训练及原视频/正式后续队列继续保持原源码，先看第96次更新固定评估。

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
