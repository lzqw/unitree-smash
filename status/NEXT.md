当前唯一GPU0作业（2026-10-24T14:20:36.363246+00:00核查）：ball_attack_recovered_normal38_cem21，guard1439677 / child1439690，session55877；精确启动源码d1e7a53559cfd5c752e753c3a71a7c3c8a96cdff。4代×64候选×TRAIN[1,65]，有限7200秒/409600控制环境步；无累计GPU小时/日历截止，仅GPU0，无后续队列。继续观察同一进程，不能因观察超时重开。不手改GPU_LEDGER/RUNS、preexisting RESOURCE_STOP及无关MPS。

输入attack_recovered_normal38_ready03_1415：policy SHA4fb9e088f283390862f18fde33dfe6d46bc4f81008ba305d73436ef87b92eee2，bank SHAba1325be61756734c30fdbc6249cdbbc57241b842e88e959524a61258d9fa0fe。38为策略坐标数，硬件仍18关节。60bank=6期望拍面向下偏角[0,.025,.05,.075,.10,.12]×5挥拍lead偏移[-.02,-.01,0,.01,.02]×2arm gain倍率[1,1.1]；Gaussian仅[7,14,37]，其余身体/恢复坐标保持实际重复攻击恢复37参考。sigma_fraction.06。新增TRAIN方法使用真实首次落点与原目标距离/内部裕量、出球速度模长，优先完整成功、攻击+恢复、拍线+恢复；不改任何原生成功标志或正式协议。

新38方案124CPU测试通过、120路287步指令预检有限有界；默认38与实际37在相同公开输入上逐位一致。实际100步、64×38/60bank/canonical参数名/mask/sigma0/128状态流/0提前结束已冻结核对：reports/G07_full_racket_v4/normal38_initialization_1421/REPORT.json。此时尚未来球，不能继承37效果或称38已击球成功。下一步观察原生接触、出球范数与向下分量、真实越网和首次落点，然后等同回合恢复。

CEM20已实际保存停止：两代256次候选执行、仅两种不同TRAIN来球，共159接触/89拍框/72空中拍线/33攻击/97合格恢复/17拍线+恢复/6攻击+恢复/64场内首落/4目标/0完整/0机器人触网/NF0。第二代128执行中35拍线/14攻击/77恢复/15拍线+恢复/5攻击+恢复。所有攻击+恢复仍超出原目标；原始frame/string标志可能同回合重叠，不能相加当总接触。第三代339步保存，3首回合结束、125待结局，不能当完整代；实际248192步/.709536062554GPU小时，guard FAILED/child exit0/无completed回执，原计费cap409600保留。核验reports/G07_full_racket_v4/poststroke_retreat37_terminal_1420/REPORT.json。

实际同37参数在TRAIN65重复得到出球8.10米/秒、合格恢复3.68/3.66秒、身体触地0；首落x4.702/4.554米，超过原目标上界4。两次强击球恢复证据分别attack_recovery_gen0_env101和attack_recovery_repeat_gen1_env3/MILESTONE.json；同参数TRAIN1分别漏球/拍框，不能组合成成功或宣称泛化。

最新41.44秒视频已上传并核对2072帧及5附件SHA：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261024_1400.mp4 。侧视/俯视/慢放，三回合独立；视频属于CEM20generation0，含强攻击+恢复但目标偏远、同参数另一来球漏球、另一候选目标命中但速度不足且身体触地。

当前完整成功仍0，正式A200+B200未运行。需要先获得同回合原门槛完整链，再扩大原TRAIN128及扰动覆盖，然后固定权重独立正式400；不能把两种来球的候选重复执行当成分布评测。原要求norm>=8、vx>0、vz<=-1、原目标X[2,4]/Y±1、原合格跳跃/落足/稳定恢复0.8秒和数值0全部保持。球拍675mm/20.709991kg、中心网高1.524m/侧1.55m、原18关节硬件限额保持。旧短拍板只作为历史对照，不继承成绩。上一NEXT原文完整保存在status/NEXT_before_normal38_1420.txt。

最新实测更新（2026-10-24T14:39:41.624775+00:00）：CEM21第0代128候选执行完成，85接触/38空中拍线/4攻击/69合格恢复/17拍线+恢复/2攻击+恢复/33场内首落/7目标/0完整/0机器人规则/NF0；其中5次目标+恢复均速度不足。当前第1代同guard继续，先观察完整结果。第一代正式总结ball_attack_recovered_normal38_cem21/GENERATION0_SUMMARY.json；TRAIN1env14目标首落[3.842,-.454]/合格恢复4.143秒，但速度6.108且vz-.9525，仍失败。

新增诊断normal38_actual_face_tracking_1431：38拍线事件实际与指令拍面夹角中位18.89度，重建实际法向误差最大2.2e-6；normal38_target_rate_diagnostic_1437：13.1rad/s为指令速率95分位，最大101.08，实测臂速捕获窗口最大5.3702rad/s，名义5rad/s未改，超调原样保留，不能宣称所有实际速度始终在5以内。

后续f40cf20分支未合入/未原生执行：可选软件臂目标速率4.5/5rad/s、提前挥拍；134CPU测试、400步默认控制逐位一致/腿不变/命令速率有界，以及4.5方案120路287步bank预检通过。输入arm_target_rate45_normal38_ready01_1444，policy6829d44c7db041846fd8a39c0468b000f289cda1d97017fec8c3635fc8183e9b，bank3f792047df0f14e6deed66733459e44f73a941febdc0df46034bd1447d71e04d。60bank=6方向[-.04,0,.02,.04,.06,.08]×5lead偏移[0,.02,.04,.06,.08]×2gain[1,1.1]；mask[7,14,37]，源CEM21gen0best已冻结normal38_gen0_best_for_slew_1444，SHAd4f116ad254aa7a3d0991b2d96177e34d3e386cadaad2d1695468b0eea7a606a。若CEM21下一完整代仍无完整成功且跟踪短板持续，再通过已验证guard保存停止，等待实际终态/回执，再合入f40cf20启动ball_arm_target_slew45_normal38_cem22；不重复GPU作业、不伪造receipt、恢复全部gate。详见arm_target_rate_ready_1440/PLAN.json。

另有TRAIN128固定策略/初始扰动诊断入口2e20830（已包含于f40cf20）完成5CPU测试与两条旧原生行交叉核对；新入口尚未原生执行，当前两种来球不能算128种覆盖。仍需首个完整同回合链、扩大TRAIN128/扰动，再冻结A200+B200正式评估。
