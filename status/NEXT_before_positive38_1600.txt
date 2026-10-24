当前唯一GPU0作业（2026-10-24T15:22:48.646757+00:00核查）：ball_arm_target_slew50_normal38_cem23，guard1449964 / child1449983，session33189，精确启动源码4ef7961a79ffb50be48fd9175f197138d62dad88。4代×64候选×TRAIN[1,65]，7200秒/409600控制环境步；仅GPU0、无累计GPU小时或日历截止，真实账本和单作业保护保留，无额外队列。不得因观察超时重启；不手改GPU_LEDGER/RUNS、旧RESOURCE_STOP/queue或无关MPS。

本轮软件臂目标变化率5rad/s，执行器原始配置仍为5rad/s/100Nm（未实机标定假设），腿23.5Nm/30rad/s、全负载20.709991kg/675mm拍、真实球网/原任务gate均保持。与CEM22所有control/physicssourcehash完全相同，仅策略输入与搜索网格变化；不是提高硬件限额。before/after-slew臂目标、实际公开臂位置/速度、时钟与firstepisode mask以100Hz保存，物理实际超调不裁剪。

输入arm_target_rate50_normal38_ready01_1540：policy SHA84f0da4785fd3b11bac910066c6eb193d31f3d1315ba86ff9cf651aced7f11c0，bank SHA918badac7467e561f50ec7d9d89e57b1fd37290bb5ca91791bcb2bcb8d18defd。父策略为实际CEM22候选15，SHA37ed738828dab99ae4c505e6f024998cb43d72deadfbf27883f56b94d5132f0a（slew45_both_case_recovery_gen0_c15_1540/policy_evidence.pt）；两个来球都完成拍线与合格恢复，但出球7.452/6.849m/s且首落x4.680/4.121，完整0，绝不继承为5方案成绩。

60bank=6法向偏角[-.12,-.08,-.04,0,.04,.08]×5lead偏移[-.02,0,.02,.04,.06]×2gain倍率[.95,1]；相对父策略lead.02，实际范围[0,.02,.04,.06,.08]，gain1.045/1.1。Gaussian mask[7,14,37]，sigma_fraction.06，其余身体/恢复保持，38仍是策略坐标而非物理关节。134CPU测试/485步默认兼容及4.5/5命令率检查已通过；本轮60bank在120路506步公开输入预检全部有限有界。

本轮实际前100步已冻结核验slew50_initialization_1550/REPORT.json：64×38/60bank/canonical名/mask/sigma0/128状态流/0提前结束，首2路真实动作与after-slew解码差1.2e-7rad、下一公开previous_actions逐位一致。此时before/after目标最高均2.765rad/s、限速尚未绑定、无来球和接触；不证明本轮击球或恢复。下一步观察200/300/400步限速实际绑定和原生出球，随后等800步完整回合，不把pending案例判成功。

CEM22软件4.5方案已实际保存停止：第一代128候选执行/2种不同TRAIN来球为79接触/50拍框/29空中拍线/52合格恢复/12拍线+恢复/24场内首落/11目标/0攻击完整/0机器人规则/NF0；所有出球<8（最高7.4522），不能因限速验证通过就认定任务前进。第1代127步中断、0终态，实际118656步/.340813255786GPU小时，guardFAILED/childexit0/无completed回执，声明cap409600不改。slew45_terminal_1544/REPORT.json已核验。候选15两选定来球的真实4.5策略臂速度100Hz峰值4.510/4.999rad/s、恢复3.714/3.755秒，身体触地0；它不是分布验收。

CEM22实际前400步快照/分析：slew45_contact_prefix_1535，36closed/92pending、79接触/29拍线/21拍线待结局；before目标最高78.97rad/s，after4.500012，最大目标调整.9016rad，全部79事件的limited目标与PhysX实际command差<=2.4e-7rad。实际样本峰值5.000992rad/s和2个>5.0001样本原样保留；这是100Hz控制观测，不是物理子步最大值。拍线actual-to-limited指令夹角中位10.51度，但actual-to-desired仍13.36度；不能把贴近受限指令当更好球路。

CEM21此前两代256候选执行（仅2种不同来球）70拍线/10攻击/132恢复/26拍线+恢复/4攻击+恢复/65场内首落/12目标/0完整/0机器人规则/NF0；第三代50步中断/0终态，实际211200步/.606057793142GPU小时，FAILED回执与原cap409600保留。normal38_terminal_1516/REPORT.json；第1代globalbest未更新、实际拍面跟踪中位14.448度，因此转入限速/相位适配。

近网规划near_net_planning_1505仅是savedTRAIN65公开输入预测、自由飞行目标网格和同一观测机身姿态下局部IK：X1.15较1.05有更宽球路余量，但局部IK约8.5cm误差，不能只改球路平面。未来如需近网应联合移动/引拍；局部失败不证明物理不可行，旧短拍板仅几何对照、无旧动力学/恢复对照。

完整目标仍未通过：原norm>=8/vx>0/vz<=-1、真实越网首落原X[2,4]/Y±1、原合格跳跃/落足和连续.8秒恢复都不变。先得到完整同回合链，再扩大原TRAIN128和原幅度扰动，锁定源/权重运行A200+B200，按较差面板full>=.50/air>=.70/conditional-recovery>=.85/NF0。TRAIN128评测入口已合入但尚未原生执行，正式400尚未运行，旧模型得分不继承。

最新视频仍为已验证41.44秒CEM20三独立原生案例：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261024_1400.mp4 ，并非新5软件方案效果。此前NEXT保存在status/NEXT_before_slew50_1545.txt；全部历史模型/权重/失败原始数据保留。

最新完整结果（2026-10-24T15:46:07.682205+00:00）：CEM23第0代128执行/两种不同来球完成，84接触/60拍框/24拍线/80合格恢复/13拍线+恢复/19场内首落/12目标/0攻击完整/0机器人规则/NF0；第1代同guard继续，未停止。实际候选54env109/TRAIN65出球[7.694837,-1.462682,-1.454552]，norm7.966534895仍小于8，首落[4.508450,-.814484]超出原目标，恢复3.183952秒/身体触地0；当前仍失败，不四舍五入为达标。同参数TRAIN1frame失败保留。slew50_near_attack_gen0_c54_1540保存实际策略（SHA634da4f6fb65298ccf5afb50e7807324efc0bab3add5abe0b0064c26aeaa0a68）和规范判据核对。

当前采样短板已量化：slew50_near_attack_sampling_audit_1545.json，v3tier使弱候选15(tier5)优于近攻击54(tier4)，第1代实际normalbias范围[-.08,-.04289]，已离开近攻击候选+.08。原审计partial文件字节在slew50_sampling_audit_capture_1546中冻结，SHA相同。若下一完整代仍0full/无有用功率进展，应保存停止同一已验证guard，核验实际中断/回执，再新warm采样正方向附近；不直接resume旧fitness。

下一未执行输入slew50_near_attack_positive38_ready01_1545：policy SHA328b1a7d7059a06335585e19828ef63bcd7af2d001fa3d0ee39c2f692ac746c2，bank SHA64625626a87eb04e6f179145f543b4b7e1ea7fcb0d2d655c9c623e668e344d6d，软件率5保持。60bank=6normal[.02,.04,.06,.08,.10,.12]×5lead_delta[-.01,0,.01,.02,.03]×2gain[.98,1.02]；相对父c54实际lead[.01,.02,.03,.04,.05]、gain约1.0241/1.0659，其余身体恢复固定，mask[7,14,37]，建议sigma_fraction.04。120路290步CPU预检完成，506步补检另见slew50_near_attack_positive38_bank_cpu02_1550/REPORT.json（若尚未出现则仍在运行）。该方案尚未原生执行、不继承成绩；建议新run ball_slew50_near_attack_positive38_cem24。仍需完整同回合链、TRAIN128/扰动、正式A200+B200和旧短拍板动态对照，不能缩小目标。
