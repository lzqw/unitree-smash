# 当前唯一GPU：ball_cem_low_base_branch01（G07仍未通过）

仍须读原目标55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt，保持GPU0无限累计授权、有限每job保护。目标未缩减：实际合法空中拍线触球/攻击/越网首次目标落点/同回合恢复，最终A/B各200的50%/70%/85%/NF0及实际视频/全部交付；尚无完整成功或正式400，绝不标完成。

活跃exec session33164，guard PID1327511/child1327523，started_at2026-10-24T04:27:43.830995+00:00。上限3600s/25600控制-env步；8候选×TRAIN[1,65]×2代，最多8s、62.5us/10ms；训练专用不可挽回失败早停，正式FullRacketEnv/FrozenPanelEnv仍完整8秒。先查同一PID/progress/error/job.json或poll33164；观察超时不是终止，不重启。旧backoff01/session81054已经FAILED终态退出，绝不能当活跃。

旧backoff01完成首代32案例：0触球、9netfault、16bodyground、7首次落地漏球，最小拍线间隙0.2263m。第二代wrapper.reset报inference tensor原地修改异常，未实际运行第二代物理。真实390.437030s/0.108454731GPUh，已知控制-env步9984，账本51200是无完成凭据的保守上限，不能当实际执行51200。FAILURE_AND_FIX.json/error/stdout/全部第一代population/轨迹保留，不能将执行失败或32无触球说成通过。

程序修复：rally_rules_v4.py原self.quiet_frames=torch.where在inference_mode内换成了新inference tensor，后续模式外reset不能写；改成预分配buffer.copy_(where(...))，计数公式/物理/规则/阈值均未变。新增跨inference rollout→模式外reset两次的生命周期测试，33项CPU测试通过；新任务需实际验证第二代reset，不把CPU测试冒充原生证明。更改源码后不直接resume旧fitness，当前用best_candidate warm-start重测。

当前假设：原几何排序选q1≈-2.55导致整个机械臂大幅回转，后退基准2.7s身体gravityX≈.552/headZ1.252、ballZ1.885，依然太低。已从同一32次IK的另一解支选q1≈-.615、q5≈+.883、q6≈2.094，几何误差约.035m却可能更适合全身运动；不是成功保证。新种子ik_flat_low_base_rotation01取原始candidate15，数据来源hash/选择条件已保存，未改变球拍/质量/执行器。用backoff首代最佳参数初始化腿，arm offset清零/速度gain1、新种子时序校准0为初始假设；走位x.64/速度.58。实际反馈仍只用794D公开观测，训练案例仍[1,65]，不能从正式A/B删难例。

reset_eval_native02独立PhysX view核对8环境扰动已经PASSED：q/qd/root xyz误差0，quat1.788e-7；13.686993s/.003801943GPUh/48控制-env步。只初始化验证，正式400仍TODO。先前Ball smoke01真实8全触网/0触球/NF0、948.449185s/.263458107GPUh、6400步也保留。完整256无球搜索88安全/8次安全跳跃恢复（含精英重测）只作初始化，不当扣杀成绩。

后续必须连续看新种子真实训练→按命中/间隙/姿态/法向/速度/恢复迭代，必要时更灵活的全身轨迹优化或残差学习，不能停在审计或只采无球。正式前核查近网整拍代理/随挥回己方边界，冻结完整来源和权重，再A/B200独立和实际成功/失败侧视俯视/触点首次落点视频、旧新能力比较。r3资产c2264fce...和全部train128/A200/B200来球哈希/物理参数/性能门槛保持，不能用失败采样或静态IK宣称不可行。GPU_LEDGER由活跃guard独占写，不手改。

已上传38.12秒r3视频、24.17MB轨迹证据、12.28MB完成256/reset01归档，以及7.96MB首个Ball8失败+reset02独立核对包，全部核Github大小SHA；README/release有链接。当前失败记录和早停/修复代码正在本分支同步；旧资料全部保留。以下原NEXT为04:17时历史，当前PID以本页顶部为准。

# G07 r3：真实来球后退走位联合训练正在执行

必须读目标附件55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。仅GPU0，无累计GPUh、总工作窗、截止；有限每job保护，保留一切历史费用失败。大目标仍是实际合法空中拍线触球、攻击出球、越网首次目标落点、同回合稳定恢复，再冻结代码权重做A/B各200（50%/70%/85%/NF0），交付成功/失败实际视频与旧新比较。当前未达成，不标完成，不改资产/硬件/来球/门槛来过关。

活跃唯一GPU：ball_cem_backoff01，exec session81054、guard PID1325839/child1325851，start2026-10-24T04:17:15.227177+00:00，上限3600s/51200控制-env步；16候选×2个TRAIN案例[1,65]×2代，每回合最多8s、62.5us物理/10ms控制。已确认同一PID存活、generation0/tick100/1个首回合提前结束（尚未进入2s来球阶段）。先poll同一session或ps/progress/error/job.json；勿因观察超时重复启动。旧sequence session30733已终态成功，旧Ball guard1323147/child1323156已退出，不再当活跃。

上一个ball_cem_smoke01真实8回合执行完整800ticks、6400控制-env步，948.449185s/0.263458107GPUh。guard SUCCEEDED仅执行完成：8个全部在触球前robot_net_fault，0触球/0完整/NF0，没有一个有效策略。FAILURE_ANALYSIS.json有逐案CSV和50Hz诊断：约2.34s根x.850、head origin x1.467/z1.180，拍框mesh逼近网前面x1.498（采样max1.497954）；不是精确子步碰撞点，原生触网flag为准。root前倾扫网，不能把之后的物理回弹/动作当回合成功，也不能据此宣称方案不可行。

本次修改：同一r3资产/球网/质量/执行器/球参数/来球和规则，把走位目标x退到.58、速度.52，继续用ik_flat_face_fast01正面拍线中心几何种子，重新测实际fitness，不继承失败分数。新增RallyTrainingEnvV4只用于TRAIN：触网等rally fault、首次落地未命中目标、body-ground或fallen时结束已经不可能完整成功的回合；每案仍计失败且保存原因。发生合法触球并首次目标落地的回合继续至原8s以核查恢复。正式FrozenPanelEnv/FullRacketEnv仍原8s，绝不在正式评测用此早停类。早停和新日志刚开始原生执行，尚无训练改善结果。

reset_eval_native02已经SUCCEEDED：8个TRAIN输入/6ticks，直接PhysX get_root_transforms/get_dof_positions/get_dof_velocities独立核对扰动；q/qd/xyz误差0，quat最大1.788e-7；公开观测和局部重置隔离通过。13.686993s/0.003801943GPUh/48控制-env步。仅初始化烟测，不是400策略通过。之前01 writer/data/public烟测原始证据保留。

上一无球reach_near_net_backface02已完整4×64（含精英重测）256次，88次虚拟己方安全，8次还跳跃恢复；1311.131635s/.364203232GPUh/153600控制-env步。最佳虚拟球心[1.379143,.039516,1.741971]、点速[1.744774,-.809993,-.219007]，腾空.503997s、稳定1.912033s、bodyground0；仍不能算真实扣杀。r3资产c2264fce...、675mm整拍/452mm法兰至拍线中心/153g拍+夹具/总20.709991kg；13.4×6.1m球场、宽6.1深.76悬空网，中1.524侧1.55；r1漏碰/r2穿腕及旧失败均保留。冻结configs/g07_full_racket_eval_v4_r3.json与train128/A200/B200哈希从未改变。

SOURCE注意：首个Ball smoke用其source_snapshot版本运行；之后增添了中途partial轨迹/每案最近距离/first_contacts/active_first_rallies、所有接触的episode/event编号。这次backoff用新版本。其optimizer resume会严格检查源码hash，换源码/接触/目标要warm-start重测。不要用CPU测试或新编译代码冒充已原生覆盖。新训练checkpoint附训练早停契约，formal代码源锁独立执行。

剩余必须继续：看backoff实际球的触球/最近距离/预备生存/触网时间→改准备、全身时机、法向、接触位置和恢复并真实运行；必要时用已有CPU正面上部uvz.09种子ik_flat_upper_fast01（未执行），不是放大拍面；不能只无球或只文档。正式前验证近网整拍边界/随挥“全回己方后再过网”（当前racket包围点代理需原生边界样例），冻结源码/权重，再正式A/B200、视频和旧新能力对照。不能把少量失败/静态IK当不可行证明，不能删难例。

已上传38.12s r3实际效果视频、24.17MB轨迹证据和12.28MB新增源/256执行/权重/reset01归档，都核对Github大小SHA；README和release有链接。本次Ball失败与reset02将在后续同步，不要冒称已在先前归档中。Git分支g07-full-racket-court-v4-20261024。原NEXT已复制NEXT_before_backoff_0417.txt，完整旧模型来源/规则/早期失败/账本等仍在MODEL_RULES和前版报告。GPU_LEDGER正由新guard独占写，勿手改。

最近后退批次partial至generation0/tick300：18/32首回合结束，0触球，14仍in_play，最近拍线距离min0.2263m（基准约0.442/0.541m）。基准没有上一批的早触网，但2.7s身体gravityX约.552、headZ1.252而ballZ1.885；2.8s前倾更大，说明不能只退站位。已保留新的未执行CPU替代分支ik_flat_low_base_rotation01：来自同一原始32次IK结果，q1≈-.615/q5≈+.883，避免当前几何最优q1≈-2.55的整个机械臂大幅回转。只是初始化假设，需实际原生试验；当前backoff两代任务仍运行，不在观察超时时重启。
