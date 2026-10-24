视频已实际发布（2026-10-24 06:49 UTC）：Current_effect_return_recovery_20261024_0640.mp4，31.56s/1578frames、全解码通过，main相同回合合法air strings→球触网后越网→对方场内首落→稳定3.62s；攻击及指定目标失败，配对TRAIN65失败，full仍0。视频SHA50a7e6d60728618f28f2d4e9d02a2134619d72b67b2219edb95d620dd37d106a；50,530,699B证据zip SHA3533b95276a2748bdbf157edb5abc79a107c4d6cacc6fdf1ab40bfb018910937，5个release资产均核SHA/大小，见video/Current_effect_return_recovery_20261024_0640_upload_receipt.json。当前仍guard1350322/child1350331/worker1349683/exec74286，gen1全程正在收尾，不重复启动。不改live GPU_LEDGER/RUNS或preexisting RESOURCE_STOP。

实测链条进展（2026-10-24 06:47 UTC）：CEM02第0代64次已完整结束，16contacts/7airstrings/11qualifiedrecoveries/1airstrings+recovery/4task-valid crosses/0target/0complete/NF0。candidate6/env12/TRAIN1同一次qualifiedjump：takeoff2.531、air字符串2.772453、落足2.94997、shuttle真实越网2.930692（球有Net接触但未blocked）、首次落在对方场内[3.004436,-1.066558,.025156] at3.897022，最终稳定3.619920s、bodyground0、robotfault0。出球[4.636972,-1.316661,-.250078]、速4.826764；攻击8m/s/vz<=-1及原目标|y|<=1均未过，横向miss.066558m。配对TRAIN65/bodyground2.88s/没触球。真正完整仍0，正式400未跑，不能称G07通过。

本候选是第一个实测“合法空中回球+同回合qualified恢复”，但两案例min/mean全局optimizer第0代winner为candidate1。单例进展已单独保存return_recovery_gen0_env12/{policy_evidence.pt,native_episode.npz,contact_event.json,episode.json,MILESTONE.json}，candidate6参数严格等于population_000[6]，并非拼接不同回合。contact_generation0_snapshot01完整不可变64回合raw/权重/source；contact_generation0_launch_snapshot01有原最初RNG/模板。原all-lane gen0 causal log未及时保存，现被gen1覆盖，此缺口明确保留，不把重算的public forecast冒称原env12日志。当前实际gen1 causal初始日志已另存contact_generation1_causal_snapshot01.json。未来logger已补每代causal_plans_XXX，当前进程不会加载这个未来改动。

新32秒同回合全程视频Current_effect_return_recovery_20261024_0640.mp4已CPU渲染/1578帧全部解码（50fps/1280x720），含真实初次落点/target边界/paired failure，正随证据上传；未发布前查upload receipt，不要先声称上传成功。

下一轮warm选择：return_recovery_warm07/policy.pt精确保留candidate6原arm/leg参数与seed，旧fitness只作initialization；return_recovery_fixed_leg08另一个未跑提案仅把leg_event_time_gain置0并按保存lane0 TRAIN1公共观测推算event2.727414234转换p2为1.336531401（参考clock误差<1us），其他arm不改。原两TRAIN预测差2.97068ms被从腿时钟去掉，这不是证明两case差异由该时差导致。fixed08带同seed0/映射后p2的RECOVERY_LEG_BANK.json，不能给它套old seed1 bank。两种warm都需要新native fitness；当前active guard1350322/child1350331/worker1349683/exec74286继续gen1，不重复启动。当前worker只跑这一job，结束后不会自己启动下一job。

下一阶段依当前gen1结果选择：若攻击点速仍由root后仰主导为负，从真正joined-return/recovery候选而非仅worst-score winner启动--pitch-grid（参数gain.4/.7/1/1.3×armlead-.03/0/.03；52CPU已过，尚未native）。可用64candidates×TRAIN[1,65]×2gen（128env，当前64env仅约3825/16303MiB、GPU50%，只是本机当前占用参考），guard5400s/204800envsteps cap；或32candidates保持既有规模。仅GPU0，先确认旧guard与worker确实terminal及receipt；无需旧总预算/期限。若出现真正强出球/完整，应优先扩大TRAIN128与原扰动再冻结A/B200。

速度诊断与后续方法（2026-10-24 06:32 UTC）：当前唯一GPU仍1350322/1350331，CEM02按已归档51测试版本运行，不是pitch-grid。首代300ticks出现16次初次接触（7初拍线/9拍框，皆己方初始合法；腾空/恢复最终以各episode记录为准），400ticks49首回合结束，其中6空中拍线，0完整；env12拍线首回合仍在8s恢复观察，别提前当成功。velocity_feedback_cpu01冻结实际contact JSON与population参数，7个原生拍线点的URDF重建误差<4um：env20手臂相对normal+2.523，root旋转-3.483、平移-.018，合成-.978m/s。固定实测body运动的bounded CPU求解可给出normal约+1.19，但新全身反作用/执行器跟踪/球路未验证，绝非原生成功或全局可行性证明。

已实现下一步可选--pitch-grid（仍contact-curriculum/continue-after-miss，原26参数不扩展）：12组现有p15公共gravity俯仰补偿gain[.4,.7,1,1.3]×arm lead偏移[-.03,0,.03]，只变14/15，其余保留，可再加原8recovery-leg交叉。52CPU检查通过，当前p15仅.181。等当前CEM02两代实际结果后，若仍普遍world点速向己方，可从真实最佳contact/recovery参数fresh warm-start ball_pitch_compensation_cem03（32×TRAIN[1,65]×2gen/sigma.025/5400s/102400cap），不要把待跑方法说已运行；若已出现真正完整/强出球，应优先扩大TRAIN而不是机械重跑旧试验。

当前运行后磁盘代码新增了未来pitch-grid采样选项，当前actor/物理无变化，但search/helper源码SHA与已启动run不同；要恢复当前optimizer必须还原contact_generation0_launch_snapshot01/source_snapshot.zip中的精确源，或下一轮warm-start重测fitness，不能绕过source检查。该snapshot保存了本代最初generation_start.pt/RNG/模板/原source归档。还需在gen0完成后保存当前causal_plans.partial.json为本代专属副本，避免gen1覆写。

当前唯一GPU已切换（2026-10-24 06:17 UTC）：ball_contact_recovery_cem02 guard1350322/child1350331；顺序worker1349683仍live，exec74286。state=contact_recovery_queue01/state.json已RUNNING_GUARDED_JOB；不要再按下方旧WAITING段手工启动。旧upper_strings guard1344215/child1344227已实际退出，jobSUCCEEDED，128次0hit/0recover/0complete/NF0，实际77120步/.464637646GPUh；最后存活env3亦未满足qualified recovery。FINAL_TRAIN_SUMMARY.json与128行episodes.csv已写，不改历史raw。旧ledger由guard自动封账，新guard独占资源锁；勿手改GPU_LEDGER/RUNS。

新run32候选×TRAIN[1,65]×2gen，原8s/62.5us/10ms，sigma.025、contact-curriculum、12arm lead/speed网格、8recovery-leg交叉、10高斯、push范围.30；5400s/102400步cap。warm06精确原recovery gen1 candidate31，legbank原candidate12两个漏球qualified恢复，全部新fitness重测。原生配置/10USD审核和generation_start已生成，当前源验证51CPU已过。先读本段实际PID与queue state；code源/driver加载后固定，无需因观察超时重开。

CPU attack_window_cpu01/PLAN.json和INTERPRETATION.json只规划，不是原生成功/不可行证明：原实际触球比公共观测预测的x1.167参考状态晚26.68ms；实测法向点速-1.2466，理想早窗回球原法向约[.996,.027,-.079]需+1.029m/s。模型出球[6.9,0,-4.1]可越网落x3.307，但跟踪、关节/反作用/全拍合法性必须原生验证，不改变门槛。

接续方法更新（2026-10-24 06:12 UTC）：同一contact_recovery_queue01的新worker1349683 / exec74286已实际WAITING；保留当前GPU作业，等待其真实SUCCEEDED退出。32候选含2参考/均值、12只改arm时机/速度、8个实测恢复腿参数交叉、10高斯。--recovery-leg-bank contact_recovery_warm06/RECOVERY_LEG_BANK.json，来自原gen1 candidate12；与原contact31严格相同seed/family，按.25/.5/.75/1混合body参数，只再变arm_lead[-.03,0]，其余arm/球拦截参数保留。51项CPU测试通过，尚无新模式原生结果。其余调度、源hash冻结、单GPU/cap约束见下文；不可手工再启动重复任务。

最高优先调度更新（2026-10-24 06:09 UTC）：已有串行接续 worker1349683 / exec74286，state=reports/G07_full_racket_v4/contact_recovery_queue01/state.json。它等待旧GPU guard1344215和child1344227都实际退出且job.json SUCCEEDED，再自动启动 ball_contact_recovery_cem02（warm06原candidate31，32×2TRAIN×2gen、sigma.025、contact-curriculum+contact-grid）。不要手工再启动该任务；先查worker/旧guard/新guard实际PID及state。worker启动前冻结了quadsmash全源码+search/guard/protocol/warm SHA，排队期间改这些源码会明确拒绝启动。每个GPU任务仍独立5400s/102400步cap。worker不写GPU_LEDGER/RUNS，原guard退出后由新guard独占资源锁。若用户要求停止训练，还要处理这个本项目worker及其子guard，避免停止后又接续。旧正文的“手工接续命令”已由本自动队列取代。

# 唯一活跃GPU：ball_upper_strings_cem01，真实上部拍线与恢复联合训练

继续前必须读 /home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。完整目标不缩减：真实完整球拍/标准悬空网/己方实际合法空中拍线+原攻击>=8m/s、vx>0、vz<=-1+真实越网首次目标落点+同回合无bodyground稳定.8s，之后原A/B各200 worst .5/.7/.85/NF0、真正成功/失败侧视俯视视频、旧新能力比较与全部Github交付。当前完整成功仍0，正式400未跑，绝不能标完成或说不可行。仅GPU0无累计GPUh/总工作窗/截止，有限单job保护、全部历史失败/费用保留。

当前同一exec3642，guard1344215/child1344227，ball_upper_strings_cem01；32候选×TRAIN[1,65]=64环境×2代，62.5us物理/10ms控制，每回合原8s，单job5400s/102400控制-env步cap。先ps/poll同一handle、progress/error/job.json，观察超时不是退出，不重复启动。GPU_LEDGER/RUNS由活跃guard持有，勿手改。旧recovery_cem01/exec53083/guard1334795已SUCCEEDED退出，不再当活跃。

当前控制：--warm-start upper_strings_warm05/policy.pt --ik-seeds ik_recorded_upper_strings05/seeds.json --continue-after-miss --push-search-range .30。warm取旧gen1 candidate12，两TRAIN均实际qualified jump+稳定3.9559/3.7869s、无bodyground，但都漏球，不是成功回球。用该成功恢复姿态的实际GPU snapshot重新拟合上部拍线uvz+.09，静态位置误差.02224m、速度投影[2.10577,-.30825,-.64999]m/s，仅初始化不是实际击球速度。旧04是失败恢复来源的草稿，已改用来源更可靠的05。新arm会改变基座反作用，所有fitness从头重测。

seed05保留794D公开观测/因果ball forecast，nominal projected pitch取该TRAIN姿态，leg_event_time_gain0固定腿时钟；warm p2由1.3965557按源case1事件2.7209535转换成1.3675091，保持源case1原腿时间线而不是无故移后。半挥窗.08，新的strike residual清零，armlead仍原实测初值。argsIK清armoffset/armgain1，真正root始终地面.323起始且floating，实际关节/速度/力矩限幅保持。原outgoing8m/s等成功门槛未变。

当前课程v3：允许TRAIN在漏球、shuttle-only反向/挡网失败后继续观察同一跳跃恢复，rally始终dead/fault，后续弹跳不能变成功；robot-rule/bodyground/fallen仍提前结束。正式FullRacketEnv/FrozenPanelEnv始终原8s/原判定，不用TRAIN子类。45项CPU测试包含不复活回合、混合robot故障不能被shuttle故障掩盖。新记录将保存全部lane全8s50Hz姿态/球/clock/first-valid，以及exact contact/terminal states；每代local episode id从0重新开始，避免旧logger全局id混淆。新记录路径all_first_episodes_000.npz等。旧recoveryjob没有这些新字段，不冒称补录其未保存轨迹。

刚完成recovery_cem01两代128次候选首回合（含elite重测）：13次qualified jump recoveries全部来自漏球回合、4接触中1空中拍线、0完整/NF0。实际2095.264856s/.582018016GPUh/102400步。gen0 3recoveries，gen1 10recoveries；gen1 candidate1/12/24两案例均完成body恢复，只有1组的最低稳定.803s，不能称泛化强。env62/episode2实际合法空中拍线高1.702071m，out[4.824756,-.361456,-1.049205]、速4.950732，后真实shuttle触网挡回，body恢复因旧v2早停未验证；不是攻击或完整成功。FINAL_TRAIN_SUMMARY.json/128行episodes.csv/全部raw/权重/精确source保留。旧gen0 raw contacts中有自动reset后episode1事件，不混进候选首回合统计。

近网判定新修复已实际验证：02组near_net_prehit_invasion/own_contact/empty_corner三个nonground/fixed-root单控制步原生案例全部SUCCEEDED/NF0；NATIVE_PLANE_REGRESSION_0549.json。真正提前越网在第一个62.5us子步抓到Frame_02；合法接触点x1.491005在己方而球心x1.515918已在对方，实际strings/cross_net且无fault；空AABB角max1.526224但真实primitive support1.492，正确不判越网。只目标回归，不能称policy或完整裁判验收。Full-return逻辑CPU正反测试通过；后续实际合法扣杀仍须检查动态随挥及全回己方。

01旧native scene-query分支的提前越网/调试两次FAILED原样保留（费用计入）。原因为DirectGPU CPU scene-query姿态旧，wide query有actor而live侵入volume不命中，与NVIDIA官方DirectGPU API限制说明一致；不能恢复该错误分支当当前判定。现基于引擎当前GPU body pose计算声明USD96 primitive的精确box/capsule交叠与有限cylinder凸QP，1um显式plane数值容差，不加力、不改asset/质量/碰撞材质/网/球/执行器/门槛，也不开无效CPU scene-query支持。若数值几何求解未决会硬报错而非默认合法。新原生三个02通过后才启动当前实际训练。

后续按当前native contact/attack/cross/firsttarget/recovery实际结果持续优化，不停在弱触球或CPU几何；必要时任务空间/角速度反馈、反作用协调、轨迹优化/残差学习。扩大TRAIN128/原扰动，冻结真正最佳源码权重（含此前原生核对的10层USD依赖），原A/B200独立评估，真正完整成功/失败全身视频和旧新比较。原r3 c2264fce...资产675mm整拍/451.8mm法兰至拍线中心/153g拍夹具/20.709991kg总重及全协议hash保持。

最新已发布实际触球视频28.94秒：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4
已发布gen0不可变snapshot与gen1部分1.702m触球证据和程序，均核Github大小SHA。完整两代/本次02修复和新warm05正在同步；活跃upper_strings任务不能当完成数据。旧上下文NEXT_before_upper_strings_0549.txt及更早NEXT。不得改preexisting status/G07_RESOURCE_STOP.json，不停止无关MPS/GPU任务。

2026-10-24 06:03 UTC续进：最新30秒原生高位触球/独立恢复视频及42文件证据已上传核SHA，branch2de9f3d。upper_strings gen0已完成64次（405ticks/25920steps），0contact/0recover/0complete/NF0，46bodyground、13robotnet、5invasion；gen1仍在同一guard运行，不因观察超时重开。UPPER_STRINGS_GEN0_DIAGNOSIS_0601.json。新seed改变耦合反作用；下一轮已准备contact_recovery_warm06/policy.pt，精确取原recovery gen1 candidate31实际1.702m接触的原seed/26参数，未继承fitness。新可选contact-curriculum+contact-grid优先保留实际合法空中拍线，观察挡网后的同跳恢复，12组只变arm lead/speed、不动腿时钟；全50项CPU测试通过，但新模式未原生运行。当前guard退出后直接run_gpu_job有限5400s/102400步，32candidates×[1,65]×2gens、sigma.025、push_search_range.30、continue-after-miss；不加IK seed覆盖。原完整成功门槛/正式环境不改。源candidate31的TRAIN65原回合2.99s bodyground，不能称两case成功。
