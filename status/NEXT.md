# 当前继续入口：准备位置与高位挥拍组合（2026-10-23 16:19 UTC）

G07 IN_PROGRESS；正式400未运行，不能标PASSED。固定授权仅GPU0：UTC2026-10-23 08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），新增最多10 GPU-hours。历史保留，恢复不重置窗口。实际费用见GPU_LEDGER.json。

## 活跃作业：恢复先查，禁止重复启动

统一exec会话75206，reports/G07/vertical_preparation_matrix_fine1536_01，启动16:10:05 UTC。12候选×128，max2400s/envcap1305600，62.5us/.01s/8.5s，seed719051。候选vertical_preparation_candidates_v1/manifest.json，开发panel和calibration仍原文件；本批显式--sweep-audit-iterations4096，容差1e-6不变，原查询/碰撞响应不变，源快照保留dirty。恢复先查ledger/PID，勿重复启动。

候选0为原back2参考；1..4为back2+vertical seed1倍+pitch-.015、共同vx gain0/-.05/-.10/-.15；5/6为vertical种子.5/1.5倍、vx-.10；7/8为back3、vertical1、vx-.05/-.10；9为back2+vertical1+vx-.10+heading[.6,.02]；10/11同back2+vertical1+vx-.10，pitch bias0/-.03。实际参数和SHA见manifest。0仍旧差分body模式，其余v2共同vx模式，参数选择器目前不混此类模式。

## 刚完成：拍面/heading比较

arm_clearance_matrix_fine1536_01 SUCCEEDED，12×128；没有固定候选两组同时合格。candidate10（forward9+verticalIK种子+pitch-.015）B完整32/64=.5、AIR61/64、恢复48/61=.786885、Net13；标准41/64、恢复51/64、Net12。candidate1 back2参考标准46/64、rec63/64、Net0；B完整29/64、AIR62/64、rec52/62=.83871、Net4。candidate9 back2+heading+pitch-.025：标准38/64、rec59/64、Net3；B完整24/64、rec55/62=.8871、Net3。全1536数值3（candidate0标准1、candidate5B2），各失败保留。

参数选择器CV未过：forward共同前缀0,2,3,4,6,7,8,10,11，full-state ridge1/width8标准54/64、B31/64且Brec51/61=.8361；back2前缀1/5/9 ridge1/width128标准51/64、B31/64且Brec54/62=.87097。精简body-ball与body-ball-arm特征也未达标，所以尚未把selector送GPU。CPU实现存在，但不是策略成绩。

碰网时序必须按每回合contact_time对齐，不以绝对5秒就称击球后。contact_recovery_cpu01/net_contact_order_diagnostic.json：c10有25Net，其中16在球触球前、8之后、1无有限接触时刻；c0有7前/9后。许多相差几十毫秒。public_clearance_braking_cpu01发现大多数新候选Net已非walking阶段，不能只修准备制动。

LearnedHybridPolicy新增可选contact_recovery：只读取当时public773（真实飞行后触球phase）锁存观测时刻，按arm_blend_s/arm_delay_s收臂至既有recovery_arm_target；可选leg_delay_s提前既有standingactor。不读取未来真实接触时刻。默认关闭，对当前活跃作业无影响。CPU异步flag/从未触球lane/复位/legs保持一致均通过（contact_recovery_cpu01/metrics.json）；物理尚未运行，且仅能解决击球后Net。拟备用结合更高起跳等方法，不能宣称已改善。

## 刚完成的前向反馈比较

forward_body_matrix_fine1280_01 SUCCEEDED（启动15:25:55 UTC，已结束，不能重启）。10×128。候选9 forward15_back1标准完整45/64=.703125、AIR64/64、条件恢复60/64=.9375；B完整30/64=.46875、AIR62/64、条件恢复54/62=.8709677，Net3+8、该候选数值0。恢复达标但B完整仍不足.5。候选2 stop_back2cm标准完整46/64=.71875、恢复62/64=.96875；B完整28/64=.4375、AIR64/64、恢复54/64=.84375、Net2+5、数值0。原参考0的Net11+13；此提前停车+后移点大幅减少Net，但增加前向gain并非单调收益。全1280数值1来自candidate4，保留。

多数越网后落点远至x10m的失败伴有球/网等其他接触和很小中心面净空，不能简单称为横向瞄准错误。当前集中测试拍面俯仰增加净空，并测试heading补偿；原始阈值/场景不改。

基于3.0秒公开观测对共同前缀candidate2/3/4的分组CV，B完整最好29/64=.453125，没有超过常量candidate9；不部署为物理结果。旧初始观测选择器同样无改善。

## 最近结果

body_feedback_matrix_fine1280_01 SUCCEEDED；最好candidate8=preparation_pitch+0.35_rate-0.06.pt，SHA5abca6cb4dd47e9dd576e5f95f8fe15b75b81b215cd5ba3f11bb460323eaebfe。标准完整44/64=.6875、AIR64/64、恢复54/64=.84375；B完整29/64=.453125、AIR63/64、恢复47/63=.74603；该候选数值0，但Net7+13，多数早在walking2.6–3.0秒发生。全1280数值2来自其他候选，原数据保留。

original_walk_arm_body_matrix_fine1536_01 SUCCEEDED（已完成，不重启）；12×128。恢复原walking并保留arm/body并不能保留强扰动收益：最佳B完整16/64=.25；candidate6标准33/64、恢复61/64，但B完整16/64、恢复43/61。全部候选均未通过，两处候选数值故障合计4。故本批尝试更早停车、小幅后移准备点、只在推离时反馈前向速度。

frozen_candidate_matrix_fine768_01六候选也均未合格，最好preparation标准29/64、B23/64。旧adaptive_planner_profiled_fine128_01为24/64与10/64。均开发结果，不是正式400。

## 后续执行

1. 等当前vertical1536结果，按候选分别看canonical/B。两组均full≥.5、AIR≥.7、条件恢复≥.85、所选候选数值0才进入正式。不能用混合候选平均或事后oracle冒充策略。
2. 如不足，分析失败时间和实际接触/恢复数据，再在剩余固定窗口内做有假设的改进；为最终400至少预留约40分钟。独立fresh开发面板causal_profiled_fresh_ground128_v2已在调参前冻结（launch718701/reset718702），条件选择器若学习旧面板必须真正在fresh面板复验。
3. 正式协议configs/g07_causal_profiled_net_eval_v3.json SHA6ce5c9929aa163504b39c24c5e85f1f2e939d43dde0d10beb48b55dc77a5319c，冻结A/B各200，原scenarios未改。lock01已弃用且从未跑；冻结一个新checkpoint和代码后用lock_g07_hybrid_evaluation.py生成lock02；同一策略依次A/B，不能中途换权重/源码。aggregate工具只重算，不自动发布PASSED。
4. 若当前模型数值阻塞：runtime sweep仍原256迭代/tol1e-6。CPU sweep_iteration_diagnostic_cpu01显示3条原unknown在1024/4096同float32/tol下收敛miss；这不是原数值0、未改运行审计。如实现更大迭代只用于unknown，需要版本记录、针对性正例/原失败验证，原故障永不回写。
5. UTC18:04:13前保存、停止本项目进程、准确记录费用与BUDGET_EXHAUSTED（若仍未通过）。不能因预算到期把goal标complete。

## 场景和证据边界

资产20.646991kg、SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3，原力矩/速度/PD不变。网v3中心1.524/侧1.55、宽2/厚.02，32段抛物线顶；仍是ground-to-top实体研究障碍，并非真实织网/场地标定。长准备球初位可数十米，属于已版本化模型场景。各网版本成绩不混。

assets/assumptions.md是G01签名证据，必须保持SHA b78c443cb89de38d73f8b730fc7fba7e72b28c4e8f2dd2ffca4db70835a26b4a；不要再编辑。G07新增场景说明在reports/G07/ASSET_ASSUMPTIONS.md。b48bd1c已恢复原文；structural_pending_check_20261023_1510.json显示G00–G06依赖全部通过，G07仅缺自身evidence.json。

## 上传和视频

仓库https://github.com/lzqw/unitree-smash，分支g04-guided-learning-20261022，已推送0051129。release g07-progress-20261023。

新网真实视频G07_profiled_net_current_20261023.mp4已上传，SHA a80225bbc268e425d7ff4af852fa5668233730b615764db4cc3f7cf0f75cd4ee，来源768matrix candidate2/source0/globalenv566，固定录制案例实际完整；画面统计为该候选全128 full52/recovery98，不是400。

开发机身反馈权重G07_body_feedback_development_20261023.pt/json已上传；SHA5abca...，minimum source dac845b，建议最新分支。其B/恢复未合格，不能宣称通过。

64MB历史增量G07_profiled_development_5d19f95_1415UTC.zip，SHA358dbe2ba5fb81d02bc5b6e1a445fcbbbae03c8b883c99fd35b3529c7bcbc39e；不含后续body实验，且快照有G01说明文档签名不符。配套G07_documentation_integrity_fix_b48bd1c.zip已上传且digest核对，SHA b7eaacceefd3904b2a49f689a594514320bdc8f76e223c364e27b1f6e7afccec，或用最新分支。历史包不覆盖。

Isaac解释器/home/autodrive/Isaac/env_isaaclab/bin/python；system python3无torch/numpy。CPU视频依赖PYTHONPATH=/tmp/unitree_video_deps和llvmpipe。恢复先检查ps/ledger防重复，勿清理MPS/AWESun/无关进程。原G04–G06保持PASSED；不把搜索失败当硬件不可行证明。

最新附件G07_body_forward_development_0051129.zip/json已上传（94,208,112bytes；最终SHA66745a5e8bd557c733399d4f54a4b872c8d1b58c86724864a5e8b5682379eb98，已核对GitHub digest）；完整两批body/原walking对照，包含G01说明修复，不含forward1280后续结果。最新成功+失败视频G07_body_feedback_success_and_failure_20261023.mp4已上传，41.82秒，SHA7ae3c0a1b606a8c4c685dc19e024da04ac45f681913c4dbbe7cdda24a8450351；body8预先录制case0/1均保留。

## 备用参数选择入口（代码已实现，物理未运行）

scripts/fit_g07_causal_selector.py只接受已SUCCEEDED且completed的开发原始数据，严格验证候选主干除允许的body_feedback_parameters/arm pitch bias/heading gains/ref/joint residual以外完全一致，并验证checkpoint与源manifest SHA。当前arm pool可共同前缀候选0,2,3,4,6,7,8,10,11；1/5/9的waypoint不同，不能混进同一后段选择器。

LearnedHybridPolicy在声明3.0秒只用当时public61:67(log)、0:45、760:766做RBF得分选择，不输入caseID/后验标签；每lane持久保存参数，最早控制变化phase+.75（body）或+.8（arm），旧主干/行走/规划/恢复固定。selector schema g07-public-rbf-parameter-selector-v1，参数表C×body3、pitch1、heading2、joint6。Fitter使用各候选自己的真实3秒context与实际后果，不平均不同origin状态。

CPU报告在causal_parameter_selector_cpu01：关闭selector时12当前候选与GPU启动时源码动作差0；以旧body8/9拟合只作plumbing check，128混合选择105/23与独立实例动作差0，reset清除状态，score float32/float64最大误差5.91e-5。当前GPU任务是在这一扩展前启动，实际执行startup snapshot；扩展默认关闭的兼容性已核对。

若当前arm1536有一个固定候选两组均达标，直接冻结单一checkpoint做fresh128或正式400，不必引入selector。若固定未过但同前缀CV明显达标，可先按diagnose_g07_initial_selection.py --context-time 3 --candidate-ids 0 2 3 4 6 7 8 10 11分组CV选ridge/width；再fit脚本生成新checkpoint、在fresh128实际测试，合格后再正式400。不能用CV或oracle当物理验收。原forward2/3/4存在body mode/vx bias差异，fitter会拒绝，不能直接混成当前固定mode参数表。

## 同查询数值计算预算改进

sweep_compute_cpu02和sweep_compute_gpu01均通过10条固定查询：6个解析正负/切向/平移/转动/四元数双覆盖，4条既有未收敛查询。保持原函数、几何、1e-6容差，256有4 unknown、4096全收敛，CUDAfloat32与CPUfloat64结果一致；没有新PhysX回合或力响应。GPU查询费用.0005059478h已记账。BallEnvCfg默认256保留旧行为，G07评测可显式4096，manifest记录；unknown仍为硬数值失败，历史256失败不改。

正式新协议v3与v2的两个panel文件/哈希、400门槛、资产/初始态契约、物理步、目标和网体完全相同，仅审计最大计算迭代4096；新SHA6ce5c992...。原v2保留未正式运行。新协议仍须新lock02同策略/源码两组200，不能面板间更改。

fresh_ground128_v2/validation_plan_v1.json已在读arm1536结果前固定seed719001、4096审计/原容差/62.5us/8.5s。只有开发固定候选或分组CV两组均达原门槛才值得此独立物理复验，fresh与正式400当前仍未用。

可选vertical_push_calf_delta_rad/window已实现在LearnedHybridPolicy，默认0无效果；窗[.98,1.04,1.16,1.24]，小腿目标+delta、大腿目标-.5delta，原关节范围/执行器不变。CPU关闭时与当前启动source动作差0，原生高度收益未知。当前活跃vertical1536不含此字段。若该批仍失败，可考虑下一批将best组合与小幅推离补偿及实际触球后收拍联合测试；先看本批结果，不把CPU种子当动力学成绩。
