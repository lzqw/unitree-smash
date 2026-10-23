# 当前继续入口：拍面净空/朝向反馈开发比较（2026-10-23 15:49 UTC）

G07 IN_PROGRESS；正式400未运行，不能标PASSED。固定授权仅GPU0：UTC2026-10-23 08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），新增最多10 GPU-hours。历史保留，恢复不重置窗口。实际费用见GPU_LEDGER.json。

## 活跃作业：恢复先查，禁止重复启动

统一exec会话59925，reports/G07/arm_clearance_matrix_fine1536_01。guard PID1219217 / child1219225，启动15:46:30 UTC；max2700s/env-step-cap1305600。12冻结候选×128开发场景，physics62.5us/control.01s/episode8.5s，seed718951。输入arm_clearance_candidates_v1/manifest.json、原causal_profiled_dev_ground128_v1/panel.json与causal_dev_panel64_v1/calibration.json。源snapshot保留dirty。

候选0/1为上一批forward9/back2参考；2/3/4分别forward9加arm pitch输出bias-.01/-.025/-.04rad；5为back2加-.025；6/7为forward9当前公开heading/omega_z反馈gains[.6,.02]/[1.2,.04]；8/9为forward9/back2的heading[.8,.03]+pitch-.025；10为forward9加CPU vertical IK种子+pitch-.015；11为forward9降低共同vx gain到-.08加pitch-.015。详细实际weights/hash看manifest。

LearnedHybridPolicy新增arm_heading_reference/gains和arm_pitch_output_bias_rad、arm_joint_position_residual_rad；默认0保留旧行为。heading参考来自已有teacher214实测公开观测均值，不调用教师函数；当前heading只用公开gravity/command normal估计，joint1反馈最大.10rad，原执行器限制不变。vertical seed仅CPU受限IK，在名义姿态使拍心升约1.48cm，绝非动态保证。实际结果待本批。

## 刚完成的前向反馈比较

forward_body_matrix_fine1280_01 SUCCEEDED（启动15:25:55 UTC，已结束，不能重启）。10×128。候选9 forward15_back1标准完整45/64=.703125、AIR64/64、条件恢复60/64=.9375；B完整30/64=.46875、AIR62/64、条件恢复54/62=.8709677，Net3+8、该候选数值0。恢复达标但B完整仍不足.5。候选2 stop_back2cm标准完整46/64=.71875、恢复62/64=.96875；B完整28/64=.4375、AIR64/64、恢复54/64=.84375、Net2+5、数值0。原参考0的Net11+13；此提前停车+后移点大幅减少Net，但增加前向gain并非单调收益。全1280数值1来自candidate4，保留。

多数越网后落点远至x10m的失败伴有球/网等其他接触和很小中心面净空，不能简单称为横向瞄准错误。当前集中测试拍面俯仰增加净空，并测试heading补偿；原始阈值/场景不改。

基于3.0秒公开观测对共同前缀candidate2/3/4的分组CV，B完整最好29/64=.453125，没有超过常量candidate9；不部署为物理结果。旧初始观测选择器同样无改善。

## 最近结果

body_feedback_matrix_fine1280_01 SUCCEEDED；最好candidate8=preparation_pitch+0.35_rate-0.06.pt，SHA5abca6cb4dd47e9dd576e5f95f8fe15b75b81b215cd5ba3f11bb460323eaebfe。标准完整44/64=.6875、AIR64/64、恢复54/64=.84375；B完整29/64=.453125、AIR63/64、恢复47/63=.74603；该候选数值0，但Net7+13，多数早在walking2.6–3.0秒发生。全1280数值2来自其他候选，原数据保留。

original_walk_arm_body_matrix_fine1536_01 SUCCEEDED（已完成，不重启）；12×128。恢复原walking并保留arm/body并不能保留强扰动收益：最佳B完整16/64=.25；candidate6标准33/64、恢复61/64，但B完整16/64、恢复43/61。全部候选均未通过，两处候选数值故障合计4。故本批尝试更早停车、小幅后移准备点、只在推离时反馈前向速度。

frozen_candidate_matrix_fine768_01六候选也均未合格，最好preparation标准29/64、B23/64。旧adaptive_planner_profiled_fine128_01为24/64与10/64。均开发结果，不是正式400。

## 后续执行

1. 等当前1536结果，按候选分别看canonical/B。两组均full≥.5、AIR≥.7、条件恢复≥.85、所选候选数值0才进入正式。不能用混合候选平均或事后oracle冒充策略。
2. 如不足，分析失败时间和实际接触/恢复数据，再在剩余固定窗口内做有假设的改进；为最终400至少预留约40分钟。独立fresh开发面板causal_profiled_fresh_ground128_v2已在调参前冻结（launch718701/reset718702），条件选择器若学习旧面板必须真正在fresh面板复验。
3. 正式协议configs/g07_causal_profiled_net_eval_v2.json SHA6d11397ef63354217d63088d779fe449fe717ebeff10d3dbaf582b4b13c651a7，冻结A/B各200，原scenarios未改。lock01已弃用且从未跑；冻结一个新checkpoint和代码后用lock_g07_hybrid_evaluation.py生成lock02；同一策略依次A/B，不能中途换权重/源码。aggregate工具只重算，不自动发布PASSED。
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
