# 当前继续入口：前向起跳反馈开发比较（2026-10-23 15:31 UTC）

G07 IN_PROGRESS；正式400未运行，不能标PASSED。固定授权仅GPU0：UTC2026-10-23 08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），新增最多10 GPU-hours。历史保留，恢复不重置窗口。实际费用见GPU_LEDGER.json。

## 活跃作业：恢复先查，禁止重复启动

reports/G07/forward_body_matrix_fine1280_01，guard PID1214658/child1214666（启动15:25:55 UTC）；max2700s/env-step-cap1088000。10候选各128开发场景，physics62.5us/control.01s/episode8.5s，seed718901。输入forward_body_candidates_v1/manifest.json、causal_profiled_dev_ground128_v1/panel.json、causal_dev_panel64_v1/calibration.json。当前代码未提交部分为LearnedHybridPolicy新增前后腿共同vx反馈与walking waypoint小幅后移；guard保留启动source_snapshot/dirty。

候选顺序reference,stop_original,stop_back2cm,forward10,forward20,forward20_back3,extended_back3_forward,original_walk_forward15,original_walk_forward25,forward15_back1。新增mode differential-pitch-common-vx-v2：pitch/omega仍前后腿差分，vx共同大腿修正只在push1.00–1.25相对秒，单关节最大.12rad；不写根速度、无外力、不改硬件。CPU kv=0与旧模式动作差0，正参考速度产生同向前后腿命令，仅计算检查。实际成效等本批candidate_metrics.json。

## 最近结果

body_feedback_matrix_fine1280_01 SUCCEEDED；最好candidate8=preparation_pitch+0.35_rate-0.06.pt，SHA5abca6cb4dd47e9dd576e5f95f8fe15b75b81b215cd5ba3f11bb460323eaebfe。标准完整44/64=.6875、AIR64/64、恢复54/64=.84375；B完整29/64=.453125、AIR63/64、恢复47/63=.74603；该候选数值0，但Net7+13，多数早在walking2.6–3.0秒发生。全1280数值2来自其他候选，原数据保留。

original_walk_arm_body_matrix_fine1536_01 SUCCEEDED（已完成，不重启）；12×128。恢复原walking并保留arm/body并不能保留强扰动收益：最佳B完整16/64=.25；candidate6标准33/64、恢复61/64，但B完整16/64、恢复43/61。全部候选均未通过，两处候选数值故障合计4。故本批尝试更早停车、小幅后移准备点、只在推离时反馈前向速度。

frozen_candidate_matrix_fine768_01六候选也均未合格，最好preparation标准29/64、B23/64。旧adaptive_planner_profiled_fine128_01为24/64与10/64。均开发结果，不是正式400。

## 后续执行

1. 等当前1280结果，按候选分别看canonical/B。两组均full≥.5、AIR≥.7、条件恢复≥.85、所选候选数值0才进入正式。不能用混合候选平均或事后oracle冒充策略。
2. 如不足，分析失败时间和实际接触/恢复数据，再在剩余固定窗口内做有假设的改进；为最终400至少预留约40分钟。独立fresh开发面板causal_profiled_fresh_ground128_v2已在调参前冻结（launch718701/reset718702），条件选择器若学习旧面板必须真正在fresh面板复验。
3. 正式协议configs/g07_causal_profiled_net_eval_v2.json SHA6d11397ef63354217d63088d779fe449fe717ebeff10d3dbaf582b4b13c651a7，冻结A/B各200，原scenarios未改。lock01已弃用且从未跑；冻结一个新checkpoint和代码后用lock_g07_hybrid_evaluation.py生成lock02；同一策略依次A/B，不能中途换权重/源码。aggregate工具只重算，不自动发布PASSED。
4. 若当前模型数值阻塞：runtime sweep仍原256迭代/tol1e-6。CPU sweep_iteration_diagnostic_cpu01显示3条原unknown在1024/4096同float32/tol下收敛miss；这不是原数值0、未改运行审计。如实现更大迭代只用于unknown，需要版本记录、针对性正例/原失败验证，原故障永不回写。
5. UTC18:04:13前保存、停止本项目进程、准确记录费用与BUDGET_EXHAUSTED（若仍未通过）。不能因预算到期把goal标complete。

## 场景和证据边界

资产20.646991kg、SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3，原力矩/速度/PD不变。网v3中心1.524/侧1.55、宽2/厚.02，32段抛物线顶；仍是ground-to-top实体研究障碍，并非真实织网/场地标定。长准备球初位可数十米，属于已版本化模型场景。各网版本成绩不混。

assets/assumptions.md是G01签名证据，必须保持SHA b78c443cb89de38d73f8b730fc7fba7e72b28c4e8f2dd2ffca4db70835a26b4a；不要再编辑。G07新增场景说明在reports/G07/ASSET_ASSUMPTIONS.md。b48bd1c已恢复原文；structural_pending_check_20261023_1510.json显示G00–G06依赖全部通过，G07仅缺自身evidence.json。

## 上传和视频

仓库https://github.com/lzqw/unitree-smash，分支g04-guided-learning-20261022，已推送b48bd1c。release g07-progress-20261023。

新网真实视频G07_profiled_net_current_20261023.mp4已上传，SHA a80225bbc268e425d7ff4af852fa5668233730b615764db4cc3f7cf0f75cd4ee，来源768matrix candidate2/source0/globalenv566，固定录制案例实际完整；画面统计为该候选全128 full52/recovery98，不是400。

开发机身反馈权重G07_body_feedback_development_20261023.pt/json已上传；SHA5abca...，minimum source dac845b，建议最新分支。其B/恢复未合格，不能宣称通过。

64MB历史增量G07_profiled_development_5d19f95_1415UTC.zip，SHA358dbe2ba5fb81d02bc5b6e1a445fcbbbae03c8b883c99fd35b3529c7bcbc39e；不含后续body实验，且快照有G01说明文档签名不符。配套G07_documentation_integrity_fix_b48bd1c.zip已上传且digest核对，SHA b7eaacceefd3904b2a49f689a594514320bdc8f76e223c364e27b1f6e7afccec，或用最新分支。历史包不覆盖。

Isaac解释器/home/autodrive/Isaac/env_isaaclab/bin/python；system python3无torch/numpy。CPU视频依赖PYTHONPATH=/tmp/unitree_video_deps和llvmpipe。恢复先检查ps/ledger防重复，勿清理MPS/AWESun/无关进程。原G04–G06保持PASSED；不把搜索失败当硬件不可行证明。
