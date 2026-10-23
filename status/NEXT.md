# 当前继续入口：六冻结候选高精度开发复验（2026-10-23 14:13 UTC）

G07 IN_PROGRESS；正式400未执行，不可标PASSED。不要重复启动已停止的训练。

## 活跃作业（先查，不能重复启动）

统一exec会话7801，reports/G07/frozen_candidate_matrix_fine768_01。GPU0 guard max2700s/env_steps_cap652800，62.5us物理/.01s控制/8.5s回合，6冻结控制器各128开发场景，共768（不是正式400）。输入manifest reports/G07/frozen_development_candidates_v1/manifest.json，顺序baseline、baseline_replan、preparation、preparation_replan、local、local_replan。所有文件已冻结，不能在运行中改权重。Panel为causal_profiled_dev_ground128_v1，calibration仍causal_dev_panel64_v1/calibration.json，seed718201。

新评估入口 --candidate-checkpoint 可追加开发候选；有--protocol或--policy-lock时强制拒绝多候选。CandidateBatch给每个控制器独立NN历史/公开观测，只按固定随机排列的env映射分发，不给控制器场景标签。raw rows有candidate_id/source_scenario/development_group，结束后candidate_metrics.json逐候选、canonical/perturbed_b分组，不能用全768的合计当任何一个策略成绩。每候选都必须AIR≥.7、full≥.5、rec≥.85、fault0，两组都满足才考虑锁定正式400。

候选来源：baseline来自adaptive_planner_cem01/best_candidate.pt（11规划系数f615...）；preparation来自已停止preparation_arm_cem01/best_candidate.pt（粗步长32训练17/32完整，恢复32/32、Net0、fault0；非独立成绩）；local来自local_preparation_arm_cem01/best_candidate.pt（125us32训练10/32）。每个另有replan_at_s=3.0的独立冻结版本，仅从当时公开球状态重新预测到达时刻并调整phase/arm amplitude，既不改准备位置也不改球/机器人状态。CPU一致性检查通过，尚无物理改善结论。新方法的实际重规划日志保存causal_replans.json。

大步幅preparation checkpoint没有readiness_extension字段，按True保留其原语义：heading准备停止条件且walk最迟4.5秒。local checkpoint明确preparation_readiness_extension=False，保留旧准备停止规则与3.4秒截止，只给行走actor小幅yaw命令；不能忽略该字段混用实现。原checkpoint未启用replan保持旧路径。

追踪视频：manifest.recorded_env_ids给每个候选原始source case0/1的随机全局env编号，traces_fixed_scenarios_0_1.json按该列表顺序，link_poses_fixed_scenario_GLOBALID.npy按全局编号保存。prepare_g07_causal_video.py --scenario GLOBALID --geometry-reference reports/G07/compact_native_contact01 可提取；会按candidate_id过滤完整统计，核对资产SHA/body_names。CPU渲染仍需PYTHONPATH=/tmp/unitree_video_deps、Isaac解释器。首次done后不继续录重置。

本批保存initial_public_observation.npy和完整初始状态；preparation_snapshots.json有起跳前多时刻状态。未决扫掠精确输入有专门JSON，原判据/容差未改。

固定授权GPU0独立新增10 GPU-hours；T0 UTC08:04:13.742590，截止18:04:13.742590（北京时间10月23日16:04至10月24日02:04）。重启不重置，历史用量保留并排除新上限。所有GPU作业经scripts/run_gpu_job.py --budget-window g07-net-feasibility-20261023，带max-seconds、env-steps-cap、新目录和completed.json回执。

## 近期停止的两个训练（已保存，不能重复启动）

- 旧exec48903 preparation_arm_cem01，.25ms、16×32、原请求6代，实际完成2代后按方法切换SIGTERM保存退出；最佳第1代group8训练17/32、rec1、Net0、fault0。guard FAILED仅因无原6代完成回执，原因单存intentional_stop_reason.json。全部population000/001与raw保留。
- 旧exec22969 local_preparation_arm_cem01，.125ms、12×32、原请求4代，完成2代最佳10/32，未改善。已验证best与population001可加载后停止未完成的下一代采样，guard exit-9/FAILED如实保留，不能称完成4代或程序崩溃。最优仍generation0/group5（delay比原少.05），checkpoint有readiness_extension=False。
- candidate_matrix_native_smoke01：4个.2秒短回合已SUCCEEDED，仅验证开发路由/地面reset/保存，全部无接触、完整0，不冒充任务成绩。CPU CandidateBatch与独立控制器混合路由逐动作差0；仅计算检查。

## 最新失败（已完成，不能重复运行）

旧exec81037 adaptive_planner_profiled_fine128_01 SUCCEEDED，用冻结f615...在net-v3真实高精度128：标准64 AIR1、full.375、rec.921875、Net0、fault0；B扰动64 AIR.9375、full.15625、rec43/60=.7166667、Net1、fault0。合计full34/128、rec102/124、fault0。development_group_metrics.json已重算。正式门槛未满足，不启动400。

新原生网体诊断profiled_net_native_control01 16/16球挡/通行正确、4/4故意机器人碰网阳性；原恒高net-v2课程不替代新侧高网验收。

## 网体补齐与正式协议

原net-v2为恒高1.524矩形网，虽然局部坐标/碰撞已正确，却未实现原ACCEPTANCE/Goal中的侧边1.55。旧成绩保留为矩形网课程，不作完整标准高度验收。新物理静态闭合三角网32段抛物线顶，中心1.524/侧1.55，原宽2/厚.02；未降低网高或改机器人。OrderedSmashEvents按同一分段顶边计算y处净空。profiled_net_native_control01原生20环境测量几何正确，16/16挡球/通行和4/4故意机器人碰网阳性通过；CPU tests/test_smash_events.py 16项通过。

新configs/g07_causal_profiled_net_eval_v2.json SHA6d11397ef63354217d63088d779fe449fe717ebeff10d3dbaf582b4b13c651a7，引用causal_profiled_eval_panel_a_v2和_b_v2各200。复制时逐项确认scenarios数组与旧预冻结A/B完全相同，仅网体/场景元数据另存。旧协议v1/旧panels/lock01保留未用；lock01因开发未合格、实现变更已弃用。正式门槛仍400/.5/.7/.85/0，取较差比例、数值累加。只有开发高精度合格后，调用scripts/lock_g07_hybrid_evaluation.py生成新不可变锁，使用同一checkpoint与实现依次正式A/B，不得中间改策略。aggregate_g07_hybrid_evaluation.py能重算，但不会自行发布PASSED。

## 下一步

1. 先等当前准备/残差CEM，再用其冻结best到新目录独立62.5us128复验；分标准/扰动两组。若均满足完整≥.5、air≥.7、rec≥.85、数值0，可准备同一新锁和正式400；尚未运行，不能先称通过。
2. 若失败，备用机械臂姿态残差已实现但未实际运行：scripts/train_g07_adaptive_planner.py --arm-pitch-basis reports/G07/arm_pitch_residual_basis_cpu01/metrics.json，使用新profiled开发面板的标准64子集（需单独冻结，不要忽略robot_reset；trainer遇reset会拒绝）。4维残差依赖公开预测速度/高度，原关节限幅，不调用教师动作。CPU基向量只是种子，真实球CEM后还要细步长复验。
3. 当前策略仅初始公开球状态做模型预测；若需要在线重规划，应单独版本化实现与测试，不能读取未来标签。固定calibration JSON未改；学习系数存在checkpoint。

## 保留的关键结论

- G04-G06原证据/PASSED未改；原硬件资产SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3、质量20.646991kg及执行器限制不变。
- 旧物理网世界/局部平移错导致旧53/64教师、54/64学习结论撤回；net_blocking_probe03/verification32/32真正挡球。分组接触过滤零值曾无效，现one-source-per-environment-v2经阳性验证。
- 无借力教师214冻结单球16/16真空中面/越网/首次目标、恢复0。PWA由实际数据拟合12段，留出最大动作误差4.17e-6；CPU误差不代替仿真。
- 学习Go2行走/站立actor+PWA击球，teacher_calls0。矩形网可靠冻结参数在.25ms/62.5us各32/32完整/无Net/数值0，仍同一球课程。另一个候选高精度仅11/32且Net9；全部失败保留。
- causal_hybrid_dev64_02矩形网高精度变化发球：AIR1、rec.890625、full.40625、Net1、数值0。adaptive_planner_fast_dev64_01粗步长：AIR1、rec.84375、full.453125、Net10、数值0，步长不同不可直接称提升。
- adaptive_planner_cem01完成6代16组×16开发场景真实球搜索，best第5代group8 full.625/rec.9375/Net0/fault0，训练而非泛化成绩。checkpoint与population_000..005.pt保存了参数/均值/方差/RNG。
- 细步长和原大场景仍可能出现sweep_unresolved；阈值/审计没有放宽，出现则保留并诊断。

## 入口与上传

仓库https://github.com/lzqw/unitree-smash，分支g04-guided-learning-20261022，最新已推送720f45d（包含128失败、准备学习入口）；后续矩阵/重规划源代码与训练证据待本次提交。
发布https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023。完整视频G07_learned_hybrid_complete_nominal_20261023.mp4与清洁名义权重SHA0620b79d4b85fa746334d50122b1cef0cee933c8af179aa51b514395bbb41dc1为矩形网课程，不能代表新侧高网验收。原始旧视频/zip保留并分开标注。

Isaac解释器/home/autodrive/Isaac/env_isaaclab/bin/python；系统python3没有numpy/torch。CPU视频依赖PYTHONPATH=/tmp/unitree_video_deps、llvmpipe。所有inference缓存原生状态写入/forward/history reset须放torch.inference_mode；策略缓冲分配在其外，持久copy_。不要清理MPS/AWESun/其他作业，只管理本项目PID。
