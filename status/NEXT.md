# 当前继续入口：起跳pitch/rate反馈原生复验（2026-10-23 14:50 UTC）

G07 IN_PROGRESS；正式400未执行，不可标PASSED。固定窗口UTC08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），GPU0独立新增10 GPU-hours，不重置历史/窗口。

## 当前活跃作业（不能重复启动）

统一exec会话58404，reports/G07/body_feedback_matrix_fine1280_01。guard max2700s/env-step-cap1088000，62.5us/.01s/8.5s，10冻结候选各128开发场景。输入body_feedback_candidates_v1/manifest.json，顺序baseline_reference及4个kp±.35/kd±.06反馈，再preparation_reference及4个相同反馈。Body参考由已完成net_free_teacher214_replay01/teacher_dataset.pt的公开pitch、omega_y、vx均值构成；只按固定相对时钟插值参考，用当前公开状态差调整前/后大腿位置命令，.75–1.30s窗口、最大.12rad。关节限位/增益/力矩/速度/质量未改，没有外力/状态重置/教师动作函数。CPU零增益动作差0、gyro扰动导致前后腿相反修正已查。真实效果待candidate_metrics.json。

输出逐candidate和canonical/perturbed_b两组；不把1280总数当某个策略成绩。每组必须full≥.5、air≥.7、rec≥.85且总fault0，才可能冻结一个策略跑正式A/B。实际Net失败保留分母，完整和条件恢复均排除有机器人碰网的回合。

## 最近已完成、上传和诊断

- exec7801 frozen_candidate_matrix_fine768_01已SUCCEEDED，6×128高精度。六组标准/B完整率依次.390625/.140625、.328125/.140625、.453125/.359375、.453125/.234375、.453125/.171875、.453125/.15625；无候选达标。全768故障1（candidate0/source112 sweep_unresolved），其他组0不代表全量0。
- complementarity.json是事后oracle诊断，不能部署：标准最多46/64、B最多31/64，连B50%都不足，故没有训练/发布候选查表来冒充成功。
- 准备快照校正：baseline在3.0到3.8s rootx中位漂移<1mm，B3.8s std .0102m；当前同组接触rootx std .0636m（旧独立128测.0682）。差异主要在起跳过程放大，所以新增上述腿部pitch/gyro反馈，而非直接更改准备位置。
- preparation候选的14个Net中多数首次2.6–3.0s已发生（有Racket/Head），后段反馈不能消除已经发生的碰网。备用组件池original_walk_arm_feedback_candidates_v2/manifest.json已准备但未执行，共12项，去掉不稳的新步态，保留原walking，与后来学得arm residual组合，加/不加body反馈及3.0s重规划。该v2池所有候选3.0s前控制相同，若当前10候选不合格，优先测试此组件拆分。
- 新开发场景causal_profiled_fresh_ground128_v2/panel.json已在后续学习/评测前冻结：发球seed718701、B扰动seed718702，64新球各canonical/B。若学习条件参数，应在这批新场景实际独立评估；原正式A/B仍未使用。

## 后续可执行方向（未实现的不当成绩）

1. 等当前10候选结果。若一个冻结候选两组均合格，保持同一实现/文件，用新lock02和原profiled_net_eval_v2协议正式各200。只用同一策略，不能面板间改权重。
2. 若不合格，原行走＋新arm/body组件12池已冻结，可用evaluate_g07_causal_hybrid.py的--candidate-checkpoint在原128开发面板实际比较。此池所有早期动作相同；global3.0s时尚未开始arm/body残差。
3. 只有原walking共同前缀内的候选oracle明显超过门槛，才值得学习一个3.0s时基于当前公开状态的参数选择器。可以从preparation_snapshots的public_frame69/public_command_and_clock17和初始化预测特征学习选body gains/arm coefficients（不是按caseID，不读未来标签运行）；候选选择若用已完成结果只能当oracle，不能叫策略。此参数选择器目前未实现。要在fresh128实际评测，再考虑正式400。

## 视频与发布（已上传）

新网体v3真实效果：https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/G07_profiled_net_current_20261023.mp4 ，sha a80225bbc268e425d7ff4af852fa5668233730b615764db4cc3f7cf0f75cd4ee。来源frozen_candidate_matrix_fine768_01/candidate2/source0，固定录制全局env566，实际完整/无Net。画面统计仅该候选全部128开发回合full52/recovery98；不是formal400。
新开发归档G07_profiled_development_5d19f95_1415UTC.zip已上传同一release（64,097,559bytes、sha358dbe2ba5fb81d02bc5b6e1a445fcbbbae03c8b883c99fd35b3529c7bcbc39e），source/checkpoint/population/publicGo2actor，增量包依赖先前基础资产；GitHub返回digest一致。当前1280活跃结果不在该归档中。

视频工具prepare_g07_causal_video.py --scenario GLOBALENV读取manifest.recorded_env_ids（各候选case0/1预先固定随机位置），按candidate过滤统计，核对asset/body_names。renderer使用PYTHONPATH=/tmp/unitree_video_deps和Isaac解释器/llvmpipe CPU。后续本批候选也能生成真实视频，不需额外GPU录像。

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
