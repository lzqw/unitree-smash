# 当前继续入口：新侧高网体下128回合高精度复验（2026-10-23 13:21 UTC）

G07 IN_PROGRESS；正式400未执行，不可标PASSED。

## 当前活跃作业（先查，不能重复启动）

统一exec会话81037，reports/G07/adaptive_planner_profiled_fine128_01。GPU0预算guard，最多2100秒/108800环境控制步；62.5us物理、.01s控制、8.5s回合。冻结CEM最佳checkpoint为reports/G07/adaptive_planner_cem01/best_candidate.pt，SHAf61550e3351c18c038e51257dd3a174c6e68f78f45144c9af2002b217e0cc858。

128场景在reports/G07/causal_profiled_dev_ground128_v1/panel.json，SHAa45aa38c3948442d945b38834f5af42819ad35f1e5f0f12253a14c65d6dc0c8c。前64为原开发发球/标准地面起点，后64同来球/独立PCG64 seed717601的B宽度地面扰动。按前后64分别重算完整/恢复等，不能只看合计。新网体中心1.524、侧边1.55，version environment-local-profiled-net-v3。看stdout/error/metrics/completed及ledger；不要只凭app退出码0判成功。

固定授权GPU0独立新增10 GPU-hours；T0 UTC08:04:13.742590，截止18:04:13.742590（北京时间10月23日16:04至10月24日02:04）。重启不重置，历史用量保留并排除新上限。所有GPU作业经scripts/run_gpu_job.py --budget-window g07-net-feasibility-20261023，带max-seconds、env-steps-cap、新目录和completed.json回执。

## 网体补齐与正式协议

原net-v2为恒高1.524矩形网，虽然局部坐标/碰撞已正确，却未实现原ACCEPTANCE/Goal中的侧边1.55。旧成绩保留为矩形网课程，不作完整标准高度验收。新物理静态闭合三角网32段抛物线顶，中心1.524/侧1.55，原宽2/厚.02；未降低网高或改机器人。OrderedSmashEvents按同一分段顶边计算y处净空。profiled_net_native_control01原生20环境测量几何正确，16/16挡球/通行和4/4故意机器人碰网阳性通过；CPU tests/test_smash_events.py 16项通过。

新configs/g07_causal_profiled_net_eval_v2.json SHA6d11397ef63354217d63088d779fe449fe717ebeff10d3dbaf582b4b13c651a7，引用causal_profiled_eval_panel_a_v2和_b_v2各200。复制时逐项确认scenarios数组与旧预冻结A/B完全相同，仅网体/场景元数据另存。旧协议v1/旧panels/lock01保留未用；lock01因开发未合格、实现变更已弃用。正式门槛仍400/.5/.7/.85/0，取较差比例、数值累加。只有开发高精度合格后，调用scripts/lock_g07_hybrid_evaluation.py生成新不可变锁，使用同一checkpoint与实现依次正式A/B，不得中间改策略。aggregate_g07_hybrid_evaluation.py能重算，但不会自行发布PASSED。

## 下一步

1. 先等128开发结果；分标准/扰动两组。若均满足完整≥.5、air≥.7、rec≥.85、数值0，可准备同一新锁和正式400；尚未运行，不能先称通过。
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

仓库https://github.com/lzqw/unitree-smash，分支g04-guided-learning-20261022，最新已推送1476f01；新网体/原生阳性/128结果待后续提交。
发布https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023。完整视频G07_learned_hybrid_complete_nominal_20261023.mp4与清洁名义权重SHA0620b79d4b85fa746334d50122b1cef0cee933c8af179aa51b514395bbb41dc1为矩形网课程，不能代表新侧高网验收。原始旧视频/zip保留并分开标注。

Isaac解释器/home/autodrive/Isaac/env_isaaclab/bin/python；系统python3没有numpy/torch。CPU视频依赖PYTHONPATH=/tmp/unitree_video_deps、llvmpipe。所有inference缓存原生状态写入/forward/history reset须放torch.inference_mode；策略缓冲分配在其外，持久copy_。不要清理MPS/AWESun/其他作业，只管理本项目PID。
