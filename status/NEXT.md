# 最新继续入口：学习混合控制器已完成名义课程，正在测因果覆盖（2026-10-23）

G07 IN_PROGRESS；正式400回合还未执行。G04–G06、旧权重/证据保持不变。

## 固定资源窗口

GPU0；独立新增最多10 GPU-hours；窗口g07-net-feasibility-20261023：UTC2026-10-23 08:04:13.742590至18:04:13.742590，Asia/Taipei10月23日16:04至10月24日02:04。重启、场景/方法切换不重置。历史用量保留并排除本轮上限，实耗看GPU_LEDGER。

## 正在运行，不能重复启动

统一exec会话31917，`reports/G07/adaptive_planner_cem01`。GPU0固定授权窗口不变。train_g07_adaptive_planner.py，最多6代，16组规划参数×16个固定开发场景（causal_dev_panel64_v1索引0,4,...60），每代256环境、8.5s、开发物理步长.00025；按真实有序完整结果优化。源码只把场景初态用于reset，LearnedHybridPolicy只读公开观测与模型预测；不读取未来计划标签。每100控制步有进度，先查error/metrics/stdout，不能重复启动。

学习参数11维：阶段时序修正[常数、到达时间、入球vx、预测高度]，准备x修正[常数、时间、入球vx、高度、y]，准备y偏移与比例。预测x平面修正后重新求球到达时刻；实际原关节/执行器限制不变。新class可接收每环境候选矩阵供训练；最终checkpoint只存一组学得系数。首轮均值来自开发接触平面漂移回归，保留原始零修正候选。训练早停条件为同组16开发球完整≥.875、恢复≥.875、数值0，早停/代数上限均有明确完成回执；SIGTERM中断只保存当前代，未完成请求则无回执。

最近已完成：causal_hybrid_dev64_02（62.5微秒）64变化来球AIR1.0、恢复.890625、完整.40625、Net1、数值0，未达完整.5；名义scenario0成功。有效拍面53/64，有序越网37、目标27。不能启动正式400或标PASSED。

adaptive_planner_fast_dev64_01（.25ms开发筛选）AIR1、恢复.84375、完整.453125、Net10、数值0。步长不同不能直接称较前提升；仍未达条件。该方法作为CEM初值，不作正式策略。

CEM输入adaptive_planner_regression01/policy.pt，panel/calibration来自causal_dev_panel64_v1。中途最佳与每代population保存mean/sigma/RNG，可用--resume-population恢复到新目录；所有作业经run_gpu_job预算guard，窗口不重置。

## 可信进展与失败

- 原球网世界/局部重复平移已修复；net_blocking_probe03/verification.json 32/32真正挡球。旧教师53/64、学习54/64物理网成功声明撤回，不可混入新成绩。
- 原分组Net过滤传感器零值无效；现逐刚体传感器通过4/4阳性对照。NetContactMonitor版本one-source-per-environment-v2。
- 无借力有效击球教师214冻结单球重放16/16有效面/越网/首次目标，但恢复0。
- 从实测数据学得12段PWA，CPU留出最大动作误差4.17e-6；没有把此误差当仿真成绩。学习Go2 actor用于行走和站立，5个位置适配参数+接入时间搜索，原硬件限幅不变。
- learned_hybrid_recovery_scan01：64不同恢复参数，完整17，非固定策略成功率。
- hybrid_frozen_compare01：4固定策略各32，完整32/14/10/32，Net0，组2扫掠未决1。原数据未删，阈值未放宽。
- hybrid_fine625_compare01：两固定策略各32，较高精度完整11/32与32/32；前者Net9，后者Net0；全64数值0。最佳后者与旧组3同参数，跨两分辨率均32/32完整。
- clean frozen_candidate1_policy.pt已删除未用的44参数教师；仅PWA、冻结NN适配参数和模型常量。新评估脚本可直接加载。旧probe脚本需要旧schema，不用于加载该clean文件。

## 下一步

先查当前64覆盖：名义scenario0应与已有路线对应；若失败先区分新入口/时序、横向准备、击球、落地。phase_start已由公开球状态预测为3.23–3.87秒，y目标约±.1，尚非成绩。只用独立开发64做方法调整，不修改已冻结200+200面板。

预先固定协议configs/g07_causal_long_prepare_eval_v1.json，输入causal_eval_panel_a_v1与_b_v1各200，原A/B初始化seed100100/200200与原扰动宽度，新增发球seed717301/717401，时刻4.60–5.25、vx六种、拦截y±.1/z±.03。固定落点矩形x[2,4],y[-1,1]。源角色是版本化长准备场景，不是旧7.4s课程或真实赛场认证。

原status/g07_causal_formal_lock01.json保留，但因开发未达标未用于正式执行、实现已变而失效，原因见同名outcome。开发通过后新建版本锁定同一策略/实现/协议哈希，使用新evaluate脚本--protocol/--policy-lock对两200面板运行；指标取最差比例，累计数值故障，阈值仍400/.50/.70/.85/0。不能两面板间换策略。尚未启用可用正式lock、尚未运行正式400、未写artifacts/G07/policy.pt。

全部GPU作业通过run_gpu_job.py，指定上述budget-window、max-seconds、env-steps-cap和completed.json回执。运行中job.json未必存在，当前PID看GPU_LEDGER。主控制器/观测缓冲要持久copy_，不要在inference_mode内替换后又在外部reset；Isaac状态写入与forward需在torch.inference_mode内。

## GitHub与视频

https://github.com/lzqw/unitree-smash （分支g04-guided-learning-20261022），最新已推送c75813d；此后自适应规划学习及变化发球结果待本次推送。
发布页https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。
完整学习名义课程视频G07_learned_hybrid_complete_nominal_20261023.mp4已上传，是0.25ms组3的32/32片段，不是400验收。
当前无恢复教师视频G07_netlocal_unassisted_contact_20261023.mp4与先前错误网/旧PWA失败视频均留存、分开标注。二进制快照及教师数据/PWA见同页不可变zip；源Go2 NN来自固定公开作者归档，manifest给URL/sha，禁止替换成私有rough策略。

## 下一批开发复验（已冻结，未运行）

causal_dev_ground_panel128_v1/panel.json：64变化来球标准起点+同64来球独立B宽度地面扰动，PCG64 717601，SHA91d84a0d6015f5a13bcf626c140f0e46265fc111be6c7fcbfa75c2eae84f3347。CEM结束后先用最佳策略在62.5us测128，按两组分别统计。正式面板保持不变。evaluate脚本已修正diagnostic trace在首次done后停止保存；不改变控制/事件判据，新正式锁会覆盖代码哈希。
