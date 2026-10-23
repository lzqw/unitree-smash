# 最新继续入口：学习混合控制器已完成名义课程，正在测因果覆盖（2026-10-23）

G07 IN_PROGRESS；正式400回合还未执行。G04–G06、旧权重/证据保持不变。

## 固定资源窗口

GPU0；独立新增最多10 GPU-hours；窗口g07-net-feasibility-20261023：UTC2026-10-23 08:04:13.742590至18:04:13.742590，Asia/Taipei10月23日16:04至10月24日02:04。重启、场景/方法切换不重置。历史用量保留并排除本轮上限，实耗看GPU_LEDGER。

## 正在运行，不能重复启动

统一exec会话37762，`reports/G07/causal_hybrid_dev64_02`。命令在GPU_LEDGER当前RUNNING项：evaluate_g07_causal_hybrid.py，64种已固定来球，62.5微秒、8.5秒回合、学习控制器+初始公开观测预测。stdout每50控制步报告进度，error.txt如存在须先检查。01因初始化在inference_mode外写Isaac缓存张量，6秒退出、未开始物理回合，失败和成本保留；02已修复并运行。

该作业输入：hybrid_fine625_compare01/frozen_candidate1_policy.pt（SHA0620b79d4b85fa746334d50122b1cef0cee933c8af179aa51b514395bbb41dc1）、causal_dev_panel64_v1/panel.json（SHA05fbd73a6c7c6f0a37867576a132706394397bf5bbf127381ac90048a472af59）与对应calibration.json。规划只读公开观测，不读每场景计划时刻/目标标签。控制器实现LearnedHybridPolicy不导入教师动作函数。源NN和资产哈希均锁定。

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

开发通过后冻结同一策略/实现/协议哈希到policy-lock，使用新evaluate脚本--protocol/--policy-lock对两200面板运行；指标取最差比例，累计数值故障，阈值仍400/.50/.70/.85/0。不能两面板间换策略。尚未创建正式lock、尚未运行正式400、未写artifacts/G07/policy.pt。

全部GPU作业通过run_gpu_job.py，指定上述budget-window、max-seconds、env-steps-cap和completed.json回执。运行中job.json未必存在，当前PID看GPU_LEDGER。主控制器/观测缓冲要持久copy_，不要在inference_mode内替换后又在外部reset；Isaac状态写入与forward需在torch.inference_mode内。

## GitHub与视频

https://github.com/lzqw/unitree-smash （分支g04-guided-learning-20261022），最新已推送884e154；此后新覆盖入口/协议与结果待本次推送。
发布页https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。
完整学习名义课程视频G07_learned_hybrid_complete_nominal_20261023.mp4已上传，是0.25ms组3的32/32片段，不是400验收。
当前无恢复教师视频G07_netlocal_unassisted_contact_20261023.mp4与先前错误网/旧PWA失败视频均留存、分开标注。二进制快照及教师数据/PWA见同页不可变zip；源Go2 NN来自固定公开作者归档，manifest给URL/sha，禁止替换成私有rough策略。
