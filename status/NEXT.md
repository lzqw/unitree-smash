当前唯一GPU0作业：train128_lateral_c12_1025，guard1544539 / child1544553，session19517。先查真实进程与completed/job，再读metrics；不得重复启动。

CEM32已SUCCEEDED：candidate12在8个不同训练来球中合法空中拍线4次、接触后恢复3次、攻击1次、目标2次、完整0、NF0。全128执行是16候选x8来球，不是128不同案例。固定权重为reports/G07_full_racket_v4/cem32_lateral_feedback_1015/best_candidate.pt。待当前TRAIN128诊断验证后再决定是否开展扰动适配；不要仅凭8个来球重跑形式A/B。

当前固定A/B400结果仍为candidate5未通过（A空中24、恢复10；B空中6、恢复1；完整0/NF0）。新视频：Current_formal_evaluation_cem31_20261025.mp4；完整轨迹包：G07_CEM31_formal400_native_trajectories_20261025.zip，GitHub Release均下载回验成功。

下一种未执行的备选：基于当前公开来球预测选择己方拦截平面，使击球高度接近实际可达高度，而非所有来球固定X；这仅是方法提案，未有代码/性能证据。

接续队列已启动：reports/G07_full_racket_v4/height_native_queue_1050/run.py，PID1546233，session61744，状态文件reports/G07_full_racket_v4/height_native_queue_1050/state.json。当前等待TRAIN128实际退出和成功收据；若NF>0或已有策略达到TRAIN门槛则不启动高度搜索，源码/权重变化也会跳过。不要手工重复启动 CEM33。

已实现高度选择（不是原生性能证据）：298项测试通过；800步回放默认关闭/零调整与bce4748逐位一致，开启时137步臂命令及27步腿命令改变。初始化 reports/G07_full_racket_v4/height_intercept_ready_1045/policy.pt 明确native_execution=false。新策略目标高度1.76m、平面范围[.8,1.45]m、33平面、调整比例.5；原始来球/网高/完整拍/动力学/成功门槛全部保留。
