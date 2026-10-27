# 短窗口挥拍的顺序原生试验

登记时间 2026-10-27T04:52:32.598658+00:00。CPU等待owner1966524已实际启动，`CUDA_VISIBLE_DEVICES`为空，无子进程，当前状态为等待已确认owner1957568（提前横向7组及独立冷128）。原GPU0 guard1951185/native1951201继续9组比较，没有新增并发原生作业。

970项输入SHA全部核对。原945/后958持有内容未变；本队列只增加独立报告目录中的代码，不修改它们正在使用的文件。等待还核对父进程命令和启动tick，父进程成功退出后独立复算原全部128例和实际来源权重，再逐项重放实际来源的6例公共轨迹。任何失败保留原记录，无自动重试。来源达到原开发50/70/85时跳过本次搜索，正式评测仍待执行。

通过来源审核后，依次执行7×原全部128例搜索（716800控制环境步、最长18000秒），选中实际最好方案，再执行独立冷128（102400步、最长5400秒）。全部使用原Main GPU0 guard和账本、原地面初始化和原生episode编号。来源为 `early_lateral7_n128_01/best_candidate.pt`，来源冷评估为 `early_lateral7_best_cold128_01`；新输出分别为 `windowed_swing7_n128_01` 和 `windowed_swing7_best_cold128_01`。这些未来输出尚不存在，尚无窗口化原生成绩。

[算法与实际CPU检查](../windowed_swing_20261027/REPORT.md)、[计划及970项持有输入](PLAN.json)、[独立入口检查](PRE_NATIVE_CPU_CHECK.json)、[实际进程和进展](CURRENT_NATIVE_AND_PROGRESS.json)可查。生成器保留原生采样、日志、碰撞和计分循环；正式面板未读，物理/目标/阈值保持。G07未完成。

旧入口 `tools/verify_goal.py --root . --goal G07` 实际返回2：缺少 `reports/G07/evidence.json`。完整输出保存在 `LEGACY_G07_VERIFIER.json`，没有生成通过证据或修改验收器；旧G00–G06记录也不代表新版完整球拍的正式验收。
