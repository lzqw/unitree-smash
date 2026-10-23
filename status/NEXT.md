# G07固定工作窗结束：BUDGET_EXHAUSTED

北京时间2026-10-24 02:04原截止已到，GPU0本项目作业全部停止。实际本轮结算7.645612263 GPUh，原10GPUh上限尚余2.354387737；不能因仍有GPUh余额越过固定时间窗。历史/T0/截止均未重置。用户“随时GPU”是否取消原截止的异步问题仍待答复；未获得明确修订前不要启动GPU。

当前Goal没有完成；没有G07 evidence.json/artifacts/G07/policy.pt，禁止goal complete。G00–G06结构/证据仍通过。若同一资源限制连续3个goal turn确实阻塞，才可goal blocked；本停止turn记录在G07_RESOURCE_STOP.json，不要直接假定已满足3次。

## 已完成的真实结果

- 正式lock02全部400失败：Afull100/200、AIR200/200、rec178/200；Bfull71/200、AIR196/200、rec146/196；数值0。原始/聚合/失败锁均保留。
- contingency推离/yaw1536已SUCCEEDED、NF0，无新候选合格。
- landing_velocity_noisy1536已SUCCEEDED、NF0，无新候选合格。参考0full44/64、27/64，rec61/64、48/63；新3full40/64、24/64，rec54/64、45/64，低于参考。等落地/延迟站立普遍明显退化。
- 最后独立B16：SUCCEEDED。全部16回合完成：完整4/16，空中触球16/16，条件恢复12/16，机器人碰网1，数值故障0。样本量仅16，且该新候选已在训练组失败，不能替代正式400或宣布通过。
- 实际数据形状obs=[850, 16, 794]、action=[850, 16, 18]，有效第一回合帧13600；completed=True，观测/动作有限=True/True，最大动作绝对值1.000000。这是真实控制器输入与动作，不是新教师标签；自行模仿这些动作本身不构成性能提升。

最新完整报告reports/G07/REPORT.md，结算reports/G07/lastwindow_handoff.json。公开数据供分析/学习，不是教师标签；自我模仿不能证明提升。B16已暴露，不再把fresh128全体称未使用。

## 检查点和恢复

最佳已知参考reports/G07/formal_candidate_v1/policy.pt，SHA d24460e97fc16ea378606caf6d17682db72886b0550115a5fb9f2c144747dba6（正式未通过）。最佳新候选reports/G07/lastwindow_training_candidate_v1/policy.pt，SHA8b2b7dbb8b6c6fbd215a236fa72f66bb4feabcadd8fda593abce6c9546511902（训练未通过且低于参考）。不要混称二者合格。

CPU可运行：
`CUDA_VISIBLE_DEVICES='' /home/autodrive/Isaac/env_isaaclab/bin/python scripts/analyze_g07_public_rollout.py --input reports/G07/lastwindow_fresh_b16_stress01`

具体下一条GPU数据采集命令在status/G07_RESUME.json，当前尚未运行。先明确修订时间授权，保留累计10GPUh/T0/历史；scripts/run_gpu_job.py现有固定截止会拒绝原窗口外启动。不要直接跑child绕过guard，也不要调用legacy train launcher来避开新窗口限制。下一方法应实现/训练基于公开身体状态的恢复适配，而非继续机械固定参数搜索；该新学习器尚未实施，不能假定数据采集即完成学习。

如后来新策略真正开发合格，创建新锁03/新输出，同一权重源码完成原400；保留失败lock02，禁止改acceptance/verify/硬件/面板/阈值。独立B16不是400替代。G10最终holdout/多种子未做。

## GitHub/交付

工作分支g04-guided-learning-20261022；main与release tag自动源码包可能落后。release g07-progress-20261023已有失败400包、正式B固定case0/1视频（51秒）以及历史开发包。最新G07_lastwindow_development_20261024.zip/json已上传（73,321,942bytes，SHAa61d01eeedae61d5c28ad145f446c843468f065d64a29277dbca9c2dae711d22）；431文件，含实际权重与完整公开数据。GitHub digest已核对，详见reports/G07/lastwindow_release.json。

原NEXT和REPORT逐字节保存在reports/G07/history_before_window_end_20261024。资产说明assets/assumptions.md必须保持G01签名SHA b78c443cb89de38d73f8b730fc7fba7e72b28c4e8f2dd2ffca4db70835a26b4a；G07假设另列ASSET_ASSUMPTIONS.md。USD SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3。只管理本项目进程，不动MPS/AWESun/其他作业。
