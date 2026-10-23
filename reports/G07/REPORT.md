# G07：工作窗结束，正式验收未通过（BUDGET_EXHAUSTED）

北京时间2026-10-24 02:04原工作窗已到期，已保存并停止本项目GPU0作业。G00–G06依赖校验仍通过；G07未达到完整扣杀和条件恢复门槛，没有生成通过证据或正式policy产物。当前代码与报告在[工作分支](https://github.com/lzqw/unitree-smash/tree/g04-guided-learning-20261022)，main与发布tag自动源码包可能较旧。

## 正式400回合结果

锁02使用同一冻结权重/源码完成原固定A、B各200回合；全部失败保留。权重SHA256 `d24460e97fc16ea378606caf6d17682db72886b0550115a5fb9f2c144747dba6`，运行源码b10eeffd049064861c6245a3fd792ed8e5753f99，协议g07_causal_profiled_net_eval_v3。

| 面板 | 完整扣杀并恢复 | 空中触球 | 条件恢复 | 机器人碰网 | 数值故障 |
|---|---:|---:|---:|---:|---:|
| A | 100/200（50%） | 200/200 | 178/200（89%） | 10 | 0 |
| B | 71/200（35.5%） | 196/200（98%） | 146/196（74.4898%） | 22 | 0 |

原标准按两组较差比例判定：完整≥50%、空中触球≥70%、条件恢复≥85%、数值故障0、地面起始100%、总样本至少400。本次完整与恢复未达标；不能将两组平均值、筛掉碰网后的比例或开发小样本代替它。

原始数据：[A200](causal_profiled_formal_a_200_v2/episodes.json)、[B200](causal_profiled_formal_b_200_v2/episodes.json)、[聚合400](causal_formal400_v2/metrics.json)、[不可变源码/策略锁](../../status/g07_causal_formal_lock02.json)、[失败结论](../../status/g07_causal_formal_lock02.outcome.json)。本目录根metrics/episodes/eval_manifest保留这次正式失败400，不用后续小样本覆盖。[事件链CPU复核](causal_formal400_v2/failure_chain_audit.json)发现0个完整标记矛盾；这不等于另一次物理评测。

互斥失败顺序为数值→机器人碰网→无空中触球→无效攻击（含多次触球）→未越网→未命中首次目标落点→未恢复。A对应0/10/0/31/15/44/0，成功100；B对应0/22/2/43/23/37/2，成功71。条件恢复另对所有空中触球计算，因此目标命中后只有2次未恢复不表示整体恢复已达标。

## 后续开发试验

事先冻结的推离/朝向/触球后收拍12×128比较已完成：全1536数值0，没有新候选两组同时合格。加强推离增加碰网，过早切换站立降低恢复。原始结果在[contingency矩阵](contingency_push_yaw_matrix_fine1536_01/candidate_metrics.json)。

随后使用原64开发来球、A/B原噪声宽度的独立reset draws（719201/719202），两组均有噪声；不是正式case标签拟合。12×128比较全1536数值0，无新候选达标。[全部12候选结果](landing_velocity_noisy_matrix_fine1536_01/SUMMARY.md)和逐回合均保留。等到实际落地才启动站立或延迟到local1.85均明显退化。

| 新噪声训练128 | A完整 | B完整 | A条件恢复 | B条件恢复 |
|---|---:|---:|---:|---:|
| 参考0（原失败正式控制器） | 44/64 | 27/64 | 61/64 | 48/63 |
| 预定规则选出的新候选3 | 40/64 | 24/64 | 54/64 | 45/64 |

候选3为vx_bias_zero，未优于参考且未达标；冻结只用于下一次独立压力测试/数据采集，不替换最佳。原字节与全部排名见[lastwindow_training_candidate_v1/FROZEN.json](lastwindow_training_candidate_v1/FROZEN.json)，SHA256 `8b2b7dbb8b6c6fbd215a236fa72f66bb4feabcadd8fda593abce6c9546511902`。参考仍是formal_candidate_v1/policy.pt，但其正式结果也没有通过。

## 最后一次独立压力测试与公开轨迹

预先固定fresh面板的B前16个样本，在看到本轮结果前声明；剩余工作窗不足B32，选择B16，seed719301，原62.5微秒物理/.01秒控制/8.5秒、4096次原容差扫掠。作业状态**SUCCEEDED**，耗时897.177秒，计费0.249215882 GPUh。全部16回合完成：完整4/16，空中触球16/16，条件恢复12/16，机器人碰网1，数值故障0。样本量仅16，且该新候选已在训练组失败，不能替代正式400或宣布通过。

实际数据形状obs=[850, 16, 794]、action=[850, 16, 18]，有效第一回合帧13600；completed=True，观测/动作有限=True/True，最大动作绝对值1.000000。这是真实控制器输入与动作，不是新教师标签；自行模仿这些动作本身不构成性能提升。

在16个有实际触球后落地记录的案例中，落地前约30毫秒的公开机身下降速度为-2.865至-2.651m/s。它提供了状态条件恢复的诊断线索，不能证明某个新阈值或学习控制已有效；该16样本已暴露，只能作为后续开发数据。

原始结果/数据在[lastwindow_fresh_b16_stress01](lastwindow_fresh_b16_stress01/)，诊断见[PUBLIC_DATA_AUDIT.json](lastwindow_fresh_b16_stress01/PUBLIC_DATA_AUDIT.json)。每100步保存partial，完整数据与失败/超时严格分开；数据张量通过发布附件下载。原fresh128中的这16个B样本已使用，不能再宣称全128未见。

## 程序与能力边界

- [learned_hybrid.py](../../quadsmash/control/learned_hybrid.py)：冻结学习Go2行走/站立网络、从真实示范学得12段PWA、模型预测和公开状态反馈的组合。不是本轮新训端到端PPO；教师前缀和教师动作函数调用均0。
- [evaluate_g07_causal_hybrid.py](../../scripts/evaluate_g07_causal_hybrid.py)：原生刚体、扫掠审计、全部回合计分；开发专用`--record-public-dataset`记录真实794维观测、18维限幅动作及第一回合mask。
- [freeze_g07_training_candidate.py](../../scripts/freeze_g07_training_candidate.py)：按事前规则选新候选，拒绝正式标签作为训练输入，冻结原字节，不生成通过声明。
- [analyze_g07_public_rollout.py](../../scripts/analyze_g07_public_rollout.py)：CPU检查实际数据与触球后落地附近的公开速度/角速度/关节状态，不赋予缺失回合成绩。
- [run_gpu_job.py](../../scripts/run_gpu_job.py)：单进程GPU0、固定原截止、累计费用、失败计费、源码快照和硬超时。

标准PiPER+Go2+安装板+球拍20.646991kg，资产/质量/原PD/力矩/速度/关节范围保持原限制。网体中心1.524m、侧1.55m、宽2m、厚.02m、32段抛物线顶，仍是实体研究障碍而非真实织网。长准备发球初位可达数十米，没有现实气动/织网/实机标定。[G07资产假设](ASSET_ASSUMPTIONS.md)与原G01签名说明分开。原错误网坐标53/64教师、54/64学习声明已撤回；恒高网单一来球32/32只保留作历史课程，不作侧高网验收。

## 视频与交付

[正式B固定case0/1视频，51秒](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/G07_formal_B_fixed_examples_20261024.mp4)来自实际50Hz连杆/球位姿和原USD几何CPU离线渲染，固定录制，不按结果挑选。视频页脚的恢复总数是原始非条件恢复，正式条件指标以本报告146/196为准。视频SHA256 `f2edb5142e98816437d138a74a43e19de42d52de2a3d45eb02b45425c1f4b6f1`。

[失败400完整包](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/G07_formal400_failed_lock02_20261024.zip)（5,691,849bytes，SHA256 `e6bc7892efea822792aacc07c61c2a73a0b98561e0835fac00dd95982e6aaf2b`）保留策略、源码、锁与全部结果。最新增量包的状态、文件名和SHA以发布页及对应JSON为准；Python/Isaac环境与基础runtime资产仍按既有[恢复说明](../../docs/HANDOFF_ARTIFACTS.md)准备，不能把增量包当作完整环境。

## 资源、瓶颈与继续入口

原固定窗口北京时间10月23日16:04:13至10月24日02:04:13，仅GPU0、本轮新增最多10 GPUh、历史保留。实际结算**7.645612263 GPUh**，共64个作业，状态分布{'SUCCEEDED': 54, 'FAILED': 10}；全部搜索、失败、诊断、训练和评测计入。计费按GPU作业墙钟而非利用率积分。此次先耗尽固定工作窗，GPUh上限尚余2.354387737，不代表可以越过原时间限制继续。[最终账本摘要](lastwindow_handoff.json)、[完整账本](../../status/GPU_LEDGER.json)。

用户最新允许随时调用GPU；是否取消原02:04截止的异步澄清尚无明确答复，因此仍执行原授权“工作时间窗到期……即保存并停止”。没有新建窗口、重置T0或清零费用。停止仅因已到资源限制，未将目标标为完成。

现有证据不支持继续机械搜索固定交接时间/推离增益，也不能证明硬件不可能。后续优先验证公开身体状态条件下的恢复适配；需要新训练和独立细物理评测，不能把自己动作的BC拟合误称为改进。具体可运行的数据采集入口保存在[status/G07_RESUME.json](../../status/G07_RESUME.json)：当前最强参考在既有双噪声训练面板采集完整公开轨迹，属于重复训练数据采集，不是未见验证。该命令尚未运行，新恢复学习器也尚未实现；先获得对现有截止的明确修订，保留本轮累计10GPUh上限与全部费用。

后续真正合格的新策略还需新不可变锁、同权重同源码完成原A/B400全部评测。G10最终holdout/多种子未做。最终结构检查[structural_window_end_20261024.json](structural_window_end_20261024.json)中G00–G06均通过，G07因无合格证据而失败。此前报告与状态说明原样归档在[history_before_window_end_20261024](history_before_window_end_20261024/)。
