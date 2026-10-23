# 当前独立B16压力测试（2026-10-23 17:48 UTC）

活跃exec40287，reports/G07/lastwindow_fresh_b16_stress01，guard PID1238576/child1238588，启动17:48:08 UTC；GPU0、16×850、max959s/envcap13600，原62.5us/.01s/8.5s，4096/1e-6、seed719301，额外公开数据采集。先查ledger/ps，不能重复启动。

landing_velocity_noisy_matrix_fine1536_01已SUCCEEDED，1536数值0、无候选达标。结果见该目录SUMMARY.md。原参考0A/B完整44/64、27/64，恢复61/64、48/63；最佳新候选3A/B完整40/64、24/64、恢复54/64、45/64，仍低于参考且未达标。等落地/延迟站立普遍严重退化。不能把新候选冻结说成进步或替换最佳。

按事先固定规则排除参考0，冻结候选3到lastwindow_training_candidate_v1/policy.pt，SHA8b2b7dbb8b6c6fbd215a236fa72f66bb4feabcadd8fda593abce6c9546511902，FROZEN.json保留全部排名。只用于事先固定的B16压力测试与实际公开轨迹数据采集，不是正式验收。B16来自未使用fresh128的固定前16个B样本，不能再把这16个称未使用。不能在读该子集结果后换样本/候选。

当前evaluator --record-public-dataset记录真实obs794/action18/第一回合mask，每100tick原子保存public_dataset.partial.pt，若850步与完整结果验证完成则保存completed=True的public_dataset.pt。部分数据/失败不能当作完成回合或训练成功。若guard到时中断，保存partial与FAILED记录，禁止将其丢弃重开来清零成本。

旧窗口固定截止UTC18:04:13.742590，GPU0新增累计10GPUh；不因重启重置。用户是否取消旧时间截止的异步澄清仍未答，不把等待当授权。当前job按原截止设硬超时。到期保存并停止本项目GPU，STATUS=BUDGET_EXHAUSTED；若目标尚未达标不得goal complete。G00–G06保持通过。

GPU结束后CPU整理：检查completed/job/候选指标/公开数据，更新报告和真实费用；运行verify_goal（仍应缺G07 evidence）；显式stage本批原始逐回合/元数据，推工作分支；/tmp/build_g07_lastwindow_archive.py会在ZIP关闭后计算SHA并包含权重/源码/实际新数据，先审阅目录清单再打包上传新附件，不覆盖历史附件。现有正式B视频和失败400包已上传。新小样本不是替代正式400。

# 当前新训练比较（2026-10-23 17:32 UTC）

活跃exec37919，reports/G07/landing_velocity_noisy_matrix_fine1536_01，启动17:28:54 UTC；12×128，max1800s/envcap1305600，62.5us/.01s/8.5s，4096/1e-6，seed719251。先看ledger/ps，不能重复启动。

输入新训练panel causal_profiled_noisy_both128_v2/panel.json：原64开发发球分别配原A/B噪声宽度，重置PCG64种子719201/719202；不使用正式回合/标签，正式A/B文件不改，旧fresh128仍未使用。新panel原样冻结，creation_manifest.json说明实际新reset seeds；顶层继承reset_seed仅历史源元数据，程序实际读每case明确robot_reset数组。此批分组training_a/training_b均有扰动，不与旧canonical/B比率直接混。

新recovery_wait_for_landing字段：当local>=1.25并当前public774真实落地phase时锁存，然后仅在原standing_start已到且见到落地后才接管standingactor，默认False保留旧行为。不是使用未来落地时刻。landing_gate_cpu01/metrics.json通过：关闭时与上批启动源码动作差0、落地前不启动、每lane独立、bounce保持锁存、reset清除。原开发时序诊断显示正常无Net落地约local1.752，未恢复B样本约1.79，说明固定交接敏感，但新方式实际效果待本批。

12冻结候选landing_velocity_candidates_v1：0原失败formal策略reference；1等实际落地；2standing_start1.85；3/4 vx目标bias0/-.1；5bias-.1+等落地；6/7强vx gain-.25、bias0/-.1；8准备后移4cm+bias0+等落地；9后移4cm+bias-.1+pitch-.025+等落地；10原后移3cm+bias0+pitch-.025+等落地；11关闭共同vx+等落地。原joint/PD/力矩/速度不变，实际性能尚未完成。

前批contingency_push_yaw_matrix_fine1536_01已SUCCEEDED，没有新的固定候选两组同时合格。最强提前standing那组rec仅约.29，推离多数增加Net，因此不机械重复。完整候选指标/已知formal失败仍保留。

原固定截止UTC18:04:13.742590（北京时间02:04），GPU0且当前窗口累计10GPUh不重置。异步问题已发，用户“随时GPU”是否取消旧时间截止仍待明确答复；不能把等待当授权。若收到明确取消，只能取消wallclock而不重置GPU10h/T0/历史，需更新实际budget/授权记录及guard支持；guard当前job持有内存ledger，勿在作业运行时并发修改ledger以免其退出覆盖。可先更改budget+独立授权文件，作业结束后或下个guard统一写ledger。

本批完成后如无正式合格，不得伪造通过。旧窗口不足新完整400时，保存最佳、原始失败与下一条入口；时间到按要求停止GPU，G07为BUDGET_EXHAUSTED而非PASSED。G00–G06保持不变。

# 原开发改进试验运行中（2026-10-23 17:07 UTC）

统一exec91880，reports/G07/contingency_push_yaw_matrix_fine1536_01，启动17:05:12 UTC，12×原128开发，max1500s/envcap1305600、62.5us/.01s/8.5s、4096/1e-6、seed719101。先查ledger/ps，禁止重启。候选contingency_push_yaw_candidates_v1已在17:00前、任何正式结果出来前冻结：frozen8基线及yaw+.04/.08/.12、推离+.05/.1、组合yaw.06/触球收拍，以及旧back2/forward9组合，详见manifest；都不是已通过策略。

lock02正式400已全部完成且失败：A完整100/200=.5、AIR200/200、rec178/200=.89、Net10、NF0；B完整71/200=.355、AIR196/200、rec146/196=.744898、Net22、NF0。较差full/rec未达原门槛。根metrics/episodes/eval_manifest已指向causal_formal400_v2，旧根原样归档history_before_formal400_lock02。没有G07 evidence或artifacts/G07/policy.pt。锁02不改，独立outcome记录失败。源码b10eeff评测两组相同；现在两组已完，可以提交后处理修复/报告和下一批。

当前GPU只读原开发128文件，不训练或输入正式回合标签。若此12个固定候选合格，合理预留剩余窗口做新冻结400，并记录这是第二次G07开发验收，不能声称未见G10holdout。候选/源锁新建版本v2/lock03，旧400失败永不覆盖。fresh128仍未使用，可在额外时间获准后优先作独立复验以识别旧开发集调参乐观偏差。

用户最新“继续跑你可以随时调用gpu”；已异步澄清是否取消原北京时间02:04截止、仍保持GPU0与新增10GPUh总上限不重置。问题待答，不能把等待时间当授权。现有试验继续按原固定UTC18:04:13.742590截止运行。授权原文附件020eb582...明确到期保存停止，原账本保留。

# 正式B组运行中（2026-10-23 16:49 UTC）

A组causal_profiled_formal_a_200_v2已SUCCEEDED，完整100/200=.5、AIR200/200、条件恢复178/200=.89、Net10、数值0。当前统一exec17902，reports/G07/causal_profiled_formal_b_200_v2（先查ledger/ps，禁止重复）。seed717401、同一冻结policy SHA d24460e97fc16ea378606caf6d17682db72886b0550115a5fb9f2c144747dba6、同protocolv3 SHA6ce5c992.../源码锁02。B结束前不得更改锁定策略/源码/配置。A/B eval_manifest均应source_git_sha b10eeff，当前保持Git HEAD不变。

B完成后运行：
/home/autodrive/Isaac/env_isaaclab/bin/python scripts/publish_g07_causal_result.py --protocol configs/g07_causal_profiled_net_eval_v3.json --policy-lock status/g07_causal_formal_lock02.json --checkpoint reports/G07/formal_candidate_v1/policy.pt --panel-output reports/G07/causal_profiled_formal_a_200_v2 --panel-output reports/G07/causal_profiled_formal_b_200_v2 --aggregate-output reports/G07/causal_formal400_v2

该CPU发布入口会先调用aggregate按原始400回合重算；任何门槛未满足都返回失败且不动根evidence/policy。成功才原样归档旧根文件、拷贝冻结policy到artifacts/G07、写含实际job起止时间/成本/源码/产物哈希的evidence并verify_goal，通过后更新PASSED。源码后处理aggregate增加ROOT导入路径修复，不改变评测逻辑；未编辑锁定runtime文件。publication_gate_negative_cpu02验证把真实开发批次传入时被拒绝、根报告/证据/策略不变。原导入失败记录cpu01保留。两者仅CPU软件测试，不是策略成绩。

如果400不达标，完整聚合和两个原始面板必须保留，不改标准也不训练正式case标签。在读任何正式结果前，已根据原开发数据预先冻结contingency_push_yaw_candidates_v1/manifest.json共12个备用权重（小幅joint1正yaw/有界calf推离/可选当前触球后收拍），尚未GPU运行；只能在剩余授权内用原开发128真测。当前已锁定400必须用原权重完成，不能途中过渡备用。备用原型不当成绩。

资源固定UTC08:04:13.742590至18:04:13.742590，仅GPU0，新增最多10GPUh，历史保留。最新用户继续允许调用GPU，未明确重置原窗口/增加上限，按已登记配置执行。窗口剩余约1小时以上，预留发布和最终评测。

# 正式验证入口（2026-10-23 16:30 UTC）

固定资源窗口UTC08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），仅GPU0、新增最多10 GPU-hours。历史不清零、重启不续期。

vertical_preparation_matrix_fine1536_01已SUCCEEDED，不能重启。candidate8=vertical_back3_vx10，标准full47/64、AIR64/64、rec57/64；Bfull33/64、AIR62/64、rec56/62；该候选及全1536数值0。Net标准6/B5仍保留在分母。唯一已开发合格的固定候选已复制到reports/G07/formal_candidate_v1/policy.pt，FROZEN.json保留源SHA、开发数据与未正式通过声明。原前向/arm比较未过的数据均保留。

旧exec46369 / formal A已完成。源码Git HEAD b10eeff，请保持到两组都完成；不要编辑锁定源码。不可变status/g07_causal_formal_lock02.json已生成（policySHA d24460e97fc16ea378606caf6d17682db72886b0550115a5fb9f2c144747dba6）。下一步A完成后直接同权重/源码运行B200。协议（旧lock01弃用，从未正式运行），协议configs/g07_causal_profiled_net_eval_v3.json，SHA6ce5c9929aa163504b39c24c5e85f1f2e939d43dde0d10beb48b55dc77a5319c。按冻结A/B各200，使用完全相同checkpoint与所有锁定源码；面板间不能修改quadsmash、eval入口或锁定配置。阈值400/full.5/AIR.7/rec.85/NF0不变。Formal seeds按原冻结panel.seed717301/717401，不挑种子。A/B原文件scenarios未改，原侧高网v3、62.5us/.01s/8.5s。sweep最大4096/原1e-6容差，旧256故障保留；CPU/CUDA10查询复核已通过。Formal为真实独立验证，不能把开发结果先标PASSED。

若正式两组都完成且aggregate重算均达门槛，再发布artifacts/G07/policy.pt（与冻结文件原字节相同）、G07 evidence/报告，并verify_goal。若任何失败，原始逐回合照保留，不训练或挑选正式case，不改门槛；回到原开发面板改进。预算到期保存停止，不把未完成称complete。

运行前先查ledger/ps，不能重复启动。Isaac Python=/home/autodrive/Isaac/env_isaaclab/bin/python。Formal每panel envcap170000，guard max2100s（必须在固定截止前），输出NEW目录并require completed.json。程序scripts/evaluate_g07_causal_hybrid.py --checkpoint reports/G07/formal_candidate_v1/policy.pt --protocol configs/g07_causal_profiled_net_eval_v3.json --policy-lock status/g07_causal_formal_lock02.json --launch-panel reports/G07/causal_profiled_eval_panel_[a/b]_v2/panel.json --calibration reports/G07/causal_dev_panel64_v1/calibration.json --physics-dt .0000625 --episode-seconds 8.5 --seed [717301/717401] --headless。

资产20.646991kg/原USD SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3，原PD/力矩/速度不变。中心1.524/侧1.55、宽2/厚.02实体研究障碍，不是真实织网/场地标定。长准备发球是版本化模型场景，不声称现实硬件认证。

G01签名assets/assumptions.md保持SHA b78c443cb89de38d73f8b730fc7fba7e72b28c4e8f2dd2ffca4db70835a26b4a；G07新说明单列ASSET_ASSUMPTIONS.md。16:19 verify依赖G00-G06全过、G07只缺正式evidence。勿修改验收器/阈值/历史证据。

代码报告分支g04-guided-learning-20261022，最新已推e92e0e0。手工增量G07_arm_audit_update_e92e0e0.zip已上传（32,240,141bytes，SHAb90c4795da2f97fe6b3d08313e80046099625b14673f6a2dfc0cae8f90c8be5e）；不含本批刚完成vertical实际结果。release g07-progress-20261023。已有body8成功+失败视频和forward9权重均为开发历史记录。发布tag自动源码只代表最早tag时刻，当前源码应看分支或手工增量。

备用contact_recovery/vertical_push/causal_parameter_selector代码已存在，但当前已合格candidate8不包含这些字段，保持关闭；CPU逻辑检查并不当策略成绩。原CV没有达标，所以不把选择器送GPU。fresh128独立开发面板仍未使用；对已合格固定策略直接做更大正式400，合理预留评测时间。
