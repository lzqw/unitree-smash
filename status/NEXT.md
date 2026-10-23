# 正式验证入口（2026-10-23 16:30 UTC）

固定资源窗口UTC08:04:13.742590至18:04:13.742590（北京时间10月23日16:04至10月24日02:04），仅GPU0、新增最多10 GPU-hours。历史不清零、重启不续期。

vertical_preparation_matrix_fine1536_01已SUCCEEDED，不能重启。candidate8=vertical_back3_vx10，标准full47/64、AIR64/64、rec57/64；Bfull33/64、AIR62/64、rec56/62；该候选及全1536数值0。Net标准6/B5仍保留在分母。唯一已开发合格的固定候选已复制到reports/G07/formal_candidate_v1/policy.pt，FROZEN.json保留源SHA、开发数据与未正式通过声明。原前向/arm比较未过的数据均保留。

下一步：生成不可变status/g07_causal_formal_lock02.json（旧lock01弃用，从未正式运行），协议configs/g07_causal_profiled_net_eval_v3.json，SHA6ce5c9929aa163504b39c24c5e85f1f2e939d43dde0d10beb48b55dc77a5319c。按冻结A/B各200，使用完全相同checkpoint与所有锁定源码；面板间不能修改quadsmash、eval入口或锁定配置。阈值400/full.5/AIR.7/rec.85/NF0不变。Formal seeds按原冻结panel.seed717301/717401，不挑种子。A/B原文件scenarios未改，原侧高网v3、62.5us/.01s/8.5s。sweep最大4096/原1e-6容差，旧256故障保留；CPU/CUDA10查询复核已通过。Formal为真实独立验证，不能把开发结果先标PASSED。

若正式两组都完成且aggregate重算均达门槛，再发布artifacts/G07/policy.pt（与冻结文件原字节相同）、G07 evidence/报告，并verify_goal。若任何失败，原始逐回合照保留，不训练或挑选正式case，不改门槛；回到原开发面板改进。预算到期保存停止，不把未完成称complete。

运行前先查ledger/ps，不能重复启动。Isaac Python=/home/autodrive/Isaac/env_isaaclab/bin/python。Formal每panel envcap170000，guard max2100s（必须在固定截止前），输出NEW目录并require completed.json。程序scripts/evaluate_g07_causal_hybrid.py --checkpoint reports/G07/formal_candidate_v1/policy.pt --protocol configs/g07_causal_profiled_net_eval_v3.json --policy-lock status/g07_causal_formal_lock02.json --launch-panel reports/G07/causal_profiled_eval_panel_[a/b]_v2/panel.json --calibration reports/G07/causal_dev_panel64_v1/calibration.json --physics-dt .0000625 --episode-seconds 8.5 --seed [717301/717401] --headless。

资产20.646991kg/原USD SHA bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3，原PD/力矩/速度不变。中心1.524/侧1.55、宽2/厚.02实体研究障碍，不是真实织网/场地标定。长准备发球是版本化模型场景，不声称现实硬件认证。

G01签名assets/assumptions.md保持SHA b78c443cb89de38d73f8b730fc7fba7e72b28c4e8f2dd2ffca4db70835a26b4a；G07新说明单列ASSET_ASSUMPTIONS.md。16:19 verify依赖G00-G06全过、G07只缺正式evidence。勿修改验收器/阈值/历史证据。

代码报告分支g04-guided-learning-20261022，最新已推e92e0e0。手工增量G07_arm_audit_update_e92e0e0.zip已上传（32,240,141bytes，SHAb90c4795da2f97fe6b3d08313e80046099625b14673f6a2dfc0cae8f90c8be5e）；不含本批刚完成vertical实际结果。release g07-progress-20261023。已有body8成功+失败视频和forward9权重均为开发历史记录。发布tag自动源码只代表最早tag时刻，当前源码应看分支或手工增量。

备用contact_recovery/vertical_push/causal_parameter_selector代码已存在，但当前已合格candidate8不包含这些字段，保持关闭；CPU逻辑检查并不当策略成绩。原CV没有达标，所以不把选择器送GPU。fresh128独立开发面板仍未使用；对已合格固定策略直接做更大正式400，合理预留评测时间。
