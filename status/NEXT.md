# 当前继续入口：正确球网下靠近网前准备与起跳（2026-10-23）

G07为IN_PROGRESS；正确物理网下无借力完整成功仍为0。G04–G06与所有旧权重保留。

## 固定授权与当前作业

GPU0；独立新增最多10 GPU-hours，窗口 `g07-net-feasibility-20261023`：Asia/Taipei 2026-10-23 16:04:13 至 2026-10-24 02:04:13（UTC08:04:13至18:04:13）。重启或切换方法不重置。成本以GPU_LEDGER.json为准。

`walk_close_high01`已完成30代×128；冻结在0.25ms夹具 `walk_close_fine_replay01` 独立32/32走到x1.10附近并站稳、0碰网。`composed_compact_flight01`主动保存完成5/20代：最佳点[1.426609,.002736,1.637931]，预测净空.020573m，腾空且全程无网接触，恢复尚未通过。为做真实球校准而提前结束，运行器FAILED与method_switch.json保留；后续可从population_004.pt恢复mean/sigma/RNG。

`compact_native_contact01`已完成64回合：净无碰网18，其中有序越网14、首次目标落点8，恢复/完整0，数值0。教师214族8/8无碰网、6/8目标；冻结case37独立16回合 `net_free_teacher214_replay01`：16/16空中有效拍面/越网/首次目标、0网接触/故障，恢复0。最新纯净metadata的相同控制器是 `compact_native_contact01/net_free_teacher214_v2.pt`（v1原件保留，v2删除从另一lane继承的无用诊断字段，执行参数相同）。

当前统一exec会话66329：`standing_recovery_search01`，128环境×16代无球落地搜索，冻结有效击球，接入原学习Go2 actor的零速度站立/落地控制；已有通过原无机身触地、无碰网、0.8s稳定恢复的回合。先轮询该进程，不重复启动。运行中PID在GPU_LEDGER；job.json结算后才生成。

`fit_composed_strike214_01/policy.pt`已CPU训练12段反馈，留出动作最大误差4.17e-6。接下来用 `probe_g07_composed_contact.py --candidate standing_recovery_search01/best_candidate.pt --strike-policy fit_composed_strike214_01/policy.pt --recovery-scan --launches compact_native_contact01/launches.json --launch-index 37 --num-envs 64`，均以reports/G07路径为前缀且使用GPU预算guard。该扫描变动恢复接入时间/腿部偏移/actor位置输出幅度，主击球为学习PWA，不调用教师函数。混合扫描数据不能拟合单一时钟策略；选中后冻结独立复验。

## 可信与无效证据

- 物理网修复为环境局部[1.5,0,.762]。`net_blocking_probe03/verification.json`：32/32位置正确并实际挡球；诊断原metrics末速度判定错误，须读verification。
- 旧53/64教师、54/64学习策略的网实际偏移，已撤回。正确网冻结PWA复验0/64完整（恢复36/64）。发布页有更正和当前失败视频。
- 首版分组ContactSensor源/目标数量不匹配，零值无效；当前每刚体一个过滤传感器，`net_robot_contact_positive02`已4/4检出真实网接触。不要再用旧分组零值证明不碰网。
- `composed_replay_netlocal02`：从root0走到击球32/32空中有效拍面接触与越网，首次目标落点2/32、恢复2/32，但32/32碰网，数值1，无借力完整0。冻结无球复验 `composed_motion_net_timing01` 显示Racket在4.670s先碰网，早于约4.895s击球，不能用作可行教师。
- `composed_high_ready01`20代×128：无碰网、腾空、恢复的最佳目标匹配点[1.313577,.005240,1.579334]m，仍不足，不是硬件上界。

## 当前方法与下一步

`close_base_ik_cpu01/pose_seed.pt` 是条件IK：机身x1.10、高.75、水平，拍面中心附近可达[1.4,0,1.65]、法向[.866,0,.5]且拍面与网SAT分离约5.5cm。只证明给定机身状态下几何可行，未证明动态到达、自碰撞或全臂安全。

先检查close高臂行走是否稳定无碰网，再用该walker、该arm seed运行 `search_g07_composed_reach_v2.py`，可用 `--deep-crouch --vertical-seed --flight-objective --landing-feedback --arm-pitch-feedback`。41参数schema为composed-reach-v3-41，索引40为倾角肩关节补偿增益；最新44参数为composed-reach-v4-44，41/42/43为收拍开始、时长和肩关节后收量，40/41参数兼容。深蹲仅扩展搜索范围，实际原关节/扭矩/速度限制不变。飞行评分是理想恢复系数/无阻力弹道预测，只作塑形，必须实际BallEnv验证。

所有GPU命令包在scripts/run_gpu_job.py，固定上述budget-window；保存checkpoint和完成回执。不把条件IK、无球拍面点、教师或单发重复当正式策略成绩。真实可行后继续学习、扩大来球/目标、全程自主独立双固定面板至少400回合。

## 复现资料

Go2源行走actor、来源和sha：configs/g07_go2_walk_transfer.json。TorchScript在assets/vendor/go2_flat_baryn20251217（忽略目录），可按原URL/sha下载。原硬件标准PiPER+Go2+球拍20.646991kg，资产sha bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3。行走为冻结学习actor+搜索反馈/臂姿，击球目前为搜索教师，不冒称新神经策略。

延长准备7.4s场景的球在reset时自由发射一次，部分初始位置达数十米，是单独模型探测版本，不能混入原面板。GitHub进展页：https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。


## 最新分支与明确失败

`composed_close_feedback01`主动完成5/24代后保存切换：腾空、击球前无碰网点[1.411354,-.048992,1.619491]，4.730s；首次网接触为Racket4.814s。代表理想飞行净空-.01952m，尚未实际击球。`02`主动完成16/24代，直立推蹬分支停滞在约1.486m、预测净空缺口.145m。两个作业FAILED是主动未满足请求代数回执，非崩溃；原数据和方法切换json保留。

当前256环境分支把实际腾空作为候选硬条件，保持初值中同一arm seed的已优化姿态/挥拍参数，目标投影x1.45，仍需全臂/拍面动态避网。CPU重建较高候选的拍面/法向误差约2e-7m/7e-7，沿x直接前移2cm会与网交叠，不能靠整体平移解决。

CPU incoming_velocity_bank_cpu01给出锁定阻力下4.8s来球：初始vx=-8..-150，最终仅约-1.71..-5.34，作为多入球速度塑形，不是实测。新`probe_g07_composed_contact.py --population ... --population-candidates 8 --strong-serve-grid`可在真实BallEnv批量比较不同教师（配合相应best_candidate.pt共享metadata），保存selected_teacher.pt；该混合教师数据不许直接拿来拟合单一时钟策略。选定后必须冻结重放再学习。程序现在也保存选中回合50Hz实测连杆位姿供视频。

程序/报告最新已推送646d3a5。修正网下的二进制快照已上传发布页资产 G07_correct_net_snapshot_646d3a5_1134UTC.zip（SHA256 954eef639ad082716afa70b94784ad027f1a57aa64e9f1eb31c4db3d0677eb3c）；其中真实球探测状态是pending，不代表完成。
