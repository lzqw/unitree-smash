# 当前继续入口：正确球网下靠近网前准备与起跳（2026-10-23）

G07为IN_PROGRESS；正确物理网下无借力完整成功仍为0。G04–G06与所有旧权重保留。

## 固定授权与当前作业

GPU0；独立新增最多10 GPU-hours，窗口 `g07-net-feasibility-20261023`：Asia/Taipei 2026-10-23 16:04:13 至 2026-10-24 02:04:13（UTC08:04:13至18:04:13）。重启或切换方法不重置。成本以GPU_LEDGER.json为准。

当前统一exec会话15404：`walk_close_high01`，30代×128环境×4秒，从root0学习行走至x1.10，提前抬臂；无球课程。先轮询现有进程并检查job.json/metrics.json，不重复启动。

## 可信与无效证据

- 物理网修复为环境局部[1.5,0,.762]。`net_blocking_probe03/verification.json`：32/32位置正确并实际挡球；诊断原metrics末速度判定错误，须读verification。
- 旧53/64教师、54/64学习策略的网实际偏移，已撤回。正确网冻结PWA复验0/64完整（恢复36/64）。发布页有更正和当前失败视频。
- 首版分组ContactSensor源/目标数量不匹配，零值无效；当前每刚体一个过滤传感器，`net_robot_contact_positive02`已4/4检出真实网接触。不要再用旧分组零值证明不碰网。
- `composed_replay_netlocal02`：从root0走到击球32/32空中有效拍面接触与越网，首次目标落点2/32、恢复2/32，但32/32碰网，数值1，无借力完整0。冻结无球复验 `composed_motion_net_timing01` 显示Racket在4.670s先碰网，早于约4.895s击球，不能用作可行教师。
- `composed_high_ready01`20代×128：无碰网、腾空、恢复的最佳目标匹配点[1.313577,.005240,1.579334]m，仍不足，不是硬件上界。

## 当前方法与下一步

`close_base_ik_cpu01/pose_seed.pt` 是条件IK：机身x1.10、高.75、水平，拍面中心附近可达[1.4,0,1.65]、法向[.866,0,.5]且拍面与网SAT分离约5.5cm。只证明给定机身状态下几何可行，未证明动态到达、自碰撞或全臂安全。

先检查close高臂行走是否稳定无碰网，再用该walker、该arm seed运行 `search_g07_composed_reach_v2.py`，可用 `--deep-crouch --vertical-seed --flight-objective --landing-feedback`。深蹲仅扩展搜索范围，实际原关节/扭矩/速度限制不变。飞行评分是理想恢复系数/无阻力弹道预测，只作塑形，必须实际BallEnv验证。

所有GPU命令包在scripts/run_gpu_job.py，固定上述budget-window；保存checkpoint和完成回执。不把条件IK、无球拍面点、教师或单发重复当正式策略成绩。真实可行后继续学习、扩大来球/目标、全程自主独立双固定面板至少400回合。

## 复现资料

Go2源行走actor、来源和sha：configs/g07_go2_walk_transfer.json。TorchScript在assets/vendor/go2_flat_baryn20251217（忽略目录），可按原URL/sha下载。原硬件标准PiPER+Go2+球拍20.646991kg，资产sha bfb831e4afc8a1a705bb0fb6f001c25eeaa401822978267a0ee5e151007511f3。行走为冻结学习actor+搜索反馈/臂姿，击球目前为搜索教师，不冒称新神经策略。

延长准备7.4s场景的球在reset时自由发射一次，部分初始位置达数十米，是单独模型探测版本，不能混入原面板。GitHub进展页：https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。
