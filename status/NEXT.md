# 当前继续入口：正确物理球网下重新搜索（2026-10-23）

G07仍为IN_PROGRESS，没有满足正确物理网的有效完整策略。G04–G06和旧G07权重全部保留。

## 固定授权

`g07-net-feasibility-20261023`，GPU0，独立新增最多10 GPU-hours；Asia/Taipei 2026-10-23 16:04:13 至 2026-10-24 02:04:13（UTC08:04:13至18:04:13）。重启/阶段切换不重置。此前用量不清零、不占用本轮10h。当前已完成作业实收约1.214976 GPUh，运行作业另有预留，账本为准。

## 必须知道的更正

旧BallEnv/步态环境在子节点Net使用世界坐标作局部平移，导致物理网远离标称位置。修复为环境局部[1.5,0,0.762]，尺寸[.02,2,1.524]和机器人资产/执行器均不变。

- `net_blocking_probe02/verification.json`：32/32旧网偏移，0挡球。
- `net_blocking_probe03/verification.json`：32/32正确位置且在网前反射、不穿网。原诊断metrics的末速度判定受后续反弹影响，须读verification.json，原数据保留。
- 正确网冻结PWA复验 `eval_piecewise_netlocal02`：64/64空中触球，有效越网/目标首次落点/完整均0，恢复36/64，数值0。此前53/64教师、54/64学习策略结论撤回，仅保留历史/几何数据。对应目录新增scene_invalidation.json。
- 旧击球和旧行走的球拍与正确网有几何交叠，检查在 `net_blocking_probe03/old_motion_net_intersection.json`。需要修正行走腕部姿态，并推迟起跳前的引拍。

## 正在运行（轮询现有进程，不能重复启动）

[{"output": "/home/autodrive/LZQW/unitree_smash/reports/G07/composed_reach_netlocal02", "child_pid": 1160469, "guard_pid": 1160458}]

统一exec会话80971，对应 `composed_reach_netlocal02`：正确Net下，从root0开始调用已学习Go2行走actor，接40参数全身击球教师。16代×128环境×7.4秒，无球课程，不是实际击球成绩。新增独立机器人/网接触力监测，初期回合已实际运行，无程序错误。

## 最新可用控制器与来源

- `search_walk_netlocal02/best_candidate.pt`：正确网下搜索出的12参数行走适配器（速度/反馈、臂姿与腕部角度）；搜索多条成功、典型终点约x1.0，无保守球拍干涉。尚待冻结独立复验，不要把搜索内不同参数数目当重复成功率。
- 行走actor为作者baryn公开平地Go2 TorchScript，48维输入、50Hz、动作.25rad，腿部25/.5增益、23.5Nm、30rad/s。来源和sha在configs/g07_go2_walk_transfer.json，二进制在assets/vendor/go2_flat_baryn20251217（忽略目录）；URL可按sha重新下载。已审查TorchScript仅Linear/ELU/Identity，无修改全局环境。
- 原学习PWA `fit_piecewise_curriculum03/policy.pt` 保留作初始化，不是正确网达标策略。
- `composed_reach_teacher.py` 保留步行臂姿直至起跳阶段再展开，参数schema为composed-reach-v2-40；其中后两维为引拍开始/时长，前38为原全身/挥拍/恢复反馈。绝不把它标为新的学习策略。

## 后续

先看当前搜索能否产生足够高、无机器人网接触的有效拍面事件，再做从初始自由来球发射开始的真实碰撞搜索。CPU长准备时间飞行候选在delayed_launch_cpu02（已匹配每个目标高度，01其他高度为近似旧记录）；条件IK在walk_strike_ik_cpu01。均不是动态成功证据。

当前无球目标约[1.4,0,1.65]、法向[.866,0,.5]，在实际矩形拍面内选择最接近目标的点，避免只奖励最高角。保持完整硬件、标准网和原验收。发现真实可行路线后继续学习、扩展来球/目标，并进行原始起点全程自主双固定面板至少400回合；不能在课程阶段停止。

GitHub旧发布页已明确撤回物理网成功结论： https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023 。旧资料不可混入修复后评估。更早继续入口已归档至status/history/NEXT_before_net_coordinate_fix_20261023.md。
