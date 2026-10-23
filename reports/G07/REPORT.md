# G07：高位高速下扣搜索与独立评估（IN_PROGRESS）

G07 仍未通过固定协议。G04–G06 权重和证据保持不变。所有新增真实击球结果都来自 GPU0 上的 Isaac Lab/PhysX 原生刚体梭球、真实关节执行器和固定 1.524 m 球网；没有注入冲量、传送球位姿、降低球网或修改硬件。

这轮先修正了旧诊断：0.725 m 是旧 ±0.12 rad 动作带的采样结果，不是机械工作空间上限。完整关节范围采样的球拍中心最高 z=1.253 m（`reports/G07/full_arm_range03/metrics.json`）；G04–G06 默认动作范围仍为 0.12 rad，G07 高臂分支才使用 2.0 rad 的有界位置命令。

高臂 PhysX 教师在 `reports/G07/search_attack_cfg05/metrics.json` 中达到 64/64 空中接触、64/64 攻击（速度至少 8 m/s、出球垂直速度不大于 −1 m/s），最高 8.202 m/s、最低 −7.948 m/s。加入所有状态的 BC/DAgger 训练后，`reports/G07/train_high_attack_all01/checkpoint.pt` 的最佳训练轮达到 64/64 攻击。

最新冻结独立 400 回合结果在 `reports/G07/eval_high_attack_all01`，并同步到本目录的 `metrics.json`、`episodes.csv` 和 `eval_manifest.json`：

- `sample_count=400`
- `ground_init_fraction=1.0`
- `airborne_contact_rate=1.0`
- 接触后恢复率 `0.8925`
- `complete_smash_success_rate=0.0`
- `numerical_failures=0`

因此接触、攻击和恢复条件已经达到阈值，但越网/落点仍为零。最新测量轨迹（`reports/G07/root_probe_all01/trajectory.json`）显示接触约 z=1.1 m，球在接触后先落地再向前弹跳；典型出球为 vx≈0.45、vy≈−1.7、vz≈−8.2 m/s，不能在固定网高下形成合格越网。方向搜索进一步找到 vx≈4.7–5.0、|vy|<0.2、速度约 9.3 m/s 的候选，但同样因接触高度和向下速度在网前落地；固定候选的完整轨迹见 `reports/G07/fixed_candidates01`。

2026-10-23 更正：撤回“1.535 m 保守硬件上界”的表述。1.253 m 是有限随机关节采样最大值，0.147 m 是已有跳跃搜索结果，均不能证明硬件极限。任意姿态盒体竖直半延伸为 `sum(abs(R[2,:])*half_extents)`，其旋转无关上界约 0.15574 m，并非 0.11 m。现有失败只能说明这些轨迹未越网。新的 CPU 运动学、有效拍面与飞行联合分析见 [feasibility_v3_20261023/REPORT.md](feasibility_v3_20261023/REPORT.md)，不构成新的 PhysX 或策略成绩。

当前可查看的视频是 [`video/G07_current_policy_measured.mp4`](video/G07_current_policy_measured.mp4)，由最新策略的实测梭球/球拍位置渲染；它是当前效果记录，不是通过证据。旧高臂视频和 G06 接触视频继续保留。

G07 状态保持 `IN_PROGRESS`；没有创建 `artifacts/G07/policy.pt`，也没有把未满足完整成功率的结果标为通过。

旧资源窗口已到期并保留历史账本。2026-10-23 已收到新的独立授权并恢复实际运行，见下文。

## 2026-10-23 新授权与有效越网进展

新的独立窗口为 Asia/Taipei **10月23日16:04:13 至 10月24日02:04:13**，最多新增10 GPU-hours，仅GPU0。此前用量不清零，也不计入本轮10小时上限；重启和阶段切换不重置截止时间。实际限制已写入 `configs/budget.json` 与 `status/GPU_LEDGER.json`。

已运行完整负载的全身动态到达搜索，随后在原标准网中进行真实碰撞优化。无球到达搜索只是课程，没有球网或击球成绩；真实击球搜索使用固定原资产、执行器和物理步长0.25ms。课程机器人从近网地面 **root x=1.0m** 开始，因此以下结果不能替代原始root x=0的正式G07。

`dynamic_strike_refine01` 的8代、每代64回合搜索找到满足全部单回合事件条件的教师动作。冻结该控制器后，另一个原生进程和种子714501独立复现64回合，结果见 [`dynamic_strike_replay01/metrics.json`](dynamic_strike_replay01/metrics.json) 与逐回合数据：

| 条件 | 独立近网课程复现 |
|---|---:|
| 实际腾空有效拍面接触 | 64/64 |
| 达到速度与下扣门槛并实际越网 | 64/64 |
| 首次落入目标区 | 64/64 |
| 通过原稳定恢复条件 | 53/64（82.8125%） |
| 全部同回合顺序条件 | 53/64 |
| 数值故障 | 0 |

这些是冻结、可部署的观测/时钟反馈教师控制器结果，**尚未训练成新的自主神经策略**。没有教师前缀、状态瞬移或出球速度覆盖；所用来球是初始物理发射后自由运动。当前64个回合使用同一个物理来球，环境原点的浮点差异不能视为来球覆盖或泛化测试。它既不足400回合，恢复率也未达到85%，且尚未从原始起点进入击球位置。

代表回合（按完整结果选取lane 25）：接触位置 `[1.3972747325897217, 0.02074432373046875, 1.6087727546691895]` m，出球向量 `[7.363097667694092, 0.9775670170783997, -3.0984952449798584]` m/s，总速 `8.04808` m/s；球心在网中心平面的表面净空 `1.557` cm；首次落点 `[3.827606201171875, 0.34340858459472656, 0.025090768933296204]` m，随后稳定恢复。中心平面余量不等同于整个网厚度的保证；实际仿真已包含0.02m厚网体。该回合没有额外身体/球网接触标记或扫掠数值故障。

视频 [`dynamic_strike_replay01/G07_near_net_teacher_replay.mp4`](dynamic_strike_replay01/G07_near_net_teacher_replay.mp4) 使用50Hz实测连杆位姿与球位置、原USD视觉几何离线渲染，包括正常速度与慢放。所展示回合按完整结果选取，不代替统计；网格和目标区颜色为展示用绘制。

目前瓶颈已经从“没有可重复的有效越网”转为 **原始地面起点到近网击球状态的运动衔接、稳定恢复和策略学习/泛化**。第一轮带负载步态搜索只前进约0.26m，尚不够到达所需位置。第二轮保留前后腿负载支撑偏移并加入停止站稳阶段，正在实际运行。有限搜索结果不构成硬件上界。

可复现入口：

```bash
# 所有GPU命令须包在 scripts/run_gpu_job.py，选择本轮授权窗口。
/home/autodrive/Isaac/env_isaaclab/bin/python scripts/replay_g07_dynamic_strike.py \
  --output NEW_DIRECTORY --candidate reports/G07/dynamic_strike_refine01/best_candidate.pt \
  --num-envs 64 --headless
```

恢复入口和运行成本以 `status/NEXT.md`、`status/GPU_LEDGER.json` 为准。G07继续为IN_PROGRESS；原G04–G06与旧G07最佳权重保留，没有发布新的正式通过权重。

截至2026-10-23T08:47:24.726046+00:00，本轮已完成作业实收 `0.494641` GPU-hours；正在运行的作业另有账本预留，最终按实耗结算。
