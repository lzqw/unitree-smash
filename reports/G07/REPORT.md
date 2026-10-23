# G07：高位高速下扣搜索与独立评估（IN_PROGRESS）

G07 仍未通过固定协议。G04–G06 权重和证据保持不变。所有新增结果都来自 GPU0 上的 Isaac Lab/PhysX 原生刚体梭球、真实关节执行器和固定 1.524 m 球网；没有注入冲量、传送球位姿、降低球网或修改硬件。

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

几何审计给出当前瓶颈：球网越过判定要求球心高度大于约 1.549 m（网高 1.524 m 加梭球半径 0.025 m）。满载高跳搜索的机器人根部最高约 0.470 m，球拍全范围中心最高约 1.253 m；在同一轨迹中还要保持向下高速挥拍，现有接触姿态达不到所需越网高度。这个结论来自真实 PhysX 接触和连续轨迹，不能用动作后处理补齐。

当前可查看的视频是 [`video/G07_current_policy_measured.mp4`](video/G07_current_policy_measured.mp4)，由最新策略的实测梭球/球拍位置渲染；它是当前效果记录，不是通过证据。旧高臂视频和 G06 接触视频继续保留。

G07 状态保持 `IN_PROGRESS`；没有创建 `artifacts/G07/policy.pt`，也没有把未满足完整成功率的结果标为通过。
