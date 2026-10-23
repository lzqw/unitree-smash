# G07：高位高速下扣搜索与独立评估（IN_PROGRESS）

G07 仍未通过固定协议。G04–G06 保持不变，所有新实验使用 GPU0、Isaac Lab/PhysX 原生刚体梭球、真实关节执行器和固定 1.524 m 球网；没有注入冲量、传送球位姿、降低球网或修改硬件。

本轮先纠正了旧诊断：之前的 0.725 m 上限来自 ±0.12 rad 的旧臂动作采样，不是机械工作空间上限。完整关节范围采样得到球拍位置最高 z=1.253 m；旧 G04–G06 的默认动作范围仍保持 0.12 rad，G07 高臂分支才使用 2.0 rad 的有界位置命令。这个修正已写入 `scripts/probe_arm_full_range.py` 和 `quadsmash/tasks/stand/jump_env.py`，不改变执行器、关节极限或 G04–G06 默认配置。

真实 PhysX 的连续高臂搜索已经找到满足攻击阈值的教师动作。`reports/G07/search_attack_cfg05/metrics.json` 使用发射位置 `[8.18,0,1.15]`、速度 `[-6.6,0,6.82]`、1.16 s 换向，在 64 个环境中得到 64/64 空中触球、64/64 速度至少 8 m/s 且垂直速度不大于 −1 m/s；最高 8.202 m/s，最小垂直速度 −7.948 m/s。命中位置约 z=1.07 m，记录的 `crossed_net` 仍为 0，因为固定网顶是 1.524 m。

基于这个教师从 `artifacts/G06/policy.pt` 做了三轮 BC/DAgger：`reports/G07/train_high_attack01/metrics.json`。教师统计为 64/64 空中触球、64/64 攻击，恢复 25/64。学习器检查的最佳一轮为 12/128 攻击样本，仍未形成稳定空中接触策略。

冻结的独立 400 回合评估来自 `reports/G07/eval_high_attack01`，并已复制到本目录的 `episodes.csv` 与 `eval_manifest.json`。结果如下：

- `sample_count=400`
- `ground_init_fraction=1.0`
- `complete_smash_success_rate=0.0`
- `airborne_contact_rate=0.2625`
- `recovery_given_airborne_contact=0.0`
- `numerical_failures=0`

因此当前 G07 学习器没有通过。问题已经从“是否能产生高速下扣”转移到两部分：学习器还没有稳定复现高臂时序；同时命中 z 低于主网顶，向下出球不会满足固定球网的越网判定。继续训练不能把这次独立评估标为通过，也没有创建 `artifacts/G07/policy.pt`。

当前可查看的视频为 [`video/G07_high_arm_measured.mp4`](video/G07_high_arm_measured.mp4)。它来自 `high_arm_traj05/trajectory.json` 的真实球和球拍位置，展示一次约 6.0 m/s、垂直 −5.18 m/s 的高位下行接触；视频是测量轨迹可视化，不是通过证据。旧 G06 接触视频仍保留在 `video/G06_physx_contact_trajectory.mp4`。

G07 状态保持 `IN_PROGRESS`；G06 通过策略和证据保持不变。
