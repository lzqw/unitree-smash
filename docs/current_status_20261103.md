# G07 扣杀训练当前状态（2026-11-03）

## 结论

当前固定 128 例、5 m/s 完整事件链的最佳冻结策略为 **25/128（19.53%）**，在评估 seed `882072` 和 `882075` 上均复现。该结果来自阶段门控恢复头＋paired post-hit teacher guidance 分支，仍依赖基础控制器和持久恢复辅助，因此还不能称为完全独立的全身 RL。

当前最优策略：

```text
reports/G07_full_racket_v5/phase_gated_recovery_v2_env5120_01/policy.pt
```

训练设置为 PPO 学习率 `1e-4`、5120 个并行环境、连续 8 次更新；基础 actor 和 actor normalizer 固定，只训练在公开 `phase774` 激活的恢复头。失败状态使用实际 post-hit teacher residual 作为配对目标，guidance 学习率 `5e-6`、每轮 4 步、anchor `4`。

## 当前指标

| 指标 | 当前最佳冻结结果 |
|---|---:|
| 完整成功 | **25/128（19.53%）** |
| 合格空中触球 | 73/128 |
| 触球后条件恢复 | 39 |
| 球侧目标记录 | 61 |
| 数值故障 | 0 |
| 固定评估 seed | 882072、882075 |

相对此前 23/128 基线，完整成功增加 2 个、条件恢复增加 5 个，并在两个固定 seed 上一致复现。该提升是可靠的开发回归改进，但仍远未达到稳定高成功率。

## 已验证的路线判断

1. **实际配对教师目标有效。** 用统一 zero residual 监督失败状态会退化到 8/128；改为与学生动作坐标一致的真实 post-hit teacher residual 后达到 19/128；进一步降低 guidance 步幅并提高 anchor 后达到 23/128。
2. **阶段化恢复头在冻结 actor normalizer 后继续提升。** 初版只冻结网络权重、继续更新 actor normalizer，其实际 complete5 为 18/128；冻结 normalizer 的 v2 为 25/128。此前把初版旧 complete8=0 写成 complete5=0 属于指标误读，现已从原 episodes.json 重算更正。冻结 normalizer 保证基础动作映射固定，成绩差异仍来自不同训练运行。
3. **共享 actor 继续长周期 PPO 会破坏已有组合。** 从 23/128 策略继续 8 次更新降到 16/128，仅 4 次短续训也降到 9/128。因此当前 25/128 checkpoint 必须冻结保护。
4. **仅扩大 success retention 不足以解决退化。** retention21 续训为 18/128，retention23 续训为 16/128；成功回放需要和更新范围控制、阶段化恢复学习分开设计。
5. **恢复仍是主要可修复瓶颈，但不是唯一瓶颈。** 当前已有不少合格触球后未完成恢复；同时仍有大量回合没有合格空中触球。只修复落地不能直接把成功率推到高水平。
6. **从 25/128 v2 继续 4 次低学习率更新没有继续提升。** 续训固定评估仍为 25/128，但条件恢复降到 36，因此当前 v2 checkpoint 继续冻结，不再沿共享恢复头盲目续训。

## 当前技术路线

```text
冻结 25/128 最佳策略
        ↓
冻结基础 actor 与 actor normalizer，保留基础控制器与正式 5 m/s 事件判定
        ↓
从真实地面回合记录学生动作、教师修正、最终执行和结果
        ↓
只在因果确认的击球后分离/下降/着地阶段学习恢复纠正
        ↓
用有效配对纠正目标训练 phase774 阶段局部 residual
        ↓
关闭额外教师做冻结评估，确认学生是否真正学会
        ↓
恢复模块稳定后，再逐步联合微调起跳和击球前动作
```

下一步不再继续做共享 actor 的学习率、blend 或 retention 数量扫描。优先建立独立恢复模块或阶段局部更新，保护当前有效击球前缀；训练和评估仍从地面真实执行，不传送到理想空中状态，不提前支付完整成功奖励。

## 主要风险与验收标准

- **教师辅助依赖：** 25/128 不是无教师纯 RL 结果。必须同时报告教师开启、额外教师关闭、以及明确移除相应恢复辅助后的结果。
- **指标交集：** 不能把空中触球、球侧目标和恢复数量跨回合拼接为完整成功；下一版继续输出同一回合交集。
- **策略漂移：** retention 的 anchor 不能替代旧最佳策略保护；需要监控 PPO 加监督更新后的总策略变化，并保留旧最佳回退点。
- **归因完整：** 记录首次非足部接触的部件、时刻、触地前姿态和角速度，区分落脚、收拍和更早起跳状态问题。
- **可复现性：** 固定 128 例是开发回归面板，不是泛化证明；所有新结果都要记录配置、策略来源、seed、教师模式和数值故障。

## 关键代码与结果索引

- `quadsmash/rl/failure_recovery_guidance_v1.py`：阶段化失败恢复 guidance 更新。
- `scripts/build_failure_recovery_guidance_v1.py`：从真实失败回合构建 causal phase774 数据集。
- `scripts/train_shared_target_ground64_success_retention_v1.py`：success retention 与 paired guidance 训练入口。
- `scripts/evaluate_shared_target_ground64_persistent_recovery_fasttelemetry_v1.py`：固定面板冻结评估入口。
- `quadsmash/rl/phase_gated_recovery_v1.py`：冻结基础 actor、只在 phase774 训练恢复头，并冻结 actor normalizer。
- `configs/g07_shared_target_ground64_attack_recovery_v19_phase_gated_recovery_frozen_normalizer.json`：当前最佳训练配置。
- `docs/phase_gated_recovery_v2_result_20261103.json`：当前最佳紧凑结果。
- `docs/experiment_report_20261101_current_status.md`：完整实验历史、失败对照与路线审查。

模型权重、视频、轨迹和大型运行日志继续保留在本地，不上传 GitHub。

## 完整 v2 基底上的接近学习修订

逐案例审计确认 v2 保留了此前全部 23 个完整成功，新增 TRAIN23、TRAIN61，未丢失旧完整成功。首个同时训练接近头和恢复头的候选只有 17/128：旧 25 个成功保留11、丢失14、新增6；原55个未触球案例新增11次合格触球，其中仅2个完整成功。该候选不是“恢复头冻结”的对照，不替换 v2。

下一版 `FrozenV2ApproachActorCriticV1` 冻结 v2 的 actor、normalizer、恢复头和 log_std，仅训练新接近头。用足端接触历史及球拍—球距离连续交接，首次触球标志变化本身不会切断新头。使用重新验证的25条完整成功（20,000行、最大指令回放误差1.68e-5）保留集；移除不作用于接近模块的 phase774 failure guidance。7次 critic预热覆盖完整8秒回合，之后在同一连续16更新作业里开放新头。此修订尚无冻结成绩。

代码检查：13项针对性检查通过，含非单位输出尺度、全部phase导出一致性、冻结旧恢复头及normalizer、新头零初始化、因果连续交接和条件高斯似然。99,624条实际公共观测上的完整v2初始映射及JIT导出最大误差均为0。
