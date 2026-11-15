# 原生回报校准 + 受约束 SuperMimic 更新报告

日期：2026-11-15（Asia/Shanghai）
代码提交：`32488ee0ab3d7b3875cd82725b16f93c1aab0425`
分支：`feat/supermimic-native-physical-loop-20261114`

## 结论

本轮完成了一个固定的两块更新和一次完整 128 环境、8 秒原生评估。

- recovery block：16 次接受更新，0 次拒绝；
- approach block：16 次接受更新，0 次拒绝；
- 训练过程中没有启用在线 MPC、教师接管或视频录制；
- 训练和评估均使用无头 GPU PhysX，物理设备为 `cuda:0`；
- 候选的原生未折扣回报均值为 **-10.03298**，相对初始化基线 **-12.52678** 提高 **+2.49381**；
- complete5 从 **27/128** 提高到 **30/128**；
- 条件恢复从 **34/74** 提高到 **39/74**；
- 但球侧合格从 **53/128** 降到 **52/128**，低于本轮预先设定的保护门槛，因此该候选**不接受为新的正式训练起点**，保留为诊断候选。

本轮结果说明更新在原生回报、恢复和完整成功上有正向信号，但同时损失了一个球侧合格案例；按照事先定义的接受条件，不能把它标记为无条件改进。

## 固定对照和来源

初始化策略 `pi_ref`：

- 文件：`/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt`
- SHA256：`8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- 历史原生回报均值：`-12.526784349736634`
- 历史指标：合格触球 `74/128`，球侧合格 `53/128`，条件恢复 `34/74`，complete5 `27/128`

失败候选 `pi_bad` 仅作为冻结诊断对照，没有续训：

- 文件：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/learning_block_04/candidate_0256.pt`
- SHA256：`b17416c68f7ceb69489c4b33bb8932959a289d3edf7b87039ed5538db0bd1898`

本轮最终候选：

- 文件：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/calibrated_blocks_20261115_1415/candidate_0032.pt`
- SHA256：`9dccdb14a44cd78d3065ddf4887946ae44dfd63901f94efba6103bfb4ee129b0`
- 恢复块检查点 SHA256：`7e9285fd5c2c51071c464a806dd325366eca82a20512d4ae3f6df1c705c70ab4`

## 训练实现和运行证据

本轮代码实现了：

1. 显式 recovery/approach head 白名单和冻结校验；
2. block 起点与最后接受策略的累计变化量记录；
3. 分离记录 `residual_request_delta`、`final_normalized_command_delta`、`final_joint_target_delta_rad`；
4. alpha = `1, 1/2, 1/4, 1/8, 1/16` 的同一提案缩步回溯；
5. 完整最终执行命令预算检查；
6. 基于同策略原生 return-to-go 的 critic 校准报告。

实际运行：

- 输出目录：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/calibrated_blocks_20261115_1415`
- actor 更新：32（recovery 16 + approach 16）
- actor attempts：17（最终 block 的本地计数记录）
- rejected：0
- backtracked：0
- native collection：128 control ticks / 16,384 transitions
- world updates：32
- critic updates：1,622
- peak PyTorch allocated：约 1.166 GB
- 原生采集阶段没有完成 8 秒 episode；原生 return 校准标签来自已验证的 800 tick、128 episode archive。

critic 原生校准：

- 标签策略：`initial_reference`
- 标签数量：128
- 校准更新：2
- MAE：`6.76219`
- RMSE：`7.44599`
- 校准损失：`55.53709`

## 正式评估结果

正式评估目录：
`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/eval_calibrated_blocks_20261115_144745578159346`

运行完成：`800/800` control ticks，`128/128` 首回合记录，退出码 `0`，数值故障 `0`。
GPU 物理：`cuda:0`，`gpu_dynamics=true`。

| 指标 | 初始化基线 | 本轮候选 | 变化 |
|---|---:|---:|---:|
| 合格空中拍线触球 | 74/128 (57.81%) | 74/128 (57.81%) | 0 |
| 球侧合格 | 53/128 (41.41%) | 52/128 (40.63%) | -1 |
| 条件恢复 | 34/74 (45.95%) | 39/74 (52.70%) | +5 |
| complete5 | 27/128 (21.09%) | 30/128 (23.44%) | +3 |
| 原生未折扣 return 均值 | -12.52678 | -10.03298 | +2.49381 |
| 原生折扣 return 均值 | -0.34282 | -0.33179 | +0.01103 |
| 数值故障 | 0 | 0 | 0 |

评估器没有独立的全身头部/非足部接触审计字段，因此 `no_head_contact_recovery` 和 `no_independent_nonfoot_contact_recovery` 保持未知，没有填零。

## 逐案例变化

相对于初始化基线：

- complete5：保留 23，丢失 4（TRAIN17、35、37、55），新增 7（TRAIN11、21、67、77、90、95、97）；
- 条件恢复：保留 30，丢失 4（TRAIN35、37、53、72），新增 9（TRAIN11、22、67、77、86、90、95、97、107）；
- 球侧合格：保留 49，丢失 4（TRAIN17、18、35、55），新增 3（TRAIN11、21、97）；
- 合格空中触球：保留 73，丢失 1（TRAIN40），新增 1（TRAIN107）。

因此，complete5 的净增主要来自恢复与部分球侧案例的重新组合；它不是单纯保持全部旧成功后再增加。该逐案例退化是本轮拒绝候选的直接依据之一。

## 文件和复现证据

训练输出：

- `training_completed.json`
- `native_critic_calibration.json`
- `training_history.jsonl`
- `progress.json`
- `candidate_0032.pt`
- `native_transitions/manifest.json`

评估输出：

- `metrics.json`
- `episodes_summary.json`
- `episodes.jsonl`
- `command_consistency.json`
- `completed.json`
- `physics.json`
- `startup.json`

本轮无视频文件，按要求未录制视频。

## 下一步建议

不要从本候选继续长训。先保留它作为正向但不合格的诊断候选，重点分析丢失的球侧案例和4个丢失 complete5 案例；下一轮若继续，应从已接受的 27/128 策略重新开始，并先修复球侧保护或采用逐案例约束后再做有限更新。
