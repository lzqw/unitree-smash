# Approach + Recovery 联合更新评估

日期：2026-11-15
代码提交：`c135262`（训练前联合更新实现）
最终工作树提交：待本报告提交后记录

## 结论

本轮从已接受的 27/128 初始化策略开始，两个 head 使用同一个 Adam、同一次 BPTT backward 联合更新16次。16次全部接受，0次拒绝，0次回溯。

正式128环境评估完成，但候选不接受：

- 原生 return 从 `-12.52678435` 提升到 `-10.48518418`；
- 球侧合格从 `53/128` 降到 `52/128`；
- complete5 从 `27/128` 降到 `24/128`。

因此虽然 return 提升，联合候选违反球侧和完整任务保护门槛，不能作为新的训练起点。

## 训练

- 初始化：接受的 pi0，源 checkpoint SHA256 `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`；
- joint actor 更新：16；
- actor attempts：17；
- rejected：0；
- backtracked：0；
- world updates：16；
- critic updates：982；
- native ticks：64；
- 无在线MPC、无教师接管、无视频；
- base、normalizer、物理模型保持冻结；
- approach 和 recovery 参数均有非零梯度与参数变化；
- 0.64秒 imagination、0.16秒 world window、原始奖励和物理条件保持不变。

候选 checkpoint：

- 文件：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/joint_update_20261115_164125142688214/candidate_0016.pt`
- SHA256：`62fb6844da2a57c716909bf63e040cfc9df3525196dccbacafc6aa5d2c52b6ff`

## 正式评估

评估目录：
`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/eval_joint_update_20261115_170555691662292`

| 指标 | pi0 | 联合候选 | 变化 |
|---|---:|---:|---:|
| 原生未折扣 return | -12.526784 | -10.485184 | +2.041600 |
| 合格空中触球 | 74/128 | 73/128 | -1 |
| 球侧合格 | 53/128 | 52/128 | -1 |
| 条件恢复 | 34/74 | 35/73 | 分母变化 |
| complete5 | 27/128 | 24/128 | -3 |
| 数值故障 | 0 | 0 | 0 |

独立头部/全身非足部接触审计字段仍缺失，相关指标保持未知，没有默认填零。

## 逐案例变化

相对 pi0：

- 合格触球：保留73，丢失 TRAIN90，新增0；
- 球侧合格：保留50，丢失 TRAIN32、35、90，新增 TRAIN26、40；
- 条件恢复：保留31，丢失 TRAIN17、35、55，新增 TRAIN11、22、65、96；
- complete5：保留23，丢失 TRAIN17、32、35、55，新增 TRAIN96。

相对 recovery-only piR：

- 合格触球：丢失 TRAIN90；
- 球侧合格：丢失 TRAIN32、35、90，新增 TRAIN26、40；
- 条件恢复：丢失 TRAIN17、35、55、61，新增 TRAIN11、22、65、84、96；
- complete5：丢失 TRAIN17、32、35、55、61，新增 TRAIN84、96。

这说明联合更新确实改变了接近和恢复两部分，但没有保护住关键球侧和完整成功案例。不能因为平均 return 提升就接受该候选。

## 评价判定

接受条件：

1. native return 高于 -12.526784；通过；
2. 球侧至少 53/128；未通过；
3. complete5 至少 27/128；未通过。

最终判定：**拒绝候选，停止本轮，不继续从该候选训练，不自动调参重跑。**

## 说明

本轮已把 recovery/approach 从顺序 block 改为联合 block。上一轮 recovery-only 报告的 complete5 集合矛盾已由 `case_sets.json` 更正。尚未把下一版本的五项最小修复清单继续实现，因为本轮目标是在一次联合更新和一次正式评估后停止。
