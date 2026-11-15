# SuperMimic 完整刷新后有限扩训验证（2026-11-16）

## 结论

候选 **拒绝**，不升级为 `native_accepted_reference`，不从该候选继续训练。
正式评价完整完成：128 环境、800 tick、8 秒、cuda:0、无头 GPU PhysX。

| 指标 | π0 基线 | 本候选 | 变化 |
|---|---:|---:|---:|
| 原生未折扣 return | -12.5268 | **-13.0741** | -0.5473 |
| 合格空中拍线触球 | 74/128 | **73/128** | -1 |
| 球侧完整（除恢复） | 53/128 | **52/128** | -1 |
| 条件恢复 | 34/74 | **33/73** | 动态分母 |
| complete5 | 27/128 | **24/128** | -3 |
| 数值故障 | 0 | 0 | — |

接受门槛为 return > -12.526784、球侧 ≥53/128、complete5 ≥27/128；本候选三项均未通过（return、球侧、complete5）。独立头部/全身非足部接触标签在该环境中缺失，报告为未知；`body_ground_seen=62` 不能替代独立接触审计。

## 训练执行

- Recovery→Approach 顺序：Recovery 8/8，Approach 8/8；16/16 接受，0 拒绝，0 回溯。
- 冻结 Recovery 刷新：800 tick、102400 转移、world 更新 32、critic 更新 20。
- replay：2400/4096；刷新缺失字段为空。
- 真实混合监控 return：-13.6726（低于 π0，作为退化信号记录；正式结论以冻结候选评价为准）。
- peak GPU allocation：约 1.23 GB；无 MPC、无教师接管。
- 0.64 s imagination、0.16 s world window、原奖励/物理/执行合同保持不变。

## 运行故障与修复

第一次刷新因磁盘满（根分区 100%）失败；第二次修复了压缩管道但 `torch.save` 不支持不可 seek 的 pipe；第三次发现完整 `refresh_records` 导致内存增长后中止。最终版本采用：单 chunk 先序列化到内存后 zstd 压缩；刷新只在内存保留奖励/终止张量和 8 个阶段快照，完整状态保存在压缩 archive。第四次运行完整成功。

## 证据

- 训练：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/refresh_seq8_final_20261116_010229_025344060`
- 正式评价：`/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/eval_refresh_seq8_final_20261116_024733_093515204`
- 候选 SHA256：`0196f4c037dac33c4dd71311ea87b38e20f1add403d5fe247fd9efec5c278ef8`
- Recovery 中间 checkpoint SHA256：`3133d04be6e14fe3a96137e613105c99d2540159c19c5f33eeec552d157d786a`
- π0 SHA256：`8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- 代码提交：`c11478a9cc13f5ae7dc1347b6ec93a6750f8c38c`

该结果不支持继续扩大预算；下一步应优先分解球侧从 53→52、恢复从 34→33 的逐案例首次分歧，并校准刷新后 critic/事件模型，而不是直接增加更新次数。
