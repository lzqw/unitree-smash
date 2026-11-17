# R8→A8→组合策略刷新：代码与CPU合同验收

本轮状态：CODE_READY / CPU合同通过。没有训练、Isaac启动、完整回合采集或正式评价，没有修改真实checkpoint。

## 来源与改动

实际分支feat/supermimic-native-physical-loop-20261114，HEAD 35df2631ebe8094529b0c39dd688106839aeb376；在上一轮验证合同修复的dirty工作树上继续。旧报告删除、assets/vendor及两个录制脚本原样保留。round_changes.patch是相对本轮开始的增量，worktree_before.patch保存本轮开始时的差异；没有回退、清理、合并或推送。

新增supermimic_alternating_schedule_v1.py，并把同一调度驱动接入真实train_supermimic_physical_block.py。不是只写一个未被入口调用的演示调度器。

默认计划：`(R8 → A8 → Frozen Refresh) × 8`，即Recovery64、Approach64。每个head每周期提案上限128（每小块16），只要求最多64次有效接受更新，拒绝不会无限追加。该2:1提案/目标比沿用原启动器比例。某小块耗尽提案却未完成8次有效更新就停止，不跳到下一head或正式评价，不谎报完成。

每提案继续采集16个真实tick、共享WM更新4次，保持0.64秒imagination、0.16秒WM窗口、原学习率、BPTT/GAE、奖励、评分、动作合同和真实物理。actor白名单只在各自短块开放；Base/normalizer/native_accepted_reference不移动。

`--sequential-layout short-alternating`是新的顺序默认值；`legacy-blocks`明确保留旧大块对照。旧checkpoint缺少短块cursor时禁止静默迁移。`--resume-next-cycle`仍要求匹配的原生验收凭证。

## 一次组合刷新如何回流

- 每组A8结束后，由独立调度器调用一次`refresh_frozen_policy('combined')`。短块内旧Approach每8次刷新及块尾刷新被关闭，因此不重复采集。
- 使用当前两个head与冻结base的同一策略组合。等待各lane合法新回合边界，收集128个完整回合；部分旧回合后缀、额外reset前缀只进入WM，不作完整MC标签。仍使用原800tick任务，实际采集有界最多1601tick，不强行中途reset。
- 原生行记录policy_id、policy_version、stream_id、native_tick及frozen_combination/mixed_training来源。终态结果按同lane/episode匹配，正式评价与训练刷新明确分开。
- 同一批数据写入共享Replay。每次刷新32次WM总预算由Recovery事件分布16次、Approach分布16次组成，更新同一个WM；不是为每head另采一批，也不是把32翻倍到64。
- 20次原生Critic minibatch校准使用当前冻结组合的完整RTG，多阶段且保留训练/验证lane隔离。
- 成功参考库增量仍由同一批原生终态独立评分筛选，两个head都更新覆盖报告；失败与未击中数据保留在WM。
- 刷新结束检查两个head的WM及Critic质量，写明实际证据/缺项。无校准阈值写未验证；配置质量门槛时，任一失败阻止下一短块。Actor在整个采集/WM/Critic刷新前后做哈希一致性检查。
- 完成8组后仅形成一个周期候选，由外层脚本原128×800协议评价一次，并立即执行严格晋升校验。刷新本身不晋升策略。

## 断点与计数

保存计划、当前事件cursor、运行状态、每块target/proposal limit、完成回执、刷新序号及采集/拟合配置。变更这些预算或凭证后，不能当作同一断点续训。

两个head的Adam状态继续独立持久化，短块切换不清空动量、不改学习率；当前短块reference重置，step reference按原逻辑推进，native reference只由正式验收晋升。TensorBoard head累计更新不因R/A切换回零，全局attempt与accepted继续分开。

刷新前保存pre_refresh与latest。中途出错的interrupted checkpoint明确拒绝继续重放部分WM/Critic更新，必须恢复刷新前安全断点，并使用新的输出目录。已经完成的刷新在cursor恢复后不再次采集/拟合。

提案前保留pre_proposal安全断点；已进入但尚未获得完整结果的提案标记in-flight，不把未知中途状态伪装成一次接受/拒绝后继续。恢复真实环境仍从新回合/新数据流开始，不声称精确恢复PhysX接触缓存。

短块checkpoint、起点采样报告均带cycle/pair/head，避免R8重复覆盖同名candidate_0008。

## 两种运行模式（本轮均未执行）

生产模式：`--update-mode sequential --sequential-layout short-alternating --head-block-size 8 --refresh-mode pair`；R64/A64目标，共8次组合刷新。

首次集成诊断：同一入口配置R8/A8目标、各16次提案上限，使用`--refresh-mode per-head`，得到R8→组合刷新→A8→组合刷新。该模式也采集当前两个head组合，不使用两个独立策略。它与生产模式是不同计划，断点不允许静默切换。

长训启动脚本已同步为R64/A64、默认pair，且默认large profile，保留原生动作探针/WM/Critic质量凭证要求。修改配置不会自动启动；本轮没有给这些仍缺失的证据填“通过”。

## 原生开销口径

一个R64/A64周期在全部提案一次接受时：在线提案采集2048tick，8次完整刷新至少6400tick，正式评价800tick，合计至少9248tick。按每head128提案上限和每刷新1601tick上限，对应4096+12808+800=17704tick，不含独立初始化/显式resume warmup。旧8192tick仅统计4周期提案采集的说法已从配置纠正，不能拿它当包含刷新与评价的总量。

## 已运行测试

原解释器 `/home/autodrive/Isaac/env_isaaclab/bin/python`，通过scripts/python.sh，CPU使用CUDA_VISIBLE_DEVICES=''。

```
CUDA_VISIBLE_DEVICES='' QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
  bash scripts/python.sh -m pytest tests/test_supermimic*.py -q -ra --maxfail=1
```

253 passed、0 failed、0 skipped，退出0；一条既有Torch弃用警告。新增11项合同测试覆盖：8/16次刷新计数、尾块与总提案预算、实际入口回调接线、组合策略顺序、两Adam与冻结输入梯度、拒绝停止、恢复游标、刷新事务失败与完成后不重做、配置绑定。既有异步lane完整回合/RTG、Replay隔离和梯度测试一并通过。

另通过Python语法、bash -n及git diff --check。本轮测试只使用CPU合成小网络/状态，不加载或更新真实策略，没有真实任务性能新成绩。

## 仍然不能宣称的内容

上一轮接触附近球速度预测与本步球响应动作依赖的问题仍然存在；本轮没有修改球网络、放宽验证阈值或制造原生扰动证据。当前完成的是调度、共享数据回流和恢复事务的工程代码/CPU验证，**不是训练准入全部通过，也不是真实R→A→R学习改善已验证**。

已停止，等待另行授权和模型可信度问题解决后才考虑原生集成运行。
