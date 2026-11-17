# SuperMimic 大规模训练代码修复（仅代码与 CPU 测试）

基础提交：`0d233c01920aa2d21a55cd3b1be085d2a38ed389`。
分支：`feat/supermimic-native-physical-loop-20261114`。
实际文件哈希、解释器、工作树状态见 `manifest.json`、`git_status.txt`；本轮补丁见 `changes.patch`。

本轮没有启动 Isaac、GPU 训练、原生评估、WM A/B/C 消融，也没有加载更新或覆盖真实策略 checkpoint。CPU 测试中的小网络更新仅验证软件合同。已有历史报告删除和未跟踪资产没有清理，也未包含在本轮补丁中。

## 完成的修改

| 审查项 | 本轮实现与实际边界 |
|---|---|
| WM 来源×事件联合配额 | 无 episode 限额时用最大流；存在限额时按“来源、事件成员集合、episode”分组做整数配额分配，同时满足唯一窗口容量和每回合上限。使用已有 SciPy，不安装依赖。可行时不再被贪心顺序消耗稀缺事件；不可行时显式记录缺额和补样。求解未证明最优则报错，不当成无数据。 |
| 重复与统计 | 全 batch 去重，默认每 episode 最多 8 个窗口。分配类别、真实事件覆盖、真实来源和缺额分别记录。唯一窗口不足时停止，不复制样本伪装配额达标。来源配额在启动时要求合计为固定 batch 128。 |
| Approach 四类起点 | 25/35/25/15 的提案类别调度：gate onset、有效触球前、失败前、成功回球附近。有限容量保存完整 128-lane G07 状态和选择 mask；成功/失败只由同 stream、lane、episode 的真实完成记录事后标注。类别不足显式降级，不能声称实际比例已达标。 |
| 动态起点续训 | 完整快照、mask、类别、策略/数据流身份及独立采样 RNG 写入断点。恢复在线 transition 后单独恢复起点；不再把恢复 WM Replay 等同于恢复 Actor 起点。 |
| Tracking 归一化 | 使用整个 time×lane 的损失加权总和除以有效权重总和。增加有效 lane-step、权重和与非零状态梯度 lane 数。最后一个指标表示 Tracking 对物理状态的梯度，不能代替原生动作导数验证。 |
| 失败回球分类 | 已完成回合中的未越网、未命中目标落点独立标记，归入击球前/触球/分离采样索引。缺失真实字段保留未知；未来标签不进入 Actor 或想象前向。 |
| 冻结回合 Critic 覆盖 | 同一校准时刻保留所有异步到达的有效 lane，而非只有第一批。采集时编码紧凑 critic 上下文，保存 policy、episode、lane、时刻及完整 RTG；有限 minibatch 记录 eligible/used 覆盖。沿用现有校准 tick 网格，不宣称已变成逐物理事件触发的起点网格。 |
| 长期 Replay | π0 初始化数据独立保留，不随在线环形覆盖丢失；在线 ring 有界。按 stream/reset 检查窗口。按 native 追加失效的有序缓存可供每提案 4 次 WM 更新复用。 |
| 断点边界 | checkpoint 固定每个 archive 的 chunk 清单与哈希，恢复时不读入保存断点之后才追加的 chunk。恢复两头 Adam、三种 reference、动态起点、采样 RNG、阶段进度、计数和验证集身份。 |
| 顺序调度 | 有效更新目标与提案上限分开，禁止 `min()` 静默缩小目标；未完成目标停止，不自动切换下一 head。修正提案多加 1，恢复 block/global 计数；完成的 Recovery 不重训，刷新进度单独记录。 |
| 原生晋升 | 新周期移动 native reference 必须提供与 checkpoint SHA256 对应且达到原门槛的正式评价。优化器接受不算原生接受。默认脚本保留“不达标即停止”的行为，不自动反复筛选。 |
| 接触验证接口 | 增加接触事件前后 -4/-1/0/+1/+4 控制步的记录动作验证索引。大规模 profile 核对模型/执行合同/证据哈希与事先声明的阈值；每次提案前检查 held-out 接触窗口，WM 拟合排除指定验证 lane。没有有效证据时不会把模型有限差分当成 PhysX 验证。 |

## 大规模入口与预算

新增 `scripts/run_supermimic_success_guided_large_v1.sh` 和 `configs/supermimic_success_guided_large_v1.json`。脚本未执行。

- 原 128 环境、headless、GPU PhysX、cuda:0；Recovery→Approach，Base/G07/normalizer 冻结。
- 默认 4 个周期，每周期 R64→A64，总目标 512 次有效接受更新。
- 每 head 每周期最多 256 个提案，不无限追逐有效更新数。
- 每提案采集 16 个原生 tick、执行 4 次 WM 更新；WM 0.16 秒、想象 0.64 秒。
- 沿用学习率 5e-5/2e-4/2e-4、原奖励和执行权限；单步执行偏移预算仍为 0.005 / 0.04 rad。累计偏移没有凭空新增硬阈值；未配置时仍为 report_only。
- Recovery 完成后和每 8 次有效 Approach 更新后执行冻结完整回合采集；周期末独立正式评价。完整采集最多等候 1601 tick，记录部分后缀与完整回合的区别。
- **8192 不是包含刷新和拒绝的总采集 tick 数。** 只有 512 个提案全部接受时，提案采集才是 8192 tick；提案采集上限 32768 tick，36 次刷新上限合计 57636 tick，4 次正式评价另计 3200 tick。实际计数按真实执行记录。

启动脚本要求真实接触验证 receipt、经独立原生评分的参考库和显式 Tracking 权重；本轮未生成任何“通过”的验证 receipt。证据校验检查身份、阈值及文件完整性，不能独立证明用户提供的证据在科学上充分。已有模型是否使用过验证轨迹也必须由外部真实训练清单确认，代码中的后续 lane 排除不能撤销历史数据泄漏。

## 续训与资源边界

恢复学习器/Replay/优化器的断点并不等于恢复 PhysX 接触求解器隐状态。重启后原生环境进入新的数据流，不能把重置前后拼成一个连续窗口或冻结回合。中断在刷新内部时，已保存的部分动力学数据保留，未完成刷新会在独立新运行目录重新收集；不会把部分数据生成完整 MC 标签。

Replay 使用 CPU 环形数据加 GPU minibatch，额外保留初始 archive；Actor 起点缓存有限。新增采样整数分配使用 CPU、设有 10 秒求解上限，未进行原生吞吐/峰值内存测量；不能宣称已达到实时或预估训练时长。正式运行前仍需验证实际 GPU 显存和完整入口。

## 没有做、也没有宣称完成的内容

1. 没有更换球响应模型。固定当前状态时，当前球下一步输出对本步关节命令的直接导数仍为零；事件验证工具与启动证据门槛不等于修复这一模型结构或证明间接梯度有效。
2. 没有新增未经验证的 Recovery preserve loss、接触后 outgoing loss 或新球网络，没有改 BPTT/GAE 核心及原生奖励。
3. 没有运行原生 smoke、真实动作扰动、冻结 128 环境评价或消融；没有新的 return、成功率、有效更新率或模型预测精度结果。
4. 各阶段 CPU 测试通过只能证明测试覆盖范围的软件行为。当前不能标记为“完整原生链路已验收”或“可放心无人值守长训”。

## 验证

原解释器：`/home/autodrive/Isaac/env_isaaclab/bin/python`，经 `bash scripts/python.sh` 调用，CPU 测试明确设置 `CUDA_VISIBLE_DEVICES=''`。

```bash
CUDA_VISIBLE_DEVICES='' \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic*.py -q -ra --maxfail=1
```

最终结果以 `pytest.log` 为准：217 passed，0 failed，0 skipped，退出码 0；1 条既有 `torch.distributed.reduce_op` 弃用警告。另完成 16 个变更 Python 文件语法编译、两个启动脚本 `bash -n`、`git diff --check`。

新增/扩展测试覆盖：稀缺事件联合配额、带 episode 上限的可行配额、全局去重与真实缺额、完整执行状态起点及 RNG 恢复、终态标签身份隔离、π0 数据环形覆盖保护、验证 lane 排除、异步紧凑 Critic 上下文与 RTG 配对、Tracking 全局权重、阶段断点、原生晋升 checkpoint 身份和阈值、接触相对时刻与 reset 边界。

早期定向测试发现一次新增测试自身解包 `_ordered()` 的返回值数量错误，已改为实际六项接口；未删除断言或放宽容差。最终日志保存的是全部修改完成后的复测。本轮到代码与报告交付停止。
