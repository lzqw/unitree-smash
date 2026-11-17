# 双 Head SuperMimic：执行合同与可信训练接口修复

本轮仅代码修改和 CPU 合同测试。没有运行 Isaac、GPU 训练、正式评价、原生动作扰动或任何消融实验，没有更新真实策略/模型 checkpoint。

实际基础 HEAD：`0d233c01920aa2d21a55cd3b1be085d2a38ed389`。在上一轮本地 217 测试补丁上继续修改，没有回退工作树。尚未创建新 Git 提交或推送远端；不能将远端旧 HEAD 当作本轮源码。`manifest.json` 保存本地文件 SHA256 和解释器；`cumulative_changes.patch` 是相对该 HEAD 的累计代码补丁，包含上一轮未提交修复。原有历史报告删除、未跟踪资产和无关录制脚本未清理。

## 1. 已核实并修复：训练与评价门控不一致

原训练真实交互使用 `had_contact & released`；原正式评价调用默认 Actor forward。两者在未击中后触地等状态下不保证等价。

新增 `supermimic_policy_execution_v1.py`，训练、正式评价、录制部署入口统一调用 `policy_forward()`，想象 Actor 与 Anchor 参考也复用相同门控函数。函数只计算原归一化残差请求；G07 base、mixer、位置目标映射、机械臂限速和 PD 仍沿原执行路径运行。

- 新物理学习 checkpoint 显式记录 `policy_execution`；评价输出独立 `policy_execution.json` 并写入 metrics。
- 缺失、错误形状或非有限任务锁存会报错，不回退到阶段门控。
- 已知旧物理学习 checkpoint 根据确切 adapter schema 识别其训练时的触球分离合同，并记录“推断旧训练合同”的来源。其他旧策略和仅初始化导出仍保留原门控。
- 旧评价文件及其成绩没有改写。新入口重新评价旧物理 checkpoint 时，不能声称是在复现旧评价入口的成绩。
- 新周期原生晋升除核对 checkpoint SHA256 和原数值门槛，还核对评价动作合同。
- RTG 标签的策略身份增加执行合同。即使权重哈希一致，门控不同也不标为同策略标签；旧原生归档没有门控元数据时明确为未知。

CPU 测试用真实 Actor 包装类及原 mixer/限速实现验证相同状态、相同历史下最终 18 关节目标的一致性，容差保持 `2e-6`；包含未触球、未分离、已分离及部分 reset。测试不包含真实 PhysX 接触求解或完整 G07 原生轨迹复验。

## 2. 分 Head 的 WM 验证接入 Actor 更新前

新增 `supermimic_dual_head_quality_v1.py`，扩展 `supermimic_wm_validation_v1.py`。

- Approach：以真实触球/拍框事件附近窗口检查球速度、方向等误差。
- Recovery：独立取首次足端接触、支撑丢失、body-ground 和稳定事件窗口，检查角速度、足端接触、身体触地等误差，不再把球拍接触验证当成 Recovery 质量检查。
- 验证使用记录最终命令的自回归预测；未来真值只作选窗和监督目标，不作模型未来输入。
- 必需事件缺失不能通过；其他缺失类别单列 `missing_events`。目前 Recovery 的必需类别是 touchdown，缺失的失稳/稳定类别仍为未知，不能由通过 touchdown 门槛推断这些类别也已准确。
- 阈值来自事先校准的 receipt，不在代码中随意填值或自动放宽。大规模 profile 要求两个 Head 的原生动作响应证据、各自 WM 上限和 Critic 上限。
- 有效双 Head receipt 也可在 bounded profile 启用相同保护；没有配置时明确记录 `enforced=false`，不伪装成已保护。
- 每次真实采集、WM 更新之后，Actor BPTT 之前执行检查。失败保存诊断和断点并停止，不自动追加 WM 更新“追 loss”。

已有接触证据校验仍检查源模型、任务合同和证据文件哈希；新 `dual_head_quality` 包括：

| 字段 | 语义 |
|---|---|
| `heldout_lane_ids` | 整 lane/完整回合留出，避免重叠窗口泄漏 |
| `native_action_probes.recovery/approach` | 各 Head 的同前缀重复性、成对样本数、实测误差及预先声明上限 |
| `wm_upper_bounds.recovery/approach` | `预测步数/误差名` 映射；Recovery 至少角速度、足接触和 body-ground，Approach 至少出球速度和方向 |
| `critic_upper_bounds.recovery/approach` | 完整冻结回合折扣 RTG 的 MAE/RMSE 上限 |

本轮没有生成任何实测通过 receipt。证据文件校验并不独立证明实验质量；已有模型训练是否使用过留出轨迹仍需原始训练清单证明，后续数据排除不能撤销历史泄漏。

## 3. Critic 独立原生保留集

新增完整原生价值样本包：紧凑 critic 输入、完整同策略 RTG、`(lane, episode, tick)`、Head 实际门控 mask、policy_id、Actor 哈希与执行合同。

- 初始化归档使用已有当前状态缓存的多个任务时刻构建验证样本；不把单 tick0 拟合描述为全阶段训练。
- 两个 Head 的冻结回合采集均保存紧凑上下文及门控 mask。校准网格包含原 Critic/Approach/Recovery 缓存时刻，覆盖异步 lane。
- 原生 Critic 拟合排除保留 lane；WM 拟合、Actor 选择及想象 GAE 监督也排除这些 lane。仅包含保留 lane 的动态起点显式退回可用训练起点，不能导致空 Critic batch 被误报为有效更新。
- 校验每个 Head 的 MAE、RMSE、偏差及同起点时刻的 RTG 排序，报告有效样本和身份。无对应 Head 样本时标缺失，不填零。
- 用冻结策略的 RTG 衡量的是该标签策略的价值校准。更新后策略哈希/门控不同时记录 `current_policy_matches_labels=false`，不把旧 RTG 当作新策略精确价值或价值下界。
- 完整价值样本包随 checkpoint 保存，冻结刷新后替换成对应新策略的数据。resume 不再无条件重新执行旧 π0 tick0 原生拟合。

新输出包括 `dual_head_quality.jsonl`、`latest_dual_head_quality.json`、各 Head 参考覆盖文件；TensorBoard 增加分 Head WM/critic 误差和通过状态。

## 4. 两个 Head 的 Tracking 覆盖与学习路径

Recovery 与 Approach 继续共享 WM/Critic，各自使用独立 Adam，Base/G07/normalizer 保持冻结，原 0.64 秒 BPTT、0.16 秒 WM 窗口、奖励、评分和执行权限不变。

参考覆盖按 Head 记录总条数、独立评分通过条数、Tracking 权重、匹配 lane 和有效权重。某个 Head 没有参考时允许纯任务 BPTT，但标记 `task-bptt-only`；有参考也不等于本次匹配有效，另记录 `tracking_applied`。没有新增 preserve/outgoing 奖励或改变优化公式。

新增 CPU 小网络 `Recovery→Approach→Recovery` 测试，调用实际 learner 的 BPTT/接受逻辑，检查两个 Head 均有任务梯度和最终命令变化、另一 Head 与模型冻结、各 Head Adam 步数延续、native reference 不移动。它不是 128 环境真实训练，也不是 64 步真实机器人闭环验收。

## 5. 验证结果与限制

命令：

```bash
CUDA_VISIBLE_DEVICES='' \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic*.py -q -ra --maxfail=1
```

最终 **228 passed，0 failed，0 skipped，退出码 0**。有一条既有 `torch.distributed.reduce_op` 弃用警告。另完成 21 个变更 Python 文件语法编译、两个 shell 脚本语法检查和 `git diff --check`。

开发期间 CPU 检查发现并修复：新增测试及录制适配器引用了不存在的 Actor 类名（改为实际 `ResidualPPOInitializedActor`）；旧 Tracking 测试替身没有新增参考元数据（补齐其测试合同）。没有安装依赖、删除断言、扩大数值容差或更新真实权重。最终日志保存于 `pytest.log`。

**仍未解决或未执行：**

1. 球响应模型在固定当前状态时，对本步最终关节命令的直接导数仍为零。本轮没有凭空换球模型，也没有把新质量门槛当成模型结构修复。
2. Recovery 的软接触预测与 PhysX 足端力/角速度响应、Approach 的球拍碰撞响应均未做原生扰动验证。现有 body-ground/足接触标签也不是独立部件配对力审计。
3. 未执行真实 `R→A→R`、冻结 128 回合、GPU BPTT 或正式评分；没有新的 return、成功率、预测精度、吞吐或显存成绩。
4. 历史 π0/候选评分的执行门控可比性需要单独核验。本轮不重写历史 27/128、74/128 等结果，也不以这些 CPU 测试证明旧评分可直接用于新合同下的算法比较。

本轮到代码、CPU 测试和本地报告交付停止。没有运行消融，也没有自动开始训练。
