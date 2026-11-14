# 物理想象链接入：状态合同断点报告

**结论：按“遇到断点停止”的要求，本轮停在物理想象起点的因果状态完整性检查；没有完成完整PhysicalImaginationAdapter，没有更新actor，也没有新的任务成功率。** 不能将CPU测试通过写成完整链通过。

## 版本与冻结输入

以`b7f8d7d`及其父级源码为依据，工作起点是其后代`ce1f70f`，保留`c34280a`新增的原生return和动态触球分母。独立分支`fix/supermimic-physical-imagination-contract-20261114`，实现提交`eed9d3c`。原开发树及未跟踪历史实验保留。

- 初始化策略实际SHA为`8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`，不是按近似文件名选择；路径和所有完整哈希见startup_manifest.json。
- 原PhysicalWorldV2：`aa1a8013613664c96169412ee341d015a5c5771e511778807e3cf3b31c565b49`。
- 拍头监督候选：`3045993066c53142aa337bea3d5f2e10ee28effdbe3201dc1659db5fba1204c0`。
- 球响应：`bee5187824d4da81d2ee834513dfd2906f10c65a7de3d4e47fa8d8b22a558185`。
- 保留G07基底、源head、mixer、持久恢复辅助、原资产/物理、100Hz及原128环境评分。不加入在线MPC。
- 固定记录0.64s想象、0.16s模型窗口；未扩大时域、未调奖励或学习率。
- 已在CPU加载并冻结初始化actor；实际可训练参数白名单为`[]`，没有创建optimizer。将来允许开放的approach参数名称仅记录于actor_freeze_inventory.json，本轮均未开放。

## 具体断点

原始archive的25个分块全部通过SHA校验，800个控制tick、102400条真实转移均检查。五阶段样本数为32813/4370/6226/14557/44434，覆盖准备到恢复，不是0.2秒无触球smoke。

原记录保存了恢复结果布尔值，却未保存`ContactFlightRecoveryV5`的`stable_run`、`longest_stable`、飞行绑定与着地资格全套记忆。原恢复计时按约1ms的指标更新推进；旧archive只有10ms控制边界，无完整子步稳定/接触轨迹。不能声称用已存55维状态和事件8维就能精确补齐。

CPU最小反例直接调用原`ContactFlightRecoveryV5.metric`：两lane拥有相同物理条件、相同“尚未稳定”标志，只有隐藏稳定计时分别为0.799s与0s；同样前进1ms后，stable_seen分别变为true/false。这会影响恢复事件奖励和终端任务链，说明缺失状态确实会改变学习目标。

另一个确定问题是`capture_controller`将所有名为logs的成员省略为“diagnostic logs”，但SharedTargetController.teacher_action实际读取`reference.logs[plan_cursor:]`并更新valid_plan。原始首状态清单含41个对象、505个tensor、6644个value、20个alias、14个omitted项；后续懒初始化会增加字段，不能用张量总数证明完整。网络、bound callback以及alias也尚需完整分支复制验证。

旧记录中某些量可由现有archive/固定资产追溯，如torque、几何、限位，不能把全部新布局缺项都称为永久缺失。见state_dependencies.md逐项分类。

## 本轮实际修改

1. 为物理archive增加明确的辅助状态采集：原jump状态字段、contact-flight恢复计时、airtime、接触再武装、上一命令、球/工具历史、计时时钟、原奖励直接输入和原命令限位。
2. 正确保留真实G07基础规划器的因果logs；其他明确诊断logs仍可省略。
3. 新增严格起点检查及已有archive审计入口。缺少必要状态时抛出明确错误，不填零、不回退旧观测预测器。
4. 添加上述恢复记忆反例和采集隔离测试；保留已有mixer限幅/历史隔离/输入梯度及原生return测试。

这些是采集与接入合同修复，不是完整执行器或完整学习适配器。没有声称旧G07基底已可导，也没有改写真实执行命令。

## 执行结果和停止点

| 检查/阶段 | 结果 |
|---|---|
| 指定CPU测试 | 21 passed，0 failed，0 skipped；退出码0 |
| 语法、diff检查 | 通过 |
| 已有archive起点合同审计 | `BLOCKED_MISSING_CAUSAL_STATE`；退出码2，首个缺项在tick0 |
| 原生新采集接口端到端复验 | 未运行 |
| 旧/新完整执行链同状态命令回验 | 未完成；外层mixer单测不替代它 |
| 完整原生奖励重算对齐 | 未运行 |
| PhysicalImaginationAdapter完整rollout | 未实现/未启用，状态合同未通过 |
| 物理模型forward、actor/critic/WM更新 | 均0 |
| 模型-原生动作敏感性诊断 | 未运行 |
| 小预算actor更新、128环境评价 | 未运行 |

原生策略与模型权重、资产及配置哈希在结束时全部不变，见final_integrity.json。没有在缺失状态上继续训练来追求漂亮的模型loss或return。

## 当前真实return与任务值

本轮没有新的原生交互，因此无新的实时return曲线或成功率提升值。上轮已保存的原生128回合还原结果仍是：return均值 **-12.52678435**，最小 **-91.64870495**、最大 **65.02563964**；合格触球 **74/128**、球侧合格 **53/128**、条件恢复 **34/74**、complete5 **27/128=21.09375%**。这是初始化系统历史证据，不能归功于本轮。

原生return日志代码已保留：native_episodes.jsonl逐回合真实累计奖励，training_history.jsonl/progress.json包含本轮和当前进程累计统计。未结束回合不混入完整均值，条件恢复按本轮实际合格触球动态分母计算。历史54/55/57等MPC辅助结果不作为本轮无新增MPC的算法基线。

## 下一断点解除顺序

下一轮应先验证新增辅助采集在原生环境可用，以冻结初始化策略保存所缺因果状态，不做actor学习；能从旧archive、原生日志或reset前缀重放追溯的字段先复用，不能凭空补齐稳定计时。随后实现可复制/推进/部分reset的完整执行记忆，处理inference_mode及NumPy连续计算的梯度边界；通过命令、观测、奖励合同后，再把同一PhysicalImaginationAdapter用于原/候选模型，验证critic上下文和BPTT链，最后进行预先固定的小更新和原128评价。

**成功率提升目标保留，但本轮增量是未测量，不填0、更不编造正提升。**
