# Approach专项闭环：代码修订与CPU检查

本轮仅修改代码、运行CPU合同测试和语法检查。没有启动Isaac、GPU训练、原生采集、正式评价，也没有更新任何真实策略checkpoint。**没有新的return、成功率或真实WM精度结果。**

基础提交：`86a06c3603e3f603ed9f4f87095161f469280671`。
分支：`feat/supermimic-native-physical-loop-20261114`。
工作树：`/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290`。
本报告与代码按随后收到的上传指令一并交付；交付版本以包含本报告的Git提交为准。上述SHA及code_manifest中的head是测试基底，不是补丁后的交付版本。本轮没有修改原资产、物理、奖励、评分、动作映射、100 Hz、0.64秒想象和0.16秒WM窗口。

## 修复与新增

| 审查问题 | 当前代码处理 |
|---|---|
| Approach沿用Recovery配额 | 两头使用不同阶段配额，共享WM。Approach为uniform24、gate有效32、击球准备24、触球/分离24、失败16、击球后8；每提案默认4次WM更新。 |
| 单lane起点被至少4lane过滤 | gate判断适配有效batch大小；生产动态缓存保留完整128-lane G07图，另附逐lane有效mask，不再盲切G07对象树。 |
| 最近8条缓存偏向最后几个lane | 每阶段最多16个完整快照，保留该时刻全部相应lane；Approach另有真实JIT gate且未触球的专用起点池。Actor起点逐提案刷新。 |
| 无效更新/Adam动量漂移 | 分别计算任务、Tracking、Anchor梯度；gate无效、任务与Tracking无有效梯度时不执行Actor Adam提案。接受还要求最终关节命令实际改变。失败完整恢复Actor和Adam。 |
| 有效lane梯度被稀释 | 按门控及survival选出的有效lane归一化完整BPTT回报；回溯沿用原提案的lane选择权重。 |
| 回溯目标可被Critic更新污染 | 所有alpha使用相同模型、Critic、初态及随机状态，独立branch，重算完整目标；回溯结束后才更新Critic。缩步只改训练head参数，Adam保留原同梯度提案的moments，语义在日志中明确。 |
| Tracking仅有时间mask | 叠加有效lane、当前head gate、survival；Approach还要求尚未触球。无可信参考时Tracking为零，原任务BPTT保留。 |
| 绝对时间参考/无限匹配阈值 | v2参考按相对事件阶段匹配；读取当前G07 reference_controller中的预计触球时间，未规划时使用当前状态阶段匹配。必须有限距离阈值。采用拍球相对位置、相对速度、拍面法向及身体状态，逐步插值在GPU上批量执行。 |
| 参考成功标签不充分 | 构建命令要求原独立评分的完整回合记录与packed数据的lane/episode身份匹配；Approach须球侧合格，Recovery须恢复合格且body-ground为false。未知不当作成功。旧v1参考库明确拒绝。 |
| 新成功不回流 | 每次冻结刷新后，真实合格片段可增量加入参考库；按内容去重、每头最多128条，并记录policy/episode/source身份。 |
| 未定义warmup_ticks | 删除旧warmup依赖，manifest明确记录0。另修复cycle报告路径格式化的Path与百分号运算顺序错误。 |
| 冻结刷新边界与Approach缺失 | Recovery块后、Approach每8次有效更新及块末余数均可刷新。各lane等待真实回合起点，收集首个完整冻结回合；最多1601次控制调用，未完成明确报错。旧回合后缀只供WM，不作完整MC标签。 |
| 冻结结果丢失 | 保存完整紧凑转移、真实RTG、逐lane长度/身份、独立评分episodes_summary、参考库及manifest。冻结采集不等于正式晋升评价。 |
| Replay统计误导 | 采样池分配数、最终事件覆盖、来源数、缺额分开报告；同一回合按stream/lane/episode识别。池内优先不重复，稀缺才重复，fallback去重。Approach缺失元数据计数明确保存。 |
| Replay续接 | Approach续接不再默认丢弃Recovery在线归档；重建tensor数量/写指针不匹配则停止，恢复来源/阶段/门控元数据及stream编号。 |
| 旧启动脚本关闭功能 | 启动脚本默认有限R32→A8、64/16提案上限、一次周期、冻结刷新800、WM刷新32、Critic刷新20；要求显式参考库与Tracking权重。新增配置是启动默认说明，实际CLI值仍写运行manifest。本轮未执行此脚本。 |

真实Approach元数据使用当前观测的JIT gate、资产FK球拍距离、真实接触/分离标志及原5 m/s、正向出球条件。近失球、非法接触、触球后无效出球分别提供覆盖统计。未完成回合不能预判最终未击中；未来标签仅用于离线索引/监督，不输入actor或未来rollout。

任务/Tracking/Anchor梯度范数、任务与Tracking余弦、最终命令变化、目标变化写入日志/TensorBoard。修复训练监测中的条件恢复分母，使用实际完成回合里的合格触球集合；body-ground证据缺失时不填false。

## 离线工具

- `scripts/build_supermimic_task_references.py`：从完整packed数据和对应独立评分构建v2参考库；必须显式指定policy身份与有限匹配阈值。
- `scripts/validate_supermimic_physical_wm.py`：同一起点/记录最终命令下对比初始及候选WM，支持1/4/16/32/64步；按完整lane/轨迹隔离的split manifest选择保留集。
- 验证指标包含姿态、角速度、关节速度、球与拍头位置/速度/法向、出球前向符号、足端接触、首次腾空后接触时刻、body-ground和任务事件。缺少正样本或没有匹配触地事件时输出unknown/null，不填零误差。
- 支持接触、触地、恢复失败、稳定恢复等分组。`contact_ball_side_pending`只表示窗口内球侧条件尚未完成，**不当作最终球侧失败**。
- 固定六臂关节±0.001 rad最终目标扰动诊断当前模型的动作敏感度。这是模型诊断，未重放PhysX接触缓存或真实执行限速，不能称为原生反事实验证。
- 两个新脚本的`--help`入口已在原Python环境通过。没有对真实模型执行验证或拟合。

## 测试与证据

执行环境：`/home/autodrive/Isaac/env_isaaclab/bin/python`，经`bash scripts/python.sh`运行，`CUDA_VISIBLE_DEVICES=""`。

```bash
CUDA_VISIBLE_DEVICES="" \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic_*.py -q -ra --maxfail=1
bash -n scripts/run_supermimic_phase1_long_v1.sh
git diff --check
```

结果：206 passed，0 failed，0 skipped，退出码0；1条已有torch.distributed弃用警告。修改/新增Python文件语法检查、Shell语法和diff检查通过。详见`pytest.log`、`test_command.json`、`syntax.json`。

测试覆盖真实资产FK、相对事件参考、独立评分拒绝优先、参考置信mask、128-lane快照及单lane有效mask、异步reset完整回合RTG、缺失元数据、模型自回归不读取未来标签、实际learner的toy BPTT/Adam回退与命令响应。Toy网络的参数更新不是羽毛球策略训练。

`code_manifest.json`记录全部本轮代码文件SHA256，`changes.patch`含本轮tracked和新增文件。原有4个报告删除状态、未跟踪vendor资产及两个录像脚本未修改。只读进程检查未发现运行中的SuperMimic训练/评价进程。

## 尚不能声称完成的验收

1. 未执行真实128环境G07→64步想象→原生冻结刷新流程；CPU测试不是Isaac执行验收。也没有进行真实策略checkpoint的学习更新。
2. 未运行真实保留数据上的初始/在线WM比较或A/B/C数据消融。新增验证脚本不能证明新数据已经改善模型；split与实际拟合数据的独立性须核对来源。
3. 当前球响应网络仍主要依赖当前几何，对本步命令的直接球修正导数仍可能为零。本轮只新增可检验诊断，不在缺少PhysX扰动证据时盲改网络或权重。
4. Tracking匹配距离、近球0.30 m、gate0.25及窗口4步激活阈值是显式初值，未完成真实数据标定。没有把它们宣称为安全保证；仍使用shadow。
5. 旧packed归档没有JIT gate/分离元数据的部分保留NaN，只能进入可证明的普通动力学采样；日志报告实际缺额。要满足专用事件配额，需从可追溯原始归档补标或采集新数据，不能虚填。
6. 保留完整128-lane执行快照解决了维度风险，但还没有原生显存/吞吐测量。没有声称任意单lane G07对象切片或进程中途精确恢复已通过端到端验证。
7. 独立头部/全部非足部接触标签仍取决于原归档；原body-ground指标不能替代独立碰撞审计。

本轮在代码和CPU检查处停止。R32/A8、WM真实数据验证、128环境正式评价均未运行；没有生成训练完成标记或新的性能成绩。
