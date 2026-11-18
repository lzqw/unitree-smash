# 球响应准入失败后的时序审计与最小门控修复

基底0e417225f2b75c5978e6f72856d0f423d79a31c8，保留原dirty与旧数据。没有新的GPU拟合、Actor更新、Isaac启动、正式评价或阈值调整。

## 新证据

读取已完成原生800tick归档对应packed数据，检查全部112个合格/拍框接触锁存边沿。每个边沿前后4tick内的最大球速自由飞行残差都在offset0；offset−1和+1的最大残差仅约2.38e−7m/s。相邻物理状态最大不连续为0。因此排除了全数据整体错位一拍这个假设；真实接触发生于该控制步，但仍不宣称获得独立子步接触真值。

校准后模型只读检查：旧球分支在接触前一步的平均球拍模式概率为0.93474（保留集0.93199），而动作条件修正的平均支持仅0.11132。接触后平均支持仍为0.10054，可能产生虚假残差。连续161点插值诊断比三点检测多找到2/112个控制步内重叠（保留集1/30），不是整体误差的主要解释。使用真实下一机器人姿态只做oracle归因，从不进入部署、Actor或想象输入。

上一轮机器人关节位置/速度响应方向显著改善，但独立球速响应仍失败；本次进一步确认旧球门控提前预测碰撞与新分支抵消该错误之间的冲突。不能据此断言解冻门控必然修好泛化。

## 代码修复，未拟合

新增显式`--repair-ball-gate`模式。默认关闭，历史运行语义保留。启用后仅新增既有BallResidualV2.gate参数进入同一有限校准Adam；球响应网络、归一化、旧足端接触分支仍冻结并校验哈希。已有机器人body和动作条件球分支维持上一校准参数范围，不扩大网络。

监督取同一训练批次的真实当前状态和真实下一球状态：原生拍线/拍框锁存增量标记racket；与原自由飞行残差一致的转移标记free；不能解释的非自由飞行转移保持unknown并排除门控交叉熵。沿用原response_mode_labels残差阈值（速度0.001m/s、位置0.00001m），这不是新接触判定。unknown不当作free，不稀释有效样本损失。未来事件仅用于监督标签，不进入模型forward。

单独记录门控loss、梯度、实际自由/接触/未知样本数及参数哈希变化。保存新的ball.pt，并使action_ball_candidate.pt绑定其实际SHA256；不得继续将更新后的球模型标记为原bee518权重。原模型文件不覆盖，不能将旧Adam状态冒充新参数组的完整resume。所改公式是WM监督适配，不是更改SuperMimic Actor BPTT、GAE或原任务奖励。

## 检查

`CUDA_VISIBLE_DEVICES='' QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh -m pytest tests/test_supermimic_ball_mode_supervision.py tests/test_supermimic_native_response_data.py tests/test_supermimic_action_ball_v1.py -q -ra --maxfail=1`

结果19 passed、0 failed、0 skipped，退出0。覆盖未知残差不伪装自由飞行、未发球不生成标签、真实接触优先级、有效样本归一化、未知样本零梯度、实际旧球门控CPU参数更新且响应/归一化冻结、非有限与形状拒绝。该CPU小模型检查不能证明GPU拟合或物理准入。修改文件py_compile、实际CLI --help与git diff --check通过。

## 后续预算与停止点

已完成三个球模型候选：初始400次拟合、接触邻域400次拟合、原生响应256次拟合，均保留未晋升。仓库AGENTS.md要求“最多进行预算内三轮有假设的训练修订”。因此第四候选仅准备代码和固定计划，不自动启动。next_fit_plan.json和next_command.txt给出同样256次模型更新、96/32划分、lr2e−4、seed20261118的具体命令；没有Actor或新Isaac运行。需要用户授权追加这一轮有界模型修订后才能执行。

即使下一模型通过，仍需Recovery原生动作响应、独立证据有效范围、Critic和真实R→A→R闭环；没有将它们改写为已完成。总体目标仍是验收通过后128环境256次Actor长训及TensorBoard。
