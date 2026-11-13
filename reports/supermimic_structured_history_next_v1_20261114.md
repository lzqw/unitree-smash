# SuperMimic structured-history next version

日期：2026-11-14

## 当前运行审计

正在运行的旧进程仍使用 `a6294990` 临时工作树和旧的 legacy observation-delta world model；本次没有热修改或停止该进程。其最新可见状态约为 3279 次迭代、1,678,848 个原生并行环境步，尚未产生新的冻结 128 环境评估，因此不能确认任务成功率提升。

## 本次代码改动

分支：`supermimic-structured-history-event-metrics-20261114`

提交：`15a105238982023c0ca2c644d4f656fd60425fd8`

### 1. 结构化历史世界模型

794 维观测中的前 759 维是 11 帧、每帧 69 维的历史。新模型只预测下一帧增量和非历史尾部，下一观测历史由与原生 `PolicyHistory.observe` 相同的确定性移位重建，不再让世界模型自由发明整段历史。

公共接口仍为 `(delta, reward, terminal_logit)`，可复用现有 BPTT/GAE learner；旧 legacy world-model checkpoint 不与新结构模型混用。

### 2. 原生任务事件日志

训练历史新增累计原生事件与奖励统计：

- 合格空中拍线触球
- 除恢复外球侧完整条件
- 恢复成功
- complete5
- 拍框触球
- 非足部接触
- rally fault
- 原生奖励均值、正奖励比例和 done 数

这些指标在回合 reset 前从 `rally_records` 读取，避免用 reset 后张量补写旧回合。当前 `body_landing` 只记录为 `nonfoot_contact`，没有冒充 head-specific 标签。

## 验证

使用 Isaac 环境解释器执行：

```text
CUDA_VISIBLE_DEVICES='' \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic_horizon_v2.py -q -ra --maxfail=1
```

结果：`34 passed in 1.51s`。

此外通过 Python 编译检查和 `git diff --check`。

## 解释边界

本提交尚未运行新的 GPU 训练或 128 环境正式评估。它解决的是世界模型历史一致性和任务级可观测性，不能直接宣称成功率提升。下一次训练应使用新提交启动独立输出目录，并在冻结 checkpoint 后做正式评估。
