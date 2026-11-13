# SuperMimic 时域 v2 隔离 CPU 测试报告

- 测试提交：a6294990ce99444975b1820050d3a4b024b1d4e7
- 提交说明：Add task-time horizons, BPTT/GAE learner and residual action decoder correction (code only)
- 临时 worktree：/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
- 原工作树：/home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline
- 原工作树分支：baseline-9cb7983-restore
- 原工作树 HEAD：c3ecfd4222d58f24f316dba444c5b4d930ddaed7

## 解释器与模块路径

使用了：

```text
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python
CUDA_VISIBLE_DEVICES=""
bash scripts/python.sh ...
```

三个模块均从本次临时 worktree 导入，而不是原工作树、其他 worktree 或 site-packages：

- /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290/quadsmash/rl/supermimic_timebase_v2.py
- /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290/quadsmash/rl/supermimic_sequence_replay_v2.py
- /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290/quadsmash/rl/supermimic_horizon_learner_v2.py

Python 为 `/home/autodrive/Isaac/env_isaaclab/bin/python`，Torch 为 `2.7.0+cu128`，pytest 为 `9.1.1`。

## 实际时域配置

配置文件为 `configs/supermimic_task_horizon_v2.json`：

- control dt：0.01 s
- episode：8.0 s / 800 ticks
- imagination：0.64 s / 64 steps
- world model window：0.16 s / 16 transitions、17 states
- discount mode：`physical_time`
- effective gamma：0.99498743710662
- effective lambda：0.9746794344808963
- 本轮没有切换到 0.96 s，也没有扫描时域参数。

## CPU 测试结果

执行命令：

```bash
CUDA_VISIBLE_DEVICES="" \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest \
    tests/test_supermimic_horizon_v2.py \
    -q -ra --maxfail=1
```

结果：

```text
33 passed in 1.52s
pytest_exit_code=0
```

- 通过：33
- 失败：0
- 跳过：0（pytest 汇总未报告 skipped）
- 未启动 Isaac
- 未加载或训练真实羽毛球策略 checkpoint
- 未启动 GPU 训练
- 未启动原生评估

测试覆盖时域换算、物理时间折扣、BPTT 起点回报、survival、GAE、终止 bootstrap、序列 replay 边界、microbatch 图隔离、动作解码和 checkpoint 合同等 CPU 合成测试。它不证明旧策略复现、原生执行一致性、物理世界模型精度或训练性能提升。

## 隔离与停止结论

目标提交已固定在完整 SHA，没有合并、cherry-pick 或修改原工作树源码。临时 worktree 测试后保持干净。按本轮范围，CPU 测试通过后在此停止，不自动检查训练入口、不启动训练或正式评估。
