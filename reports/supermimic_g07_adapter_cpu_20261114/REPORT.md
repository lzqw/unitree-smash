# SuperMimic G07 强策略适配器 CPU 检查

日期：2026-11-14  
代码分支：`supermimic-structured-history-event-metrics-20261114`

## 本轮变更

新增 `quadsmash/control/g07_shared_target_residual_v1.py`，严格支持历史
`g07-shared-target-residual-ppo-v1` checkpoint：

- 校验共享目标 base schema、native execution 标记、资产哈希和外部执行源哈希；
- 加载原始 `native_factory.py` 生成的共享目标控制器；
- 保留 checkpoint 中的恢复参数；
- 使用相同的 18 维残差 mixer 和机械臂限幅；
- 将 learner action 作为残差请求，不重写原始执行接口；
- 将执行源路径记录为 `g07_execution_root`，不使用 `strict=False`。

`SuperMimicEnvV1`、训练入口和评估入口已按 schema 分支接入。原有
`full-racket-residual-ppo-v4` 路径保持不变。

## CPU 结果

命令：

```bash
CUDA_VISIBLE_DEVICES='' \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest \
  tests/test_g07_shared_target_adapter.py \
  tests/test_supermimic_horizon_v2.py -q -ra --maxfail=1
```

结果：`36 passed in 1.81s`，退出码 `0`。

实际检查包括：

1. 历史强策略 `8f5109fe…` 的 schema 和 794/18 actor 合同；
2. G07 外部 factory/core 文件哈希；
3. CPU 构造完整共享目标控制器；
4. 零状态动作生成、有限性和 `[-1,1]` 最终命令范围；
5. 原有 SuperMimic 时域 v2 测试。

## 仍未宣称的内容

本轮没有启动 Isaac、没有执行 128 环境评估，也没有进行 SuperMimic
训练。因此还没有证明新适配器在原生环境中复现历史 27/128。下一步应先用
该适配器做零更新原生 smoke，再决定是否运行正式评估；若零更新不能复现，
应继续检查执行合同，而不是开始 actor 更新。
