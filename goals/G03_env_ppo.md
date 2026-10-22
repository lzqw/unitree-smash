# G03：实现统一任务、训练评测入口与证据链

前置：G02。证据task_id：`environment_ppo_smoke`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：2,000,000（0表示本阶段无正式学习预算）。

实现TRAINING_SPEC中的Direct任务、统一18维动作/观测mask、真实事件机、正确自动reset、
PPO包装器、train/evaluate/stop_run入口和预算检查。32环境运行至少2000策略步，涵盖
部分reset、超时、失败、触球事件、羽毛球未落地但机器人已恢复等情况。
PPO至少10次更新验证梯度有限、checkpoint保存/加载；测试中断恢复和正常终止bootstrap。
固定dev验证规则、采样种子、目标分布版本及未来holdout生成规则。
所有训练CLI接收--stage/--seed/--num-envs/--max-env-steps/--max-gpu-hours/--run-dir，
评测接收--checkpoint/--suite/--episodes/--reset-mode/--output。配置缺失应fail-fast。
实现从原始CSV自动生成evidence.json，记录所有配置、代码、资产与权重哈希。

`python scripts/smoke.py --num-envs 32 --policy-steps 2000 --output reports/G03`
`python scripts/train.py --stage smoke --num-envs 32 --seed 0 --max-env-steps 2000000 --max-gpu-hours 0.5 --run-dir runs/G03/smoke`

不要把本阶段随机策略reward上升解释成学会动作。默认预算授权到此结束。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `env_contract_ok` | `eq True` |
| `partial_reset_isolation_ok` | `eq True` |
| `timeout_bootstrap_ok` | `eq True` |
| `checkpoint_resume_ok` | `eq True` |
| `evaluation_protocol_frozen` | `eq True` |

## 必须交付

- `reports/G03/REPORT.md`
- `reports/G03/commands.log`
- `reports/G03/interface_contract.json`
- `reports/G03/evaluation_manifest.json`

另外生成 `reports/G03/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G03`；通过后才更新STATUS为PASSED。
