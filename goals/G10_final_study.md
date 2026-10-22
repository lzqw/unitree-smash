# G10：冻结评测、多GPU实验与最终报告

前置：G09。证据task_id：`frozen_holdout_study_v1`。
累计预算建议：80 GPU-hours；所有尝试累计环境交互上限：5,000,000,000（0表示本阶段无正式学习预算）。

预注册并冻结比较：统一PPO；+状态库；+事件选择无恢复；完整方法。至少5训练种子，
每方法每种子1000固定holdout回合。必须计入预训练、库采集和可行性数据成本，报告两种
口径：端到端成本，以及共享预训练已给定时的增量成本。记录研究结论，不强求正提升。
未见来球组合、目标、延迟、工具惯量与摩擦分开测试；所有部署actor不能看真实随机化参数。
给出均值、种子间分布/置信区间、条件成功率、攻击质量、安全约束、样本及GPU-hour。
优先多GPU独立种子；单任务稳定后才分布式。每rank独立seed和env，统计global steps，
验证一GPU与多GPU batch规模变化，不承诺线性加速。
做dt减半冻结策略测试；跨引擎机器人验证是可选扩展，不把共用羽毛球代码当作独立验证。

`python scripts/run_study.py --study final --protocol configs/study_frozen.json --budget-config configs/budget.json --output reports/G10`

study_frozen.json需在本阶段启动前生成审核冻结。预算默认80 GPU-hours仅建议上限，
不保证完成全组训练；预算不足时报告未完成，不能减少种子后仍声明正式验收通过。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `training_seeds_per_method` | `ge 5` |
| `test_episodes_per_seed` | `ge 1000` |
| `matched_compute_accounting_ok` | `eq True` |
| `holdout_not_used_for_tuning` | `eq True` |
| `numerical_robustness_reported` | `eq True` |
| `all_registered_runs_completed` | `eq True` |

## 必须交付

- `reports/G10/REPORT.md`
- `reports/G10/commands.log`
- `reports/G10/results.csv`
- `reports/G10/protocol.json`
- `reports/G10/research_conclusion.json`

另外生成 `reports/G10/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G10`；通过后才更新STATUS为PASSED。
