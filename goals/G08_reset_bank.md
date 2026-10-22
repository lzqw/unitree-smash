# G08：研究分支：物理一致中间状态初始化

前置：G07。证据task_id：`reset_bank_ablation_v1`。
累计预算建议：12 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

从G04/G05 rollout提取物理一致中间状态，不从随机空中摆姿态生成。保留历史、执行器
延迟、目标时间和球状态；不能假设引擎内部接触缓存可无损恢复，优先无接触空中状态。
建立恢复执行与连续执行的数值对照。训练可在episode开始抽取中间状态，最多50%，
末期降至20%或0%；评测全部地面初始化。旧数据不直接混入on-policy PPO更新。
以G05相同初始权重/目标分布/种子和总采样预算做with/without reset bank试验；至少3种子
小规模pilot，采集状态库成本必须计入。此阶段是从匹配起点重训，不是拿G07强策略对比
G05弱策略。若预算不足3种子则未完成对照，保留部分结果。

`python scripts/collect_reset_bank.py --checkpoint artifacts/G04/policy.pt --output artifacts/G08/reset_bank`
`python scripts/run_study.py --study reset_bank --seeds 0 1 2 --budget-config configs/budget.json --output reports/G08`

工程验收要求接口正确、数据与对照完整，不要求一定提升。结果为no_gain也应保留。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `restore_consistency_ok` | `eq True` |
| `ground_only_eval_ok` | `eq True` |
| `on_policy_update_ok` | `eq True` |
| `ablation_report_complete` | `eq True` |

## 必须交付

- `reports/G08/REPORT.md`
- `reports/G08/commands.log`
- `reports/G08/reset_bank_manifest.json`
- `reports/G08/ablation.json`

另外生成 `reports/G08/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G08`；通过后才更新STATUS为PASSED。
