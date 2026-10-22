# G09：研究分支：击球与恢复联合事件选择

前置：G08。证据task_id：`recovery_aware_event_selection_v1`。
累计预算建议：16 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

冻结G07低层策略或固定训练版本，采样(state,event_goal,complete_success)。标签必须包含
实际击球+攻击/回球+恢复，失败也保留。按整条episode/场景分组划分训练、校准和验证。
训练小ensemble或单模型校准成功概率，报告Brier/ECE、false-safe、precision/coverage。
沿预测轨迹采样约64候选，使用物理必要条件筛选和可恢复成功概率选择，不强加高层RL。
允许abstain/保守回球并报告覆盖率，不能通过拒绝全部任务获得漂亮安全数据。
比较固定事件、只击球评分、击球+恢复评分；信息与预算相同。低层策略变更后，能力模型
必须刷新或重新评估。概率不是安全证明，不强求本阶段方法必优于baseline。

`python scripts/fit_feasibility.py --policy artifacts/G07/policy.pt --split-by scenario --output artifacts/G09`
`python scripts/run_study.py --study event_selector --seeds 0 1 2 --budget-config configs/budget.json --output reports/G09`

若没有足够正样本或校准不可靠，报告并阻塞在线使用，不把训练集准确率当泛化能力。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `grouped_split_ok` | `eq True` |
| `probability_calibration_reported` | `eq True` |
| `abstention_coverage_reported` | `eq True` |
| `policy_version_matched` | `eq True` |
| `selector_ablation_complete` | `eq True` |

## 必须交付

- `reports/G09/REPORT.md`
- `reports/G09/commands.log`
- `reports/G09/dataset_manifest.json`
- `reports/G09/calibration.json`
- `reports/G09/selector_ablation.json`

另外生成 `reports/G09/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G09`；通过后才更新STATUS为PASSED。
