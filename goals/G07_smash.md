# G07：训练完整扣杀、越网落点和恢复

前置：G06。证据task_id：`airborne_smash_recover_net1524_v1`。
累计预算建议：24 GPU-hours；所有尝试累计环境交互上限：500,000,000（0表示本阶段无正式学习预算）。

从G06初始化。使用真实出球、球网与落点，优化向下出球≥1m/s、总出球速度≥8m/s、
实际越网、目标区域落点，并在机器人恢复和球结果均可判断后结束回合。
主线网高中间1.524m，侧边1.55m；低网debug不得通过本Goal。目标是攻击与恢复而非跳高。
允许训练先使用攻击收益渐进权重，但评测始终同一固定协议。回合超时未判定不计成功。

`python scripts/train.py --stage jump_smash --init-checkpoint artifacts/G06/policy.pt --seed 0 --num-envs 1024 --max-env-steps 500000000 --max-gpu-hours 24 --run-dir runs/G07/seed0`
`python scripts/evaluate.py --stage jump_smash --checkpoint artifacts/G07/policy.pt --suite dev_smash_net1524_v1 --episodes 400 --reset-mode ground --output reports/G07`

完整成功≥50%，空中触球≥70%，触球条件恢复率≥85%；用每个开发面板分别计算，取较差。
如果触球点低于净空要求导致向下球无法越网，做几何/动力学诊断；不得自动降网高、
增大力矩或只选特别容易来球。在真实能力限制下尚未达标须如实报告。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `complete_smash_success_rate` | `ge 0.5` |
| `airborne_contact_rate` | `ge 0.7` |
| `recovery_given_airborne_contact` | `ge 0.85` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G07/REPORT.md`
- `reports/G07/commands.log`
- `reports/G07/episodes.csv`
- `reports/G07/eval_manifest.json`
- `artifacts/G07/policy.pt`

另外生成 `reports/G07/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G07`；通过后才更新STATUS为PASSED。
