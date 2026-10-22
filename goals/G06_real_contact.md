# G06：训练真实来球空中接触并恢复

前置：G05。证据task_id：`airborne_contact_recover_ground_v1`。
累计预算建议：16 GPU-hours；所有尝试累计环境交互上限：300,000,000（0表示本阶段无正式学习预算）。

将虚拟事件替换为从episode开始独立积分的真实羽毛球。通过初始发球条件采样与前向
预测生成经过可行区域的球，不能在接近球拍时修正球的位置、速度或时间。
从G05初始化。先小范围来球，第一次真实接触奖励+恢复奖励；尚不要求攻击性越网。
记录预测误差、接触时间误差、真实拍点速度、冲量与落地结果。

`python scripts/train.py --stage real_contact --init-checkpoint artifacts/G05/policy.pt --seed 0 --num-envs 1024 --max-env-steps 300000000 --max-gpu-hours 16 --run-dir runs/G06/seed0`
`python scripts/evaluate.py --stage real_contact --checkpoint artifacts/G06/policy.pt --suite dev_contact_v1 --episodes 400 --reset-mode ground --output reports/G06`

两个开发面板的空中接触+恢复联合成功率均≥60%。无真实接触的shaping不能计成功。
失败时先区分：虚拟事件执行误差、来球预测误差、物理接触误差；不可把三者都归为PPO差。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `airborne_contact_recovery_success_rate` | `ge 0.6` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G06/REPORT.md`
- `reports/G06/commands.log`
- `reports/G06/episodes.csv`
- `reports/G06/eval_manifest.json`
- `artifacts/G06/policy.pt`

另外生成 `reports/G06/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G06`；通过后才更新STATUS为PASSED。
