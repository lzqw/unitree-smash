# G04：训练完整负载下跳跃和落地恢复

前置：G03。证据task_id：`jump_recover_ground_v1`。
累计预算建议：8 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

无羽毛球，从地面正常初始化。PiPER/工具质量完整保留，初期臂姿态软正则随后减弱。
学习准备、蹬地、总质心升高、四足腾空、腿部落地准备与恢复。先不随机大范围质量/延迟。
目标逐步从短小跳跃到验收要求：起跳后总质心升高≥0.08m、腾空≥0.10s、恢复≥0.8s。
奖励避免全程惩罚竖直速度/倾角。真实关节扭矩/速度限制不随课程放宽。

`python scripts/train.py --stage jump_recover --seed 0 --num-envs 1024 --max-env-steps 200000000 --max-gpu-hours 8 --run-dir runs/G04/seed0`
`python scripts/evaluate.py --stage jump_recover --checkpoint artifacts/G04/policy.pt --suite dev_jump_v1 --episodes 400 --reset-mode ground --output reports/G04`

两个固定开发面板各200回合，记录较差面板成功率≥90%，不能使用空中reset计入此门槛。
若不离地先查实际力矩/负载/动作空间和站姿惩罚；若落地失败查终止时机、恢复奖励和
足端准备；若没有可行起跳证据，先做搜索与负载诊断，不增加力矩或减轻PiPER。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `jump_recovery_success_rate` | `ge 0.9` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G04/REPORT.md`
- `reports/G04/commands.log`
- `reports/G04/episodes.csv`
- `reports/G04/eval_manifest.json`
- `artifacts/G04/policy.pt`

另外生成 `reports/G04/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G04`；通过后才更新STATUS为PASSED。
