# G05：训练按时完成空中虚拟击球事件

前置：G04。证据task_id：`virtual_hit_recover_ground_v1`。
累计预算建议：12 GPU-hours；所有尝试累计环境交互上限：300,000,000（0表示本阶段无正式学习预算）。

由G04权重初始化，不冻结腿。仍无羽毛球，给定目标(time,position,normal,velocity)。
目标来自已验证可达区域及动态rollout，不能独立随机采样出大量明显不可能的组合。
先学位置，再方向，再接触点速度与精确时序；训练时间窗80→20ms，位置12→4cm。
先用可行低速目标，后扩大；最终目标分布必须在训练前固定，不按失败情况删难题。

`python scripts/train.py --stage virtual_hit --init-checkpoint artifacts/G04/policy.pt --seed 0 --num-envs 1024 --max-env-steps 300000000 --max-gpu-hours 12 --run-dir runs/G05/seed0`
`python scripts/evaluate.py --stage virtual_hit --checkpoint artifacts/G05/policy.pt --suite dev_virtual_v1 --episodes 400 --reset-mode ground --output reports/G05`

联合成功至少70%：空中满足位置/方向/速度/时间全部容差并稳定恢复。
若连续多次评估没有有效事件，可在本阶段预算内提前实现TRAINING_SPEC的状态库救援，
先完成恢复一致性单元/仿真验证，使用`ppo_resetbank_rescue`独立分支并记成本；最终仍
全部地面初始化，不能将它当作`ppo_plain`基线或宣称G08对照已完成。
若摆到目标却无速度，检查奖励时间窗及速度方向；若臂挥得好却落地失败，检查恢复
任务是否真的参与完整episode；不得永久冻结腿以回避协调问题。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `virtual_hit_recovery_success_rate` | `ge 0.7` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G05/REPORT.md`
- `reports/G05/commands.log`
- `reports/G05/episodes.csv`
- `reports/G05/eval_manifest.json`
- `artifacts/G05/policy.pt`

另外生成 `reports/G05/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G05`；通过后才更新STATUS为PASSED。
