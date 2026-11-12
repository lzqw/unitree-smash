# PPO-initialized SuperMimic fine-tuning

日期：2026-11-13  
分支：`baseline-9cb7983-restore`  
最终代码提交：`9404fc6`

## 目标与边界

本轮将 SuperMimic 改为从仓库已有羽毛球 PPO actor 热启动，而不是随机 actor。训练没有加载 MPC、教师、BC 或旧恢复头；低层 Isaac Lab/PhysX PD 执行器保持不变。起跳 MPC 研究不混入本轮。

## PPO 初始化合同

- 来源：`reports/G07/train_high_attack_all01/checkpoint.pt`
- 来源 SHA256：`7e3f7cb6dc5d0c70962d4ec6f72f587955a425ecb9ff8db02bbf5077d0db6510`
- 观测维度：794；历史长度：11；动作维度：18
- 网络：`794 -> 512 -> 256 -> 128 -> 18`，ELU
- 关节顺序、动作含义和原生关节限幅映射保持不变
- 旧观测归一化统计冻结
- 零更新最终动作最大绝对误差：`0.0`
- actor 加载使用显式严格键检查和 `strict=True`；没有用 `strict=False` 掩盖未加载参数
- 本轮 `random_initialization=false`，critic 为兼容性不明时的随机重新拟合

## 训练实现

训练命令：

```bash
bash scripts/python.sh -u scripts/train_supermimic_v1.py --headless \
  --output reports/supermimic_ppo_init_train_20261113_400 \
  --iterations 400 --num-envs 128 --batch-size 256 \
  --world-updates 2 --imagine-horizon 8 --replay-capacity 50000 \
  --checkpoint-every 200 \
  --init-policy-checkpoint reports/G07/train_high_attack_all01/checkpoint.pt \
  --device cuda:0
```

结果：

| 项目 | 数值 |
|---|---:|
| 外循环 | 400 |
| 原生环境步 | 51,200 |
| 世界模型更新 | 798 |
| 想象/BPTT actor 更新 | 399 |
| critic 更新 | 399 |
| replay 最终容量 | 50,000，`cuda:0` |
| 总训练耗时 | 668.24 s |
| 单轮平均耗时 | 1.6703 s |
| 原生采集平均耗时 | 1.6516 s |
| 世界模型更新平均耗时 | 0.00690 s |
| 想象/BPTT 平均耗时 | 0.01131 s |
| 数据传输平均耗时 | 0.000113 s |
| 峰值显存 allocated | 520,178,176 bytes |
| 峰值显存 reserved | 559,939,584 bytes |

学习链已经包含多步 replay 窗口、真实终止前状态、survival mask、末端 critic bootstrap，以及 actor 更新时冻结 world model 参数但保留输入梯度。训练过程中没有数值故障；原生训练记录的 complete 事件为 0，说明该 128 环境训练分布没有采到正式完整成功。

## 正式评估

评估命令使用 128 个环境、同一固定 seed、完整 8 秒回合和 `cuda:0` GPU Dynamics：

```bash
bash scripts/python.sh -u scripts/evaluate_supermimic_v1.py --headless \
  --checkpoint reports/supermimic_ppo_init_train_20261113_400/checkpoint.pt \
  --output reports/supermimic_ppo_init_eval_20261113_final_valid \
  --seed 882072 --sweep-audit-iterations 0 --physics-device cuda:0
```

最终运行完成 `800/800` tick，128/128 环境终止，耗时约 1281.45 s。评估使用 GPU PhysX；拍面几何采用 GPU broadphase，CPU exact query 为 0。MPC 和 teacher 介入均为 0。

| 指标 | 结果 |
|---|---:|
| 合格空中拍线触球 | 0/128（0%） |
| 除恢复外球侧完整条件 | 0/128（0%） |
| 恢复成功 | 0/128（0%） |
| complete5 | 0/128（0%） |
| recovery given airborne contact | 0 |
| 数值故障 | 0 |
| MPC 介入 | 0 |
| teacher 介入 | 0 |

完整机器可读结果：`reports/supermimic_ppo_init_eval_20261113_final_valid/metrics.json`。

## 结论

这轮证明了 PPO actor 可以被严格加载，零更新时动作合同保持一致，SuperMimic 的原生采集、world-model 更新和 GPU 想象更新可以稳定运行；但 400 轮 fine-tuning 后的冻结策略没有保留原 PPO 的击球能力，正式成绩为 0/128。问题不是评估没有运行，也不是随机初始化 smoke 策略；本轮明确是训练后 actor 退化。

当前首要问题是学习稳定性和任务回报覆盖：训练中没有正式 complete 事件，world model/imagined return 没有来自成功击球的真实正样本，actor 在 399 次想象更新中逐渐偏离了 PPO 初始化策略。下一轮应优先加入初始化策略行为保留与原生成功回归监控，或显著降低/门控 actor 更新，直到 world model 在包含真实事件的 replay 上可靠拟合；不能把当前 0/128 解释为 SuperMimic 算法在该任务上的最终上限。

