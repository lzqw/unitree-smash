# 两个互斥训练提案：本轮不执行

两者均以 `65396a21` 及本轮 CPU 审计为依据，不修改完整载荷、动作/执行器限制、固定评测或成功门槛。当前目标只授权准备工作；下列预算是请求上限，不是已授权用量。原有 `additional_revision_proposal.json` 保留，不能把新方案理解为自动追加训练。

当前保守记账 3.432439632 / 8 GPU 小时，99,685,408 / 200,000,000 次交互。两方案选择其一，不串行自动执行；选择另一个属于新的明确决策。

| 项目 | G04-pure-ppo-low-noise | G04-guided |
|---|---|---|
| 实验身份 | 原 v2c 奖励下的纯 PPO 噪声对照 | 自动教师 + 实际历史观测数据 + BC + PPO |
| 初始化 | 200 更新时的 v2c 权重，重新初始化优化器及 std | 独立训练集的教师数据训练 actor，显式初始化 critic/优化器 |
| 奖励 | v2c 原式，不启用新门控 | 拟使用 qualified-recovery-draft-v1；当前仅旁路草案 |
| 主要变量 | 初始 std 0.15、entropy 0；std 仍可学习 | 数据初始化、真实地面专家前缀及接管课程 |
| 首轮长度 | 最多 50 更新，25 更新审查，不自动延长至 200 | 分阶段，最多 100 更新；每阶段最多 25，受实际物理交互上限约束 |
| 总预留上限 | 0.766667 GPU h / 3,604,736 交互 | 1.966667 GPU h / 21,644,800 交互 |
| 当前实现状态 | 训练入口可表达；新增遥测未做 GPU 验证 | 设计提案；数据写入、BC 和前缀接管训练器尚未实现 |

详细机器可读定义见 [pure_ppo_low_noise_proposal.json](pure_ppo_low_noise_proposal.json) 和 [guided_bc_ppo_proposal.json](guided_bc_ppo_proposal.json)。前者只能回答降低噪声是否足以改善原奖励下的学习，不能据其失败否定所有 PPO。后者同时改变数据、初始化和课程，应单独命名，不能伪装成纯 PPO；它也不是当前已经可启动的训练程序。

## 先做的配对原生诊断（待授权）

准备好的 `scripts/diagnose_g04_rewards.py` 默认只打印计划。原生模式设置 8 组 × 16 环境、400 策略步、最多 51,200 物理环境交互，独立诊断种子 311001，噪声种子 311002。各组使用相同的小关节位置/速度扰动，记录实际初态和观测差异。组别为零动作、已有工程控制器、200/800 更新确定性策略，以及两个策略各自原 std / 0.15 的随机动作。随机对照共享配对标准正态噪声，不更新权重或 normalizer。

每个环境只统计初次回合；早期失败后的自动重置仍计入 GPU/交互成本，但不冒充另一个独立场景。总回报分解、实际裁剪前后动作、原模型 std 与本次执行噪声 std 分开记录。恢复计时逐物理子步观察，奖励/动作与 COM 在策略步记录；不会把自动 reset 后的零缓冲当作终态。

```bash
# 以下命令均未执行；需要新的明确 GPU 诊断授权。
python3 scripts/run_gpu_job.py --goal G04 --max-seconds 360 \
  --env-steps-cap 51200 --output reports/G04/reward_diagnostic_NEW \
  --require-artifact metrics.json -- \
  /home/autodrive/Isaac/env_isaaclab/bin/python scripts/diagnose_g04_rewards.py \
  --run-native --output reports/G04/reward_diagnostic_NEW

# 读取上述真实输出后才画实际 COM、参考、竖直速度和奖励曲线；纯 CPU。
CUDA_VISIBLE_DEVICES='' /home/autodrive/Isaac/env_isaaclab/bin/python \
  scripts/analyze_g04_alignment.py --trace reports/G04/reward_diagnostic_NEW/reward_trace.csv \
  --output reports/G04/reward_alignment_NEW
```

记录的事件链为：确认离地的回合数 → 曾经合格飞行的回合数 → 曾经合格落地的回合数 → 正式完整恢复。条件恢复率分母为曾合格落地的回合数；分母为零时写 `null`，不写虚构的 0% 或 100%。各恢复条件的最长连续窗口单独报告，不能拼接成正式成功。

## A：低噪声纯 PPO 的停止条件与命令

先完成配对诊断并冻结基线统计，之后最多 50 更新；25 更新读取训练遥测及最新完成回合，不因总 reward 上升延长训练。

- 非有限数、std 加载断言失败或预算/截止触发：立即停止。
- 至少 128 个已完成训练回合时，若没有合格飞行且机身触地率超过 50%，停止。
- 若没有合格飞行且训练行为均值越界比例比预先冻结的初始策略诊断增加至少 0.10，停止。该规则是预注册工程停止规则，不是因果检验。
- 无论如何在 50 更新停止，不能自动追加至 200。保留 25/50 的具名检查点；它们由同一次规范重启保存的文件复制，不增加一次 reset。
- 用最后安全检查点进行一次后诊断及原 400 回合正式评测；数值故障先修复后才能评测。训练随机回合统计不替代确定性验收。

```bash
# 待额外训练修订明确授权后才执行。
python3 scripts/train.py --stage jump_recover --seed 0 --num-envs 512 \
  --max-env-steps 3342336 --max-gpu-hours 0.5 \
  --config configs/g04_low_noise_50_proposal.json \
  --init-checkpoint runs/G04/corrected_v2c_main01/native/checkpoint.pt \
  --run-dir runs/G04/pure_ppo_low_noise_50_NEW

# 需要提前停止时，仅处理该作业，并保存退出。
python3 scripts/stop_run.py --run-dir runs/G04/pure_ppo_low_noise_50_NEW --save-and-stop

# 正式复评；已有 evaluate 入口会经过累计预算保护。
python3 scripts/evaluate.py --stage jump_recover \
  --checkpoint runs/G04/pure_ppo_low_noise_50_NEW/native/checkpoint.pt \
  --config configs/g04_low_noise_50_proposal.json --suite dev_jump_v2 \
  --episodes 400 --num-envs 200 --reset-mode ground \
  --output reports/G04/pure_ppo_low_noise_50_eval_NEW
```

后诊断将诊断脚本的 `--comparison-checkpoint` 指向新检查点；`reference` 默认是原 200 更新权重，`comparison` 默认是原 800 更新权重。报告以实际 checkpoint 路径与 SHA 标识两个输入，避免把新 50 更新策略误称为旧模型。

## B：教师引导方案的边界与停止条件

先在独立训练初态上对已存在的 8 参数协调轨迹做有界局部搜索，搜索、采集、验证、BC、PPO、诊断及正式评测分别记账。教师筛选不使用两个正式开发面板或 G10。

数据必须保存 rollout 当时实际的 794 维历史观测与动作，而不是用摘要重建。按完整回合及参数族划分训练/验证，避免相邻帧泄漏。只有完整成功回合可以提供整段恢复目标；合格起跳但恢复失败的回合仅用安全前缀，摔倒后动作不作为恢复标签。站立/稳定辅助样本最多占 BC batch 的 10%。

教师搜索在固定上限内若未收集到至少 4 个参数族、2 个采集批次、合计 16 条完整成功回合，则停止该恢复教师分支，保留失败数据。这个数量是提议的准入门槛，不是当前已拥有的数据。BC 离线误差低仍不能推进：独立闭环验证若合格飞行率比相同教师族降低超过 20 个百分点，或非足端触地率超过 50%，先停止定位问题。

若采用专家前缀，必须每回合从地面真实执行，再逐步将接管时刻从 1.4 → 1.2 → 0.8 → 0 s 提前。每阶段最多 25 更新；至少 128 个独立训练验证回合满足提案中的合格落地与条件恢复门槛后才推进。PPO 只收集策略接管后的 on-policy transitions，不能把教师动作当作自己的采样动作计算策略梯度。前缀失败、前缀耗时与所有物理交互仍计入成本；物理预算优先于目标更新数。中途 PhysX 接触状态恢复不在方案中。

本轮只提交此方法设计。实施前还必须实现和测试数据版本/划分、BC、前缀到 PPO 的历史与终止边界、超时 bootstrap、预算扣减；因此不提供不存在的 BC/PPO 启动命令。最终正式评测仍由策略从 t=0 全程自主控制，撤除全部专家前缀及辅助。
