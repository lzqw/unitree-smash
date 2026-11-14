# G07 强策略适配器短训练链检查

日期：2026-11-14  
代码提交：`810349d`  
初始化 checkpoint：`8f5109fe218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`

## 运行条件

- Isaac/PhysX：`cuda:0`；
- 2 个环境；
- 20 个外循环迭代；
- 0.01 s 控制周期；
- world window：0.16 s / 16 steps；
- imagination window：0.64 s / 64 steps；
- 无 MPC、teacher、BC 或旧 recovery head 介入。

## 结果

| 项目 | 结果 |
|---|---:|
| 原生步数 | 40 |
| replay 转移 | 40 |
| world-model 更新 | 5 |
| actor 更新 | 4 |
| critic 更新 | 100 |
| 最后 native reward mean | -0.01151 |
| native positive fraction | 0 |
| 最后 imagined return | -2.6269 |
| GPU peak allocated | 1,717,044,736 bytes |
| 退出码 | 0 |

训练链已经实际经过“原生采样 → 连续窗口 world-model 拟合 → GPU 想象 →
critic / actor 更新”。没有生成完整 8 秒回合，因此事件计数为零，不能用于
判断羽毛球成功率。

## 梯度解释

历史 G07 actor 的 base 和 recovery head 是冻结的，只有 approach head 和
`log_std` 可训练。前 0.4 秒中，原策略的空中接近 gate 没有激活，所以日志
中的 actor grad norm 为 0；这反映当前短段没有覆盖 gate 的有效状态，不能
说明 actor 永远无法更新。正式判断必须至少覆盖球出现、接近、击球和恢复事件。

## 结论

强策略适配器已通过真实 GPU PhysX 短训练链。当前没有新的任务成功率，下一步
需要运行足够长的连续原生段，使每个环境进入完整 8 秒回合，再检查 gate 激活、
actor 参数漂移和冻结评估结果。
