# G07 强策略适配器原生 smoke

日期：2026-11-14  
代码提交：`4f29125`  
初始化 checkpoint：`8f5109fe218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`

## 运行条件

- Isaac/PhysX：`cuda:0`，GPU Dynamics 开启；
- 2 个环境；
- 1 个训练迭代；
- 原生控制步：2（每个环境各一个 transition）；
- 0.01 s 控制周期；
- G07 执行源：`/home/autodrive/LZQW/unitree_smash`；
- 没有启动 128 环境正式评估。

## 结果

环境构造、仿真启动、reset、第一批原生动作和训练循环均完成，进程退出码为 0。
适配器实际生成了有限的 18 维最终动作，并进入原生执行链。

| 项目 | 结果 |
|---|---:|
| 原生步数 | 2 |
| replay 转移 | 2 |
| world-model 更新 | 0 |
| actor 更新 | 0 |
| critic 更新 | 0 |
| native reward mean | -0.0124777 |
| reward positive fraction | 0 |
| GPU peak allocated | 1,652,356,608 bytes |

没有完成 16 步连续窗口，因此 world model 报告等待：
`Need 16 consecutive transitions per lane`。这属于本次最小 smoke 的预期数据不足，
不是算法失败；本轮也没有评价任务成功率。

## 结论

历史强策略已能通过新适配器进入当前 SuperMimic 原生环境。下一步可以用同一
checkpoint 做更长的原生数据采集，确认 world-model 更新和 actor/critic 图隔离，
然后再进行零更新 128 环境对照。当前 smoke 不构成历史 27/128 的复现，也不构成
SuperMimic 性能提升证据。
