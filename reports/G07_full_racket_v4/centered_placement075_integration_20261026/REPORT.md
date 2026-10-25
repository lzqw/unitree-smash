# 实际中间比较结束后集成落点反馈0.75试跑

原中间比较已全部完成；1697096、1719390及1719402实际退出，原GPU账本没有RUNNING作业。集成前独立重算五组原全TRAIN128结果，并核对196项原持有源码与输入全部保持。证据见 `BEFORE_INTEGRATION.json`。

| 来源更新 | 完整 | 合法空中拍线 | 条件恢复 | 合格攻击 | 数值故障 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0 | 26 | 13 | 3 | 0 |
| 16 | 0 | 25 | 13 | 3 | 0 |
| 24 | 0 | 21 | 9 | 1 | 0 |
| 32 | 0 | 18 | 9 | 0 | 0 |
| 40 | 0 | 22 | 10 | 4 | 0 |

按已有完整→攻击→恢复→空中→同分较早更新排序，实际选择40版，公共策略SHA256为 `2e1bf93056eff22dd309ea52a777ce0648aa50e025f63ec99e5ae8fe662fd84f`。它在空中接触和恢复上弱于8版；选择依据是预先登记排序，不能宣称整体改善。所有版本都未达到原门槛。

集成代码逐字节来自已上传候选提交 `2523bf5df1d3e08d3af532e1bc7d8f0060f17bef`：实测外层平均动作迁移接口、训练入口、测试、0.75配置及主GPU账本试跑流程。只复制同一固定旧8基线上的外层actor及其归一化；critic、探索、Adam、RNG、学习计数及物理初态重新开始。原actor归一化10485760应另加新训练步数，critic从0开始。

主工作区50项相关CPU检查通过，`--inspect-only`确认原持有进程退出且账本空闲；随后 `--prepare-only`完成211项源码/资产/输入约束和具体40版来源绑定。原三个非账本用户脏文件及两个累计资源文件在集成期间保持原字节；后续资源文件只由原guard自然记录。

新配置仅将TRAIN落点奖励距离尺度0.25改为0.75、seed改为748901，最大奖励仍20，原矩形目标区、动力学、真实完整球拍、规则及50/70/85、A/B各200回合保持。该方法假设已在中间结果产生前登记，尚无新原生性能证据。

下一步通过 `reports/G07_full_racket_v4/centered_placement075_pilot8_queue01/run.py` 实际运行2048环境×8更新。流程核对首个真实更新、canonical新8及全训练轨迹，接着执行原全TRAIN128、零扰动、seed724201和同六例实际侧视/俯视视频。没有自动加长训练或继承来源成绩；新冻结证据出来后再决定后续方法。

最新已交付固定策略视频仍为[40版84.66秒六例失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_centered_residual_base8_iter40_20261026.mp4)。另有[24秒实际训练完整回合与失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_second_complete_and_failure_20261026.mp4)，不是冻结成功率或正式验收。
