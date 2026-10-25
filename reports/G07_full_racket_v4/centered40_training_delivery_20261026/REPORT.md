# 新增量学习器40：完整原生训练交付

GPU0上的canonical8→40严格续训已经成功完成。本作业实际8388608控制环境步、2.748762716 GPU小时，独立重算17156个闭合训练回合：1945次合法空中接触、916次条件恢复、2个完整成功、数值故障0。这是变化中策略的训练记录；第40版冻结成绩尚未产生，完整目标仍未通过。

[完整训练、权重和两次成功实录包，843MB](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered40_completed_native_training_and_two_complete_videos_20261026.zip)为843374543字节。223份内容及1份清单共224个成员的CRC、内容SHA均已在本地检查；上传和新下载文件的核验见[交付回执](ARCHIVE_UPLOAD_RECEIPT.json)。

整包SHA256：`63f4ceb476b352456fcf699cf8fff7c52f97428a132f9fccf0296d3224af80c3`。

## 实际完成的学习和恢复核验

| 项目 | 证据 |
|---|---|
| 真实原生作业 | 原GPU0 guard成功回执，exit0，未超时，2.748762716 GPU小时 |
| 来源与末次更新 | 真正canonical8严格续训32次至canonical40 |
| 本学习器累计步数 | 10485760；其中本作业8388608，不是整个项目资源归零 |
| 2048条lane | 从新地面状态执行；源回合编号1至3保留，全部源编号至末编号链精确 |
| 优化器 | 全17项Adam保留并各推进640步，模型与优化器数值有限 |
| 归一化 | actor与critic各保留来源计数并增加8388608 |
| 原始动作 | 全14条选定原生训练轨迹通过100Hz公共输入、实际姿态、接触和逐回合一致性审计 |
| 诊断边界 | 实际9至40共32个非canonical模型均有限，归一化计数与更新一致 |

[训练独立回执](../centered_residual_resume32_queue01/TRAINING_RECEIPT.json)记录完整检查点、公共策略、固定旧8基线的SHA；[轨迹审计](../centered_residual_resume32_queue01/TRAINING_TRACE_AUDIT.json)覆盖全部14条保存轨迹。最终canonical40 SHA为`3446560f26548cac7bad213fc779b8d62ae60d5525692402e1e0e0a54751fe73`；公共40 SHA为`2e1bf93056eff22dd309ea52a777ce0648aa50e025f63ec99e5ae8fe662fd84f`。

## 成功训练回合与失败均保留

[TRAIN111完整动作及两段失败，37秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_first_complete_and_failures_20261026.mp4)：实际LANE1519、EP3，actor10至16，出球8.002355米/秒，首次落点[3.994049,−0.695709]，恢复4.102885秒。速度和目标后沿余量仅0.002355米/秒与5.95毫米。

[TRAIN25完整动作及身体触地失败，24秒](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_training_centered_second_complete_and_failure_20261026.mp4)：实际LANE153、EP5，actor13至19，出球8.010170米/秒，首次落点[3.929108,−0.430321]，恢复3.626920秒。实际拍上接触点速度1.881473米/秒，与出球速度分别报告。

两个成功均跨越多个训练actor、归一化和探索状态，不能归为最终冻结40的成绩。视频事后选例且保留失败；画面中的成功比例不是总体成功率。原始非零回合编号、实际子步接触和首次落点保持，未合成或拼接动作。

## 包含内容与复现边界

包内包括完整本轮原生输出：17156条最终原始回合、实际接触记录、全部14条100Hz轨迹、canonical40与公共40、全部32个边界诊断权重、日志、配置、源代码快照和哈希；另外包含canonical来源8、固定旧8公共基线，以及两段已检查视频的实际轨迹快照和审核资料。原`episodes.partial.json`与最终`episodes.json`逐字节一致，因此仅保存最终原始记录一份。

运行时资产沿用[已经下载核验的215MB依赖包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_centered_residual_probe8_native_training_and_initial_video_20261026_v2.zip)，SHA`d672edf51ec90649b1b0381fe9ff861e0c2433c98de944e0a07937e52954c4e4`。它包含原Go2步行策略、标准PiPER URDF和10层USD依赖；需恢复包内注明的外部资产原路径或等价链接，并使用记录的原生Isaac/RSL环境。清单与`REPRODUCTION.txt`明确区分canonical重启、非canonical诊断、实际动作回放，以及从来源8重新执行原命令；没有宣称能恢复碰撞中的PhysX隐藏状态。

## 正在进行的下一步

原父流程已在GPU0上启动最终40版的全原TRAIN128、零扰动、seed724201冻结评测，真实guard/native为1710061/1710073；[运行观察](ONGOING_FIXED40_OBSERVATION.json)记录实际进程、策略SHA和输入契约，不包含终局成绩。该评测及后续16/24/32比较均在不同目录，未混入本训练包。

若40没有达到原50%完整、70%空中、85%条件恢复及NF0门槛，已登记队列会继续测原全TRAIN128上的16/24/32，再按预先规定的顺序选择。正式目标仍是同一冻结来源的A/B各200回合及较差面板聚合；本次训练结束、两次训练成功和交付包均不代替正式验收。
