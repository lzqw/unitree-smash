当前唯一GPU0作业（2026-10-24T15:57:41.287037+00:00已核查）：ball_slew50_near_attack_positive38_cem24，guard1455144/child1455158/session21247，源码9e1643d1767f35bd297718e16e8f3153ac7ee17d。4代×64候选×TRAIN[1,65]；7200秒/409600控制环境步保护，只有GPU0，无额外排队。守护/子进程本次都确认存活；观察超时须重查同句柄，不重启。不要手改GPU_LEDGER/RUNS、旧RESOURCE_STOP/queue或无关MPS。

本轮新warm来自CEM23实际candidate54（TRAIN65出球7.966534895m/s，首落x4.508449，恢复3.183952秒/body0；未满足原8m/s和落点目标；同参数TRAIN1拍框失败）。新policy SHA328b1a7d7059a06335585e19828ef63bcd7af2d001fa3d0ee39c2f692ac746c2，bank SHA64625626a87eb04e6f179145f543b4b7e1ea7fcb0d2d655c9c623e668e344d6d。60组合=6normal[.02,.04,.06,.08,.10,.12]×5lead[.01,.02,.03,.04,.05]×2gain[1.0241,1.0659]；高斯mask[7,14,37]，sigma.04。其余腿/臂/恢复参数固定；不继承旧结果。

实际100步初始化已冻结positive38_initialization_1600/REPORT.json：64×38/60bank/128路地面起点与有限状态、首2路动作及臂目标解码吻合。此时0接触/0终态、限速尚未绑定，没有新的击球或恢复成绩。CEM24/CEM23全部control/physics源码hash相同。下一步检查真实300/400接触与完整800步；不要把pending判成功。

CEM23已在两完整代后主动保存停止：256次候选执行、仅2种不同TRAIN来球；66拍线/136合格恢复/28拍线+恢复/47场内首落/30目标/0攻击或完整/2机器人规则终止/NF0。第二代42拍线但最快6.6872m/s；高斯normal全负，离开第0代近攻击+.08。原始数据全部保留。第2代73步中断、0终态128pending；实际214144控制环境步/.614570775025GPU小时，guardFAILED/childexit0/无completed回执。核验slew50_terminal_1556/REPORT.json，原cap409600保持。

已准备现有笛卡尔反馈坐标的新60组合（servo_contact38_ready01_1600），只改已有[14,18,37]，servo[.35,.55,.75,1]×lead[0,.01,.02,.03,.04]×normal[.04,.08,.12]，仍software5，腿/挥拍速度方向/恢复保持。CPU预检仍不等于新原生效果；先看CEM24完整结果，再决定，不排队。

物理与验收保持：20.709991kg完整负载/675mm完整拍/原真实球网，腿23.5Nm/30rad/s、臂100Nm/5rad/s仍为未标定假设；软件目标slew5不是硬件提额。原norm>=8/vx>0/vz<=-1、真实越网首落X[2,4]/Y±1、合格跳跃/落足和连续.8秒恢复、全部合法性不变。实际速度超调不裁剪，100Hz观测不冒充子步峰值。

完整任务仍0、原TRAIN128诊断尚未原生运行、正式A200+B200尚未运行。需完整同回合链后扩大训练分布/原幅度扰动并锁定独立面板，较差面板full>=.50/air>=.70/conditionalrecovery>=.85/NF0；最终还需旧短拍板动态对照和代表性成功/失败实际视频。最新已导出视频仍CEM20的41.44秒3独立案例：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261024_1400.mp4
该视频并非CEM24新效果。历史详细状态在NEXT_before_positive38_1600.txt。

实际400步补充：positive38_contact_prefix_1605/REPORT.json已冻结77接触/15拍线（14拍线仍pending）、13closed/115pending；闭合者1拍线/1场内首落/0恢复完整/NF0。after-slew命令峰值5.0000119，实际100Hz臂样本峰值5.0005126和4个>5.0001样本均保留；全部77事件command与target日志吻合。此为部分记录，不能代替800步结果。

167,207,887字节完整CEM23两代/73步中断+本轮100步/源码/下一CPU方案已上传，SHAffc434aaee00305191e7ba530dd4ccca01eef1adf7f723bc92841d17f94e9627；回执slew50_positive_search_upload_receipt_1605.json。400步快照晚于该归档，未在包内。
