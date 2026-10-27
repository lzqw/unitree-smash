# 新origin0残差：预热7结束、actor第8次实际更新已核验

第7次诊断检查点实际核对actor及log_std与初始逐位相同，仅critic已更新；第8次actor和log_std均有实际变化，actor最大单张量参数变化0.00183203。所有模型张量有限。两者均是原训练过程中保存的诊断检查点，`canonical_restart=False`，不是可承诺恢复物理接触缓存的重启状态，也不作为最佳冻结策略。

| 实际检查 | 更新7 | 更新8 |
|---|---:|---:|
|控制环境步|114688|131072|
|新actor Adam步数|0|20|
|critic Adam步数|140|160|
|actor/critic RMS计数，各|114688|131072|
|新actor与初始逐位相同|是|否|
|首批原128全部结束|是|是|

`check_boundary.py`已实际执行退出0，直接读取两个实际PT及初始模型、核对17个参数对应的Adam状态、归一化计数、Native迭代/步数及首批回合编号；原519持有输入SHA保持。此前入口名inspect.py遮蔽Python标准库，CPU导入失败退出1，改名后重跑成功，未影响GPU训练，见[CPU_STARTUP_FAILURE.json](CPU_STARTUP_FAILURE.json)。检查结果见[RESULT.json](RESULT.json)。

首批原128回合已全部独立重算：**2次合格空中拍线、0合格恢复、0完整、NF0**，8例在来球释放前终止。其余120例包含118例没有合格空中触球及2例空中触球。原先观察到的127个闭合回合完整行SHA均保持；现保存[全部128原始行](FIRST128_EPISODES.json)，没有丢弃最后结束的TRAIN69。它们来自带随机探索的critic预热训练，不是新actor更新后的冻结评估，也不拿作正式A/B成绩。

两次空中接触分别为TRAIN112与69。TRAIN112出球约5.77m/s，首落原目标，但身体触地。TRAIN69出球4.76476m/s，首落x4.45167m、y0.71139m，超出原目标；它有3.729912秒稳定恢复记录、最大汇总身体触地力0、qualified_flight_seen为真，却qualified_jump_landed为假，原条件恢复仍判假。其最后报告起跳/落地间隔约0.001秒。这是第一个新查到的“真实空中接触＋稳定记录＋末态资格为假”样本，不能只靠既有无空中触球案例解释。

当前代码会在每次新的无支撑起跳时清除valid_jump_landed，而recovery_time保存最长稳定窗口。上述末态字段符合资格可能被随后极短离地事件覆盖的现象，但原TRAIN69没有保存完整100Hz回合轨迹，亦未保存相应逐采样飞行资格历史；尚不能还原其全部事件顺序或证明该机制。下一项应使用已有只读物理观察接口记录合格飞行、首次合法触球、落地和稳定窗口的时序，再决定是否提出计数修正。当前训练、评估器、评分和历史成绩均保持；即使未来恢复资格有修正，该例仍低于8m/s且偏出原目标，不能计作完整成功。

当前GPU0作业1896271/1923496/1923510继续原15更新。原两层源actor/RMS、物理资产、所有案例、目标区和正式50/70/85、NF0门槛保持。第8更新后才开始采集新actor变化后的轨迹，最终还需完成15并按原128例独立评估冻结权重。当前未达标，不执行正式A/B。

[实际检查点与首批128证据包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_origin0_residual_warmup7_actor8_actual_checkpoints_and_first128_20261027.zip)包含实际初始模型、model_6.pt、model_7.pt、128行、程序/结果、实际408源码快照及源策略。上传和独立新下载校验状态以UPLOAD_RECEIPT.json为准。运行仍依赖锁定的外部Torch/NumPy或Native环境；完整检查还核对原机519项绝对路径输入及上一轮127行观察，这些历史输入树并未全部嵌入本包。移机需要恢复完整源清单或显式审计路径迁移，未验证新解压环境中的CPU/Native重跑。全部训练数据和最终canonical15另在完成后交付，本包只证明当前实际边界检查。

证据包实际16,000,756字节，SHA256 `82fad145f67464e1f538e6d0bb5d52301a4fc71116c1f25bef40c1b1a4e23bab`；上传后独立新下载18成员/17 payload均通过CRC/SHA，见[UPLOAD_RECEIPT.json](UPLOAD_RECEIPT.json)。阶段结构校验实际退出2，仍缺少正式G07 evidence.json；未构造通过记录，见[G07_STRUCTURE_AUDIT.json](G07_STRUCTURE_AUDIT.json)。
