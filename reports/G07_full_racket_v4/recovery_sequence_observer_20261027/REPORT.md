# TRAIN69恢复资格时序：只读Native复现入口已就绪，尚未执行

实际新origin0训练已完成15更新、245760控制环境步、0.7990418561 GPU小时，Native guard退出0且SUCCEEDED。721个已闭合训练回合独立核对为7空中、0合格恢复、0完整、NF0；两层原源策略及归一化完整保留，全部128环境回合链通过。末canonical checkpoint为`e2f67d807ad91f669eb45dce8910b8ec9516dfeda12d74c8de21749c228bc32d`，导出policy为`21b7670cabbe72587cdb5b8270f6add4159733282f07bc93ee0217c0bb1b0b8c`。这些是训练结果，不是新冻结策略成功率。[实际训练回执](../origin0_residual_after_gain_body_queue01/TRAINING_RECEIPT.json)保留原数据。

GPU0已由guard1930046/native1930060执行冻结15版原128零扰动冷评估；seed724201、原128案例、实际128地面初态、原点0及策略SHA已核对。结果尚未完成，见[启动检查](ACTUAL_WAIT_AND_COLD_STARTUP_CHECK.json)。旧source0冷128的4空中及旧原点16米成绩均不移用给新15版。

[实际预热首128数据](../origin0_residual_warmup7_actor8_20261027/FIRST128_EPISODES.json)中，TRAIN69首次回合有合法空中拍线、3.729912秒稳定恢复、身体触地力0、qualified_flight_seen真，却qualified_jump_landed假；最后报告起落差约1ms。原记录没有保存该例完整100Hz轨迹或逐指标事件史，因此尚未证明原因，原条件恢复判0保持。该例速度4.76476m/s且首落x4.45167m偏出原目标，不能靠修正恢复标志就计为完整成功。

新`recorder.py`仅使用现有`_diagnostic_observer`接口，记录69和112的首回合跳跃指标采样：支撑足数、飞行资格、落地资格、恢复窗口、身体接触、球拍接触及其原生时间。生产G07跳跃指标通常每1ms采样，底层PhysX仍62.5微秒；不把指标采样冒充每个物理子步。FullRacketEnv在JumpEnv回调前已清除rally并递增episode_index，记录器明确处理这个顺序，球拍状态取自清除前的实际指标采样。

`check_cpu_interface.py`实际退出0：数值接口调用前后全部环境张量值及NaN位置、CPU Torch随机数状态保持；重复帧被拒绝、缓冲上限和首回合筛选及数据序列化核对。临时数值样例已移除。这是软件接口检查，不是Native轨迹、GPU RNG或任务性能证明，见[CPU_INTERFACE_CHECK.json](CPU_INTERFACE_CHECK.json)。新独立入口`audit_training_entry.py --help`实际退出0，见[CLI_CHECK.json](CLI_CHECK.json)。

`prepare_entry.py`从正在使用的实际训练程序创建独立副本，仅调整报告目录ROOT、加来源检查/代码快照、挂接及保存只读记录器。去掉这些精确注入并恢复ROOT后，字节与原生产入口完全相同，见[ENTRY_PLAN.json](ENTRY_PLAN.json)和[AUDIT_ENTRY_DIFF.patch](AUDIT_ENTRY_DIFF.patch)。原生产训练、控制器、奖励、资产、旧资格判定与全部评分字段均未改写。尚未声称记录器在Native中无扰动或复现了旧回合。

等待owner1930543目前真实存活、无子进程，持有529项输入SHA。它仅在1896271实际退出、15更新及独立冷128完成且结果核对之后，才通过原GPU0 guard运行原seed751001、128环境、前7次critic预热复现，记录指标顺序。新作业上限114688控制环境步、5400秒，不自动重试；若新冷源已达到原50/70/85门槛，跳过该复现并交由正式来源审查。状态见[state.json](state.json)，不是已开跑Native诊断。

初次等待进程1929983额外设置了OpenBLAS线程2，而原训练未设置。它尚无子进程时已被安全停止，原STOPPED记录保留于WAITING_RUN1_STATE.json；当前1930543按原线程环境重新等待，生产冷评估未被停止。原因与实际PID见[THREAD_ENV_ALIGNMENT_RECEIPT.json](THREAD_ENV_ALIGNMENT_RECEIPT.json)。所有观察器及队列代码SHA保持。

实际复现完成后，必须先比较初始新模型张量与原模型，并逐项比较全部128个原始首回合行；有差异就明确标出，不能把新诊断轨迹自动当作原TRAIN69的历史。随后再分析实际合格飞行、首次合法触球、落地和0.8秒连续稳定窗口之间的关系。当前未改分、未证明因果、未执行正式A/B、未宣称G07达标。

末次ACTUAL_UPDATE_15.json中的`all128_first_episodes_completed:false`是旧检查函数按`iteration==7`填写的标签，不能据此说首批128尚未结束：实际第7次检查及已保存的完整128行均证明其全部闭合。该旧回执与生产者代码保持原字节，在此说明标签边界。

审阅用统一补丁的上下文空行触发git diff --cached --check，提交被检查拦下。仅将展示补丁按相同两份源文件重生成零上下文格式，原上下文字节另存本地；运行副本、记录器及529输入SHA均保持，见[PATCH_FORMAT_CHECK.json](PATCH_FORMAT_CHECK.json)。
