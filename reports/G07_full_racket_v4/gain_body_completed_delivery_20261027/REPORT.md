# 全身八项比较与独立冷128：没有性能改善，新残差训练已启动

原128来球的8组比较及随后独立冷128均已实际完成。新冷128结果为**0完整成功、4合格空中拍线触球、0合格恢复、0攻击出球、数值故障0**；10次实际球拍接触，3次越网并首次落在合法单打场内，其中2次落在原目标区。四次空中接触的出球速度仍低于原8m/s门槛，均有身体触地。当前的主要失败包含触球覆盖不足、出球速度不足和恢复失败，扩大目标区不能使这轮已有四次触球成为完整成功。

| 原128搜索指标 | 原样 | 关闭两层残差 | 提前40ms | 推迟40ms | 恢复相位+.15 | 恢复相位−.15 | 走位+.10m/提前 | 走位+.10m/推迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|空中拍线|4|0|0|2|3|4|0|2|
|完整成功|0|0|0|0|0|0|0|0|
|合格恢复|0|0|0|0|0|0|0|0|
|攻击出球|0|0|0|0|0|0|0|0|
|数值故障|0|0|0|0|0|0|0|0|

原排序仍选择候选0，两层既有残差增益均为1、全身参数增量为空；原两个源actor及归一化统计完整保留。搜索实际715,136控制环境步、2.2153920972 GPU小时，guard按819,200步保守记账。独立冷128重新启动实际运行102,400步、0.3244271158 GPU小时；原guard/native及owner 1895642均已实际退出，未覆盖旧证据。

`check_and_package.py`已实际执行退出0，重新核对全部1024搜索行、128冷评估行、原案例顺序、实际Native回合编号、地面初始化、零扰动、seed724201、原点0及原协议。两次运行各400份实际源快照逐项检查CRC和SHA，并与当前对应源文件核对。冷评估前两通道1600条实际公共输入重新执行完整控制器，CPU对实际GPU最大动作误差3.516674e−6，原5e−4容差保持。完整结果见[ACTUAL_COMPLETED_DATA_CHECK.json](ACTUAL_COMPLETED_DATA_CHECK.json)。

当前冷评估和上一轮prepose冷评估的有效性掩码相同，48,305个共同有效50Hz采样的完整刚体姿态、球位置和时钟逐位一致，最大差为0。该结论只覆盖保存的实际采样，不宣称观测了未保存的所有物理子步。本轮成绩由本轮128行独立重算，不由旧成绩继承。[已发布73.08秒视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_origin0_cold128_four_air_contacts_and_miss_20261027.mp4)仍明确来自上一轮冷128；目前没有新训练冻结权重的效果视频。

[完整搜索、冷评估、权重与程序包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_origin0_gain_body8_completed_search_and_cold128_20261027.zip)为218,168,422字节，SHA256为`02491c0e379834607317ae9d6a82bcecff1308efc2ff7f42f5731afa3780b4d2`。本地及上传后的独立新下载均完成180成员/179 payload的CRC/SHA核验，文件SHA一致，见[UPLOAD_RECEIPT.json](UPLOAD_RECEIPT.json)。包中保留全部原始接触、50Hz轨迹、实际动作、源码快照、最佳权重、逐回合CSV、八项对照和触地几何诊断。另含上一轮冷评估轨迹与源权重，便于复算本次采样一致性。运行环境依赖与复现边界见[REPRODUCTION.md](REPRODUCTION.md)。

真实后续训练由owner1896271于2026-10-26 23:31:36 UTC启动，guard1923496/native1923510，明确使用GPU0。8份新安装文件与已审阅的隔离工作区SHA相同，实际128个地面初态、408项实际源文件和原482项持有输入均核对。新actor末层为零、新actor/critic RMS计数均0、新Adam为空；原2个源actor及RMS保持。实际已完成冷源正向CPU准入，1600实际输入上的零残差对源指令误差0；不再只是接口准备状态。详见[TRAINING_HANDOFF_CHECK.json](TRAINING_HANDOFF_CHECK.json)与[实际冷源准入](../origin0_residual_after_gain_body_queue01/ACTUAL_SOURCE_CPU/CPU_CHECK.json)。后者旧通用`scope`文本仍描述CPU检查边界，真正的冷源核验见其非空`actual_cold_source_admission`；它不构成新策略通过。

新训练为128环境×128步×15更新，共245,760控制环境步上限，单作业7200秒故障保护。前7次仅预热critic，之后8次学习18维残差actor，腿/臂请求探索标准差0.03/0.015rad。第1次实际更新已完成：16,384交互步，3个闭合训练回合，0空中、0完整、NF0；actor保持原样，critic八项Adam状态均20步，两套新归一化计数均16,384，见[ACTUAL_UPDATE_01.json](../origin0_residual_after_gain_body_queue01/ACTUAL_UPDATE_01.json)。该早期训练计数不是冻结策略成功率，首批128回合尚未全部结束。

下一步继续当前真实15更新；预登记核验第7/15更新及最终canonical地面检查点，随后以同一冻结权重、原128案例、零扰动和原评分进行独立冷评估。原球场、球网、完整球拍质量与尺寸、硬件限制、训练案例、目标区、攻击阈值和原A/B各200例50%/70%/85%、NF0正式门槛保持，正式评估仍未通过，目标未完成。

阶段要求的`tools/verify_goal.py --root . --goal G07`已实际运行退出2，仍缺少`reports/G07/evidence.json`，见[G07_STRUCTURE_AUDIT.json](G07_STRUCTURE_AUDIT.json)。旧G00–G06结构检查通过不代表新版完整球拍G07通过；当前真实性能也未达标，因此没有补造通过证据。
