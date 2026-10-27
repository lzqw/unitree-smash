# 残差15训练、独立冷128和实际视频

本包保存实际完成的128环境×128步×15更新训练、其canonical地面检查点与导出策略、全部721闭合训练回合、15个诊断检查点、实际接触/被选训练轨迹、源码快照，以及随后同一冻结权重的原128例独立冷评估和视频。变化中策略训练的7次空中触球，不作为最终冻结策略的空中触球成绩；独立冷128为0空中、0合格恢复、0完整、NF0，未超过原样教师4空中的结果。

`training15/job.json`和`cold128/job.json`给出实际命令、Git版本、时间、种子和真实资源。训练使用seed751001；冷评估使用seed724201、零扰动、原128案例、原点0和原评分。`training15/checkpoint.pt`是canonical地面状态：可在原程序、原128批量及相同合约下严格续训学习器，但不宣称恢复任意中途不可访问的PhysX接触缓存。`ppo/model_*.pt`是诊断检查点，不能与canonical地面重启文件混用。

两次运行各自的`source_snapshot.zip`、`source_hashes.json`保存当时408项实际源码、配置及USD快照。复现需先在全新工作区恢复对应路径，使用原`job.json`命令和全新输出目录。外部依赖仍包括锁定的IsaacLab、IsaacSim、Torch、walking策略以及完整USD/mesh引用；可复用[已核验运行资产包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip)，SHA256为`44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb`。本包不是独立安装的仿真器，绝对路径需要按原机恢复或显式审计迁移，不保证跨硬件逐位相同。

训练的`--base-checkpoint`来自原样全身教师；该权重在`reference`保存。`--body-source-evaluation`所需的原教师完整冷128及其全部来源证据，可从[218MB全身比较包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_origin0_gain_body8_completed_search_and_cold128_20261027.zip)恢复，SHA256为`02491c0e379834607317ae9d6a82bcecff1308efc2ff7f42f5731afa3780b4d2`。某些检查脚本还验证原队列绝对路径输入树，完整清单见coordinator中的ADMISSION_DECISION；它不是仅靠当前ZIP和Torch便可运行的独立脚本。本交付的下载校验只证明字节与记录一致，未声称在新解压环境重新执行CPU检查或Native仿真。

视频由本轮冷128的实际50Hz姿态和球位置、原生接触事件生成；显示三次拍框接触和两个固定未触球对照，保留完整首回合和真实结局。慢放重复原始帧，不插值运动。末态定格标为最后真实50Hz采样；首次接触使用原生事件，首次落地画面中的刚体姿态注明最近采样时间。远景保留实际飞远球和场地位置，接触单独提供斜侧放大和俯视，不能把相机远近变化当作新的执行。

TRAIN69在更早的随机预热回合中的资格疑点，没有移用到本轮冷评估。本轮69实际为拍框接触、身体触地，仍按原规则失败。只读时序复现仍在单独运行，未混入本完成包，任何后续资格判断修正均须保存旧证据和原门槛。原A/B各200例及50%/70%/85%、NF0尚未通过。
