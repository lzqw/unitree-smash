# 已完成7点搜索与独立冷128：原16米布局

[304.6MB完整原生数据、程序快照及实际最佳权重](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_contact_aim7_completed_search_cold_TRAIN128_origin16_20261027.zip)已上传并重新下载：304597428字节，SHA 046cfd38132326f354384f960d1a6adc0710d4fb69097da9b781166618c6c840；131成员CRC、130份payload SHA全部通过。包含7×128搜索首回合、各50Hz全身/球/时钟、100Hz公共输入动作、接触/计划/重置/资源和源程序，以及单独冷128的完整数据。

7候选完整依次2/2/5/3/1/2/2，原[0,.06]胜出；独立冷128实测**3完整、60空中、26合格恢复、24攻击、NF0**。完整比例2.34375%、空中46.875%、条件恢复43.333%，均未达原50/70/85要求。冷成功TRAIN29/39/105，不能把搜索5次代替冷3次。源教师原冷2/128仅作历史比较。

实际公开最佳权重SHA a7953fb80d0ea647cbd6d525d8ddbf8a0a0986f1bcecfbe8831db2c1384473a3，位于包内search7/best_candidate.pt。全896搜索行和128冷行独立核对；7候选CPU对实际GPU命令误差均≤1.675e-5，冷128前两例1600条有效输入复算最大1.03116e-5，保持原5e-4容差。首次打包检查错误地把权重中元组参考点与列表严格相等比较，修正为仅规范化容器类型后成功；权重和数值均未改写。

这是已完成的原16米数值布局结果，含[已定位的float32世界位置舍入漂移](../aim006_contact_direction_diagnostic_20261027/REPORT.md)。[新原点布局的原生精度准入已通过](../origin_precision_after_aim_queue01/REPORT.md)，[新布局8×128比较已另外运行](../prepose_origin0_after_precision_queue02/REPORT.md)，历史成绩不继承。原物理模型和评分字段保持。新布局尚无完整成绩，正式A/B未运行。

包内原生source_snapshot.zip提供当时程序；完整USD/mesh运行时伴随包和外部固定IsaacLab/IsaacSim/Torch仍需另外提供。此次新下载校验确认交付字节一致，未宣称重新在新目录运行原生物理。此前42.28秒视频展示的是搜索候选的TRAIN12成功和两类失败，它不是冷128抽样或正式结果。
