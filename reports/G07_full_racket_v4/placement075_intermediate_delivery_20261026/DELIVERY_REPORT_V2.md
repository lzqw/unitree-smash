# 075中间16/24/32完整冻结数据V2

[下载比较包（207.5MB）](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_intermediate16_24_32_frozen_TRAIN128_and_source_diagnostics_20261026_v2.zip)，精确大小207535268字节，SHA256 `de391616b2b07a8a3d39c109cddbc0640ae3497f8892afead8b7be764629ab47`。已上传GitHub，服务器摘要及独立分段新下载SHA一致，490成员CRC及489份payload SHA全部通过。

包含三个实际原生作业的完整原TRAIN128终态回合/接触、128例50Hz全身/球轨迹与有效mask、前2例100Hz公共输入和动作、三份公共策略、实际model15/23/31源诊断、canonical8和固定BASE846、精确原生源码快照及RSL源码/许可证。原生诊断不是重启点。历史partial观察文件保留在原目录，包内使用完整终态结果，失败案例全部保留。

新目录实际使用包内RSL重算三次全128并核对源RMS/17项Adam；有效公共输入分别1331、1119、1600，共4050条，导出误差均0。原数组800×2的无效终止后部分未冒充有效样本。本次未做新目录原生物理重跑。16版0完整/25空中/15恢复/0攻击，24版0/22/12/2，32版0/30/18/4，均NF0；既定规则在五版及十来源选择中最终选075第8版。

V1在新目录检查时发现遗漏原TRAIN案例文件，已保留V1程序、归档及实际失败；V1未上传。V2明确加入该文件并通过上述新目录与新下载检查。新阶段leg06训练没有因此重启或改动。

CPU比较数据及权重核验所需数据和RSL已包含；原生运行的完整资产、外部USD映射及训练依赖使用[已验证1.213GB伴随包](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip)，SHA256 `44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb`。Isaac/Kit/Torch等软件仍需原记录环境。正式A/B各200及原50/70/85尚未通过。
