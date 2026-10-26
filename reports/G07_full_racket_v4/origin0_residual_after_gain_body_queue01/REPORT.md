15更新新origin0全身残差训练已登记为后续等待队列，owner1896271实际等待全身比较owner1895642退出，无GPU子进程；不把等待状态称为已开训。当前1884350/1884474/1884488原生比较继续，482份输入SHA保持。

顺序为：全身时序/恢复8×128与独立冷128实际完成→核对真实终端及owner退出→绑定实际胜出权重及全128冷源→重新CPU复算正向来源→GPU0新128环境×15更新→独立冻结原128及CPU动作复算。初始7更新只学critic，之后8更新学18维残差actor；原2个源actor/RMS完整保留，新的actor/critic RMS和Adam从零开始，腿/臂请求探索标准差0.03/0.015rad。旧16米结果不继承。

训练上限245760控制环境步/7200秒，冷评估上限102400步/2700秒，真实资源继续由原guard计账，单GPU0、不自动重试。真实第1/7/15更新将独立核对初态、归一化计数、actor预热、Adam步数和第一批完整回合；末次检查canonical ground checkpoint与全部逐回合链。训练中成功次数和最终冻结成绩分别报告，原正式A/B及50/70/85、NF0要求不改。

[参数化实际来源CPU检查代码与兼容性证据](https://github.com/lzqw/unitree-smash/tree/g07-release-target-envelope-v4-20261026/reports/G07_full_racket_v4/origin0_selected_residual_binding_20261027)当前658有效输入零残差指令差0，CPU对GPU3.09944e-6。实际未来冷源正向准入仍待执行，必须得到actual_cold_source_admission非空，不能用初始兼容性报告替代。全新输出目录拒绝覆盖旧CPU导出，CPU策略明确native_execution=False。

STATIC_CALL_CHECK.json已实际解析两层队列全部直接本地调用的参数签名，当前可核对的准备与CPU入口文件存在。没有新Native训练、冻结评估或正式A/B成绩，目标仍未通过。
