当前没有运行中的 GPU0 作业。CEM28 的两个原生同回合完整扣杀视频仍保留，但独立验证已完成且未通过：candidate39 A/B 各200回合，A 为完整0/200、空中接触21/200、接触后恢复7/21、NF0；B 为完整0/200、空中接触9/200、接触后恢复1/9、NF0。较差面板 B 为完整0%、空中接触4.5%、条件恢复11.1%、数值故障0。

TRAIN128 也已完成：candidate30 完整0/128、空中接触7/128；CEM29 无噪声恢复搜索两代仍无合法候选，CEM30 在 B 级地面扰动下两代仍无合法候选。已新增可追溯的 TRAIN 重置扰动支持，默认旧路径不变；实现提交 `4bbe3ae`，291 项测试通过。

当前完整扣杀视频：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_complete_smash_20261025_2030.mp4

独立评测、策略锁、训练诊断和搜索终端报告：
`reports/G07_full_racket_v4/formal_and_robust_20261025.json`

GitHub 交付包（18,454,043 字节，SHA256 `736ea5f05e3128986f07c63c37c5b53fba5461547f1eb0a0b31755aa4475f46d`）：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_formal_and_robust_delivery_20261025.zip
