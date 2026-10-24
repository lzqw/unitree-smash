# QuadSmash：Codex /goal 实施与训练任务包

最新视频：[24秒真实攻击回球与恢复失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_20261024_1216.mp4)（24.42秒，1280×720、50fps）。CEM16候选17在一次己方空中拍线击球中出球8.19米/秒、向下1.65米/秒，并实际越网；随后身体触地，未观察到首次落点，完整扣杀仍失败。同参数另一独立回合是拍框误击后合格恢复3.92秒，没有拍线或合法回球。含侧视、俯视、四分之一慢放和原生接触定格；[原始轨迹、事件、权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_20261024_1216_evidence.zip)同时上传。全部1221帧解码通过，5附件GitHub大小/SHA256核对。当前GPU0已切换CEM17恢复训练，视频不包含该新方案。

此前视频：[35秒最新实际效果：空中击球、触网失败与恢复对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_1048.mp4)（34.72秒，1280×720、50fps）。最新CEM12独立回合实际空中拍线出球7.30米/秒并越网，随后机器人触网；同方案另一回合合格恢复4.45秒但漏球；此前CEM07回球7.52米/秒并在对方场内首落，但目标和恢复失败。含侧视、俯视、四分之一慢放、真实接触点与首落点；三回合均未完成扣杀。[原始轨迹、事件、候选权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_1048_evidence.zip)已上传，全部1736帧解码通过，5附件GitHub大小/SHA256核对。视频所示CEM12已保存停止，当前CEM17新四代训练仍在运行。

此前视频：[33秒当前真实效果：7.52米/秒回球、恢复与最新触网失败](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0931.mp4)（33.12秒，1280×720、50fps）。三个独立原生训练回合分别展示：CEM07空中拍线出球7.52米/秒并越网、场内首落但目标和恢复失败；另一回合击球后合格恢复2.92秒但球被网挡住；CEM08前移方案击球前触网失败。含侧视、俯视、四分之一慢放及真实接触/落点，完整扣杀仍0。[轨迹、事件、权重与精确程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0931_evidence.zip)一并上传，全部1656帧解码无误，5附件GitHub大小与SHA256已核对。当前CEM09新控制器不在视频内。

此前视频：[26秒最新实际效果：回球恢复与拍框失败对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0819.mp4)（26.04秒，1280×720、50fps）。来自两个独立原生训练回合：CEM03第二代候选16从地面起跳，在己方1.667米处拍线触球、越网、对方场内落地并合格恢复3.94秒；出球4.85米/秒且向上、横向偏出指定目标0.75米，完整扣杀仍失败。后一方案CEM04候选11拍框误击后在3.04秒身体触地，失败按原记录保留。含侧视、俯视、四分之一慢放、实际接触点和首次落点；[姿态、事件、候选权重和精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_latest_20261024_0819_evidence.zip)一并上传，全部1302帧解码无误，5个附件的GitHub大小和SHA256已核对。当前32参数新作业仍在运行，视频没有将它算作完成。

此前视频：[42秒当前实际效果：空中回球、目标命中与失败对照](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_three_cases_20261024_0713.mp4)（41.86秒，1280×720、50fps）。侧视/俯视/慢放展示三个独立原生训练回合：上一轮候选6的一次合法空中回球并合格恢复；同参数另一次执行命中目标但合格恢复失败；当前新策略候选0空中触球后在2.88秒身体触地终止。三个回合均未完成扣杀，未将它们拼成同一成功回合。保留当时的训练状态标注。[对应姿态、事件、权重与精确源码](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_three_cases_20261024_0713_evidence.zip)已随视频上传；全部2093帧解码无错误，正式A/B各200回合尚未运行。

此前视频：[32秒同回合合法空中回球与合格恢复](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_return_recovery_20261024_0640.mp4)（2026-10-24，31.56秒；侧视/俯视/慢放/原生接触快照/真实首次落点）。TRAIN1候选6从地面自主起跳，在己方1.664米处真实拍线触球，球触网后合法越网、首次落在对方场内，同回合连续稳定恢复3.62秒、身体触地0。完整扣杀仍失败：实际出球4.827米/秒、向下0.250米/秒，均低于原攻击要求；首落点[3.004,-1.067]米，横向偏出原目标6.7厘米。同候选TRAIN65未触球且身体触地，失败对照也在视频里。[完整原始数据、候选权重与精确源码证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_return_recovery_20261024_0640_evidence.zip)随视频发布，全部1578帧解码通过。单例不代表正式面板成绩。

此前[高位触球与独立恢复视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_recovery_20261024_0555.mp4)及[证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_contact_recovery_20261024_0555_evidence.zip)、[低位触球视频](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4)及[证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450_evidence.zip)保留。

此前完成训练：`ball_contact_recovery_cem02` 两代128次候选首回合共32次接触、15次空中拍线、24次合格恢复，其中1次拍线与合格恢复同回合；9次满足任务触球条件的实际越网、1次目标命中、0攻击达标、0完整扣杀、0数值故障。实际102400控制环境步、0.599666 GPU小时。第二代同候选6命中了目标并稳定3.83秒，但主落地后出现约1毫秒再次离地，原判据要求的合格落足为否，恢复仍按失败保留。没有通过修改判据重算成功。

此前完整训练 `ball_pitch_blend_cem03_resume01` 已完成两代256次候选首回合：28次接触、19次空中拍线、11次合格恢复，其中1次拍线与恢复同回合；8次符合任务触球条件的实际越网、1次目标命中、0攻击达标、0完整扣杀、0数值故障。实际204800控制环境步、0.601900 GPU小时。[第二代原始数据、完整统计及程序附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_pitch_blend_gen1_and_full_summary_20261024_0752.zip)已上传并核对摘要。原中断作业保留，恢复是从本代地面起点重放。

后一方案 `ball_momentum_velocity_cem04` 第一代128次首回合仅有2次拍框接触、0拍线、6次合格恢复、0完整扣杀和0数值故障。该固定速度方向方案已主动停止，第二代部分记录保留；守护回执仍为FAILED，不伪称整轮成功。实际0.355604 GPU小时、122368控制环境步；计费步数上限仍保留204800。

最新完整训练 `ball_native_reference_interception32_cem07` 已正常完成两代256次候选首回合：37接触、25空中拍线、10合格恢复、1次拍线与恢复同回合；8次几何越网中3次符合任务触球/规则条件，2次对方场内首次落地，0目标命中、0攻击达标、0完整扣杀、0数值故障。实际0.600544 GPU小时、204800控制环境步。较快候选32在两次同参数TRAIN1执行中出球7.468/7.524米/秒，实际越网且落在对方场内，但目标和身体恢复失败；另一候选完成拍线与2.92秒合格恢复，球被网挡回。没有把不同回合拼成完整成功。

上一轮32参数提前引拍 `ball_prepose_velocity32_cem06` 第0代128次执行为1拍框、0拍线、8合格恢复、0攻击/完整、0数值故障。实际正向法向拍头速度在部分采样达到2–3米/秒，但拍头高度和横向仍未对准来球。该方案在本代完成后主动停止，第二代31步保留，guard回执仍为FAILED；实际0.316229 GPU小时、106368控制环境步。

上一方案 `ball_power_body_bank_cem08` 已在完成两代后主动保存停止：256次首回合12接触、4空中拍线、4合格恢复、0同回合拍线与恢复、2任务有效越网、2场内首落、0目标/攻击/完整、数值0；140触网、53提前越网。第0代120次全身提案中113次规则终止、7次身体触地。FAILED回执及第三代21步保留；实际0.439862 GPU小时、149760控制环境步，原声明cap409600不改。

任务空间准备 `ball_cartesian_preparation_cem09` 完整第0代128回合为0接触/0恢复/0完整，124身体触地，其中41回合在来球前已结束，数值0。该准备动作使行走失稳，已保存停止；第1代104步与FAILED回执保留，实际0.151311 GPU小时、52608控制环境步。

关节准备修正 `ball_joint_preparation_cem10` 已在完整第0代后保存停止：128执行3接触/1空中拍线/6合格恢复/0同回合拍线恢复/0有效拍线/0攻击/完整、数值0；88触网、14提前越网。第1代131步和FAILED回执保留，实际0.345378 GPU小时、119168控制环境步。其余合格恢复均漏球，没有用漏球恢复的最佳参数替代快击球初始化。

工具运动预测 `ball_tool_motion_anticipation_cem11` 完整第0代128回合为3接触/1空中拍线/5合格恢复/0同回合/0有效拍线/0攻击完整、数值0；85触网、15提前越网、2身体工程越界。已保存停止，下一代31步与真实FAILED回执保留；实际0.307542 GPU小时/106368控制环境步。

本轮新进展：CEM16候选17的一次真实己方空中拍线出球达到8.1947米/秒（向下1.6518米/秒），实际越网余量7.45厘米，首次满足原研究攻击阈值；但3秒处身体触地，首落点未观察到，完整扣杀仍失败。此为前300步的一次已闭合训练回合，整代和正式400尚未完成。[原始事件、300步全身姿态与精确权重](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_attack_velocity_crossing_native_partial300_20261024_1208.zip)已上传并核对SHA，下一步重点把攻击和合格恢复接到同一回合。

当前GPU0运行 `ball_attack_recovery_cem17`：第一代128回合已完成，73接触/51空中拍线/12攻击出球达标/14合格恢复；其中1次己方空中拍线后合格恢复3.44秒，但出球仅5.36米/秒、被球网挡住，0目标/完整、数值0。下一代继续运行，原生记录已确认保留实际最佳候选26重测。后续可学习5个既有站立控制参数的37坐标方案在独立分支准备，108项CPU测试、122路287步指令检查和旧32/新增默认37坐标指令一致性通过，尚未在GPU执行。所有真实硬件、资产、球场/规则与正式门槛保持。

[两轮失败的完整第0代、后续中断记录、本轮源码/权重和前100步原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_arm32_failures_and_interception_program_20261024_0833.zip)已上传（185,838,293字节），125个归档文件与GitHub附件大小/SHA256均已核对。各作业精确启动源码、实际资源回执和新旧权重分开保留。没有后续排队作业，正式A/B各200回合尚未运行；原8米/秒、向下分量、目标落点、合格跳跃恢复和全部物理条件保持。

此前上部拍线课程 `ball_upper_strings_cem01` 的128次候选首回合全部未触球、未完成合格恢复，完整成功0、数值故障0；实际77120步、0.464638 GPU小时。[完整失败轨迹、权重、精确源码及接续程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_upper_strings_failed_and_contact_update_20261024_0617.zip)保留，包含全部64环境50Hz原生全身姿态、运行依赖和CPU攻击窗口提案。

更早恢复课程128次执行中，13次合格跳跃恢复均漏球，4次接触中1次为空中拍线，0完整成功、0数值故障。这批的第一代[原始数据/权重](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_gen0_and_rule_code_20261024_0528.zip)、第二代触球的[早期部分记录与当时程序](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_contact_and_curriculum_v3_20261024_0532.zip)保留；不同批次没有混算成功率。

[完整恢复训练、近网修复与程序附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_native_rule_fix_and_recovery_20261024_0605.zip)包含128次原始回合、权重、精确启动源码、10层USD、3个通过的原生近网边界回归，以及此前2个失败尝试，固定于提交 `dc0a4b2`。修复使用实际GPU刚体姿态计算球拍实体与网平面的关系，避免DirectGPU下CPU场景查询的旧姿态。目标回归通过不等于策略验收。当前分支新增[接触保留与实测恢复腿参数交叉课程](reports/G07_full_racket_v4/CONTACT_RECOVERY_CROSSOVER_PLAN_0612.json)，[51项CPU测试](reports/G07_full_racket_v4/CPU_TESTS_0612.txt)通过；新课程原生训练已实际启动，之前的失败按原始结果保留。

复现时同时解压[运行依赖附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_runtime_dependencies_20261024_0608.zip)，补齐配置读取的冻结Go2步态/站立权重、标准PiPER无夹爪源URDF及其引用网格。清单逐项列明SHA；与最新程序包解压到同一仓库根目录。外部Go2 USD层的原始路径映射见 `configs/full_racket_usd_dependencies_v4.json`。


日期：2026-10-21。对象：Go2 + 标准 AgileX PiPER + 固定安装球拍；Isaac Lab；PPO。

**交接阅读入口（2026-10-22）：[给 GPT 的当前情况与代码导读](docs/GPT_HANDOFF_20261022.md)。**
包含真实结果、失败经过、关键程序、尚未验证的假设和可直接复制给 GPT 的问题。
代码与文本证据保存在本仓库；权重、完整运行记录及资产见
[交接附件与恢复说明](docs/HANDOFF_ARTIFACTS.md)。上传不代表训练验收通过。

**当前状态（2026-10-24北京时间）：新版完整球拍、真实球场和击球规则正在适配训练。用户新授权仅GPU0、无累计GPU小时上限和截止；旧窗口与费用保留。当前资产full-racket-court-v4-r3，675 mm完整球拍、6.1 m宽悬空网、20.709991 kg总质量。原生触球、网体/网柱、网下/网上通行及直立持拍自碰撞回归已运行；新版400案例已冻结，正式成绩尚无。当前在进行真实动力学全身动作搜索。**

G00–G06的通过属于旧短拍板资产，不能继承为新版通过。旧G07正式400未达标（A完整50%/恢复89%，B完整35.5%/条件恢复74.4898%，数值0），旧权重、数据和失败保留。新版模型来源、规则映射与假设见[MODEL_RULES.json](reports/G07_full_racket_v4/MODEL_RULES.json)，评测配置见[新版协议](configs/g07_full_racket_eval_v4_r3.json)，实际下一步见[运行状态](status/NEXT.md)。[新版完整程序、资产、失败记录和无来球学习检查点](https://github.com/lzqw/unitree-smash/releases/tag/g07-full-racket-v4-progress-20261024)已上传并核对大小及SHA256。最近完成材质一致、62.5us的256次无球候选执行（含精英重测），其中8次无触网/无身体越面并完成跳跃恢复；8环境原生扰动写入/公开观测/局部reset烟测通过，真实来球联合训练正在运行。[增量权重与原始证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_full_racket_adaptation_20261024_0400.zip)已上传。上述烟测不等于独立正式400通过。

视频入口：[38秒当前完整球拍r3动作与失败记录](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_r3_20261024.mp4)（侧视、俯视、慢放；无球恢复、真实来球漏球、触网失败，尚无合法完整扣杀）、[对应原始轨迹证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_r3_20261024_evidence.zip)。历史视频：[r1机械测试与旧策略对照](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/Current_effect_v4_20261024.mp4)、[旧短拍板效果](https://github.com/lzqw/unitree-smash/releases/download/g07-progress-20261023/G07_current_effect_comparison_20261024.mp4)。历史报告：[旧G07实测报告](reports/G07/REPORT.md)、[旧视频、权重与完整轨迹附件](https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023)。G04结果与接口见[训练总结](reports/G04/autonomous_learning_20261022/REPORT.md)、[策略接口](docs/G04_POLICY_INTERFACE.md)。旧交接文档保留历史状态。
旧资产为约20.647 kg浮动系统；新版完整球拍资产为约20.710 kg，仍保留标准PiPER、Go2和原执行器限制。
统一 794 维观测、18 维动作、原生 PPO、独立评测及可恢复检查点已实现。
最新状态与累计资源记录分别见 `status/STATUS.json`、`status/NEXT.md` 和 `status/GPU_LEDGER.json`。
工程通过或存在权重文件均不表示策略达到成功率要求；G04 的两个固定面板必须各达到 90%。

## 开始
1. 将本目录内容放入新的研究工作区；已有项目先建立分支，不覆盖原有 AGENTS.md。
2. 阅读 `configs/budget.json` 和 `status/NEXT.md`。当前授权窗口为`g07-full-racket-unlimited-20261024`，仅GPU0、无累计GPU小时或截止限制；保留所有旧费用、单任务有限超时和单GPU作业保护。
3. 在同一工作区打开 Codex CLI/IDE。先输入 `CODEX_GOAL_PROMPTS.md` 的“第一轮启动目标”。
4. 搜索、诊断、失败重试和评测均计入对应授权账本；使用`scripts/run_gpu_job.py`，不可因恢复会话重复启动作业。G08以后尚未授权。
5. 不需要每轮重贴完整方案；Goal 引用这些持久文件。换会话时先读状态和最近报告。

官方 Goal 用法为 `/goal <目标>`；`/goal` 查看，`/goal pause` 暂停，`/goal resume` 恢复，
`/goal clear` 清除。见 `docs/SOURCES.md` S1–S3。若命令未启用，可按官方文档检查
`codex features enable goals`，不要编造 `codex --goal` 或 `codex goal run` 命令。

暂停 Codex 并不意味着其启动的训练进程/调度器作业必然自动停止。
本项目必须另行记录作业 PID/Job ID，支持保存并停止项目自己的作业。

## 阅读顺序
- `AGENTS.md`：禁止事项、证据、预算与执行规则。
- `PLAN.md`：阶段依赖和任务清单。
- `TRAINING_SPEC.md`：算法、观测动作、课程与数值要求。
- `ACCEPTANCE.md`：通过标准及研究结论边界。
- `goals/Gxx_*.md`：本阶段实施细节、命令契约、失败处理。
- `configs/acceptance.json`：机器可读的验收条件。
- `status/STATUS.json`：初始全部 TODO，不得预填 PASSED。

## 已可运行的检查
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/verify_goal.py --root . --goal G00
```
G00–G03 的真实证据已保存在对应 reports 目录；缺少证据或未达到指标的阶段应当校验失败。
校验器检查证据文件、摘要、依赖、指标和必需产物；**不独立证明日志来自真实仿真**，
仍需保留原始逐回合数据、执行日志和可复现命令以接受审计。

## 最小可交付边界
G00–G07 = 能审计的跳跃扣杀仿真基线。
G08–G09 = 可恢复状态初始化与恢复约束事件选择研究分支。
G10 = 冻结测试、消融、多种子及结论审计；结果可以是不支持方法优势。

本次[完整CEM07统计/第二代原始数据及当前四代训练程序、权重和前100步原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_interception_complete_and_body_learning_20261024_0912.zip)已上传（78,543,764字节，SHA256和49个归档文件均已核对）；第0代完整姿态见此前71MB附件。正式A/B各200回合仍未运行，完整扣杀仍为0。

最新[前移准备失败与任务空间准备控制的原生证据附件](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_body_preparation_failures_and_cartesian_native_20261024_0938.zip)（120,166,563字节，84个文件SHA及GitHub摘要/大小已核对）包含CEM08两代完整轨迹、第三代21步中断、真实FAILED回执，以及CEM09精确源码/输入、81项CPU检查、120路回放和实际前100步。首100步只证明初始化与执行，后续触网规避与扣杀效果仍待实测。

最新[任务空间准备失败与关节准备修正的程序/原生证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_cartesian_failed_and_joint_preparation_native_20261024_0952.zip)（54,384,135字节，78文件与GitHub SHA核对）包含CEM09完整第0代失败和第1代104步中断、CEM10精确源代码/输入/86CPU检查/真实前100步。CEM10随后已实测前200步128回合均未提前结束；后续触网、击球与恢复仍待运行结果。

当前[关节准备修正第0代前400步原始证据](https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_joint_preparation_native_partial400_20261024_1000.zip)（49,772,106字节，26成员和GitHub大小/SHA核对）已发布：121首回合结束、7待恢复，1空中拍线+2拍框；拍线出球7.235米/秒并越网，但随后规则故障且恢复失败。3秒截点机器人规则终止由116减至92，触网仍未解决，完整扣杀仍0。四代作业继续运行，正式400未运行。
