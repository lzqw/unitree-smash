# QuadSmash：Codex /goal 实施与训练任务包

日期：2026-10-21。对象：Go2 + 标准 AgileX PiPER + 固定安装球拍；Isaac Lab；PPO。

**交接阅读入口（2026-10-22）：[给 GPT 的当前情况与代码导读](docs/GPT_HANDOFF_20261022.md)。**
包含真实结果、失败经过、关键程序、尚未验证的假设和可直接复制给 GPT 的问题。
代码与文本证据保存在本仓库；权重、完整运行记录及资产见
[交接附件与恢复说明](docs/HANDOFF_ARTIFACTS.md)。上传不代表训练验收通过。

**当前状态（2026-10-24北京时间）：新版完整球拍、真实球场和击球规则正在适配训练。用户新授权仅GPU0、无累计GPU小时上限和截止；旧窗口与费用保留。当前资产full-racket-court-v4-r3，675 mm完整球拍、6.1 m宽悬空网、20.709991 kg总质量。原生触球、网体/网柱、网下/网上通行及直立持拍自碰撞回归已运行；新版400案例已冻结，正式成绩尚无。当前在进行真实动力学全身动作搜索。**

G00–G06的通过属于旧短拍板资产，不能继承为新版通过。旧G07正式400未达标（A完整50%/恢复89%，B完整35.5%/条件恢复74.4898%，数值0），旧权重、数据和失败保留。新版模型来源、规则映射与假设见[MODEL_RULES.json](reports/G07_full_racket_v4/MODEL_RULES.json)，评测配置见[新版协议](configs/g07_full_racket_eval_v4_r3.json)，实际下一步见[运行状态](status/NEXT.md)。[新版完整程序、资产、失败记录和无来球学习检查点](https://github.com/lzqw/unitree-smash/releases/tag/g07-full-racket-v4-progress-20261024)已上传并核对大小及SHA256。无球搜索已有无触网的起跳恢复样例，真实球4案例探针未击中球并已保留部分失败记录；当前继续搜索，并已排队原生扰动重置核对及真实来球联合训练。这些均不代表正式400通过。

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
