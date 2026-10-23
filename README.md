# QuadSmash：Codex /goal 实施与训练任务包

日期：2026-10-21。对象：Go2 + 标准 AgileX PiPER + 固定安装球拍；Isaac Lab；PPO。

**交接阅读入口（2026-10-22）：[给 GPT 的当前情况与代码导读](docs/GPT_HANDOFF_20261022.md)。**
包含真实结果、失败经过、关键程序、尚未验证的假设和可直接复制给 GPT 的问题。
代码与文本证据保存在本仓库；权重、完整运行记录及资产见
[交接附件与恢复说明](docs/HANDOFF_ARTIFACTS.md)。上传不代表训练验收通过。

**当前状态（2026-10-24北京时间）：G00–G06已通过；G07正式400未达标（A完整50%/恢复89%，B完整35.5%/条件恢复74.4898%，数值0）。后续两批1536开发比较均未找到新的合格策略。原02:04工作窗已到期，实际新增结算7.645612 GPUh，GPU0本项目作业已停止，状态BUDGET_EXHAUSTED。权重、失败数据、公开轨迹与正式固定案例视频已保留；没有G07通过证据。**

最新入口：[G07实测报告](reports/G07/REPORT.md)、[继续执行与资源窗口](status/NEXT.md)、[视频、权重与完整轨迹附件](https://github.com/lzqw/unitree-smash/releases/tag/g07-progress-20261023)。G04结果与接口见[训练总结](reports/G04/autonomous_learning_20261022/REPORT.md)、[策略接口](docs/G04_POLICY_INTERFACE.md)。旧交接文档保留历史状态。
生产资产为标准 PiPER、Go2、安装板和球拍组成的约 20.647 kg 浮动系统。
统一 794 维观测、18 维动作、原生 PPO、独立评测及可恢复检查点已实现。
最新状态与累计资源记录分别见 `status/STATUS.json`、`status/NEXT.md` 和 `status/GPU_LEDGER.json`。
工程通过或存在权重文件均不表示策略达到成功率要求；G04 的两个固定面板必须各达到 90%。

## 开始
1. 将本目录内容放入新的研究工作区；已有项目先建立分支，不覆盖原有 AGENTS.md。
2. 阅读 `configs/budget.json` 和 `status/NEXT.md`。G07原GPU0独立授权新增最多10 GPU-hours，固定窗口北京时间2026-10-23 16:04至10月24日02:04已结束；继续GPU前须明确修订截止，保留历史费用和原T0。
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
