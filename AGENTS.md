# QuadSmash：给 Codex 的执行约束

先读 README.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md、configs/budget.json、
status/STATUS.json，再读当前目标文件。只执行依赖已通过且预算已获授权的 Goal。

## 目标与范围
实现 Go2 + 标准 PiPER + 球拍的耦合仿真，使用事件条件化全身 PPO，先学跳跃恢复，
再学空中虚拟击球，再学真实触球和扣杀恢复。第一阶段不做视觉、VLM、真机控制。
模型具体版本、安装位姿和未标定执行器参数必须写入资产清单及 assumptions.md。
未经实际资产核对，不把标准 PiPER、PiPER H/L/X 混用，也不删除真实质量提高成功率。

## 工作纪律
- 先审计本机版本和参考仓库，不盲目安装 latest；锁定 Git SHA、包版本和资产哈希。
- 不覆盖用户未提交修改，不擅自改系统驱动、全局 Python、权限或共享服务器作业。
- 不运行未经审阅的远程安装脚本，不绕过审批，不创建付费云资源。
- 同一工作区只有一个写入主代理；GPU 调度由一个进程管理，不能重复启动旧作业。
- 不将未运行代码、Mock、单元测试通过或训练 reward 上升称为仿真/学习验收通过。
- 不为通过验收更改测试集、成功定义、资产参数或硬件限制。
- 不能偷偷降低网高、放大拍面、添加悬空外力、瞬移机器人、击球前吸附羽毛球。
- 每个 Goal 必须形成代码→测试→训练（如需要）→独立评测→证据报告的闭环。
- 数值测试容差是显式配置；GPU rollout 不承诺跨硬件逐位复现。

## 预算与阻塞
默认只授权 G00–G03；到 G04 停为 BLOCKED_PERMISSION，等待明确预算授权。
已有 GPU 不等于获准占用所有 GPU；检查分配环境变量/调度器和资源授权。
预算是所有重试、训练、评测 GPU-hours 的累计；重开进程不能清零。最长作业时间和
环境交互步数也要设上限。无 GPU/资产/依赖时标记 BLOCKED，不得用 CPU 假环境冒充。
最多进行预算内三轮有假设的训练修订。物理不可行证据不足时写“未找到可行方案”，
不把优化器失败当成数学不可能证明。资源耗尽为 BUDGET_EXHAUSTED，不是成功。
不要逐步要求用户确认常规代码工作；只有权限、预算、硬件/任务契约改变才暂停询问。

## 持久状态与证据
每次运行保留 command、Git SHA+dirty patch、配置和资产哈希、随机种子、GPU-hour、
环境步数、checkpoint SHA、日志路径、指标和逐回合 CSV。只允许原子写状态文件。
每完成阶段运行：python tools/verify_goal.py --root . --goal Gxx
校验器与 acceptance.json 属于验收契约；发现错误应提交更改建议，不可自改降低门槛。
状态：TODO / IN_PROGRESS / PASSED / FAILED / BLOCKED / BLOCKED_PERMISSION /
BUDGET_EXHAUSTED。CODE_READY 与 TRAINED 是内部进度，不等于 PASSED。
阶段交付报告写明已运行/未运行、失败原因、下一条可执行命令。
训练进程退出前保存可恢复检查点；清理仅限本项目创建的进程。
