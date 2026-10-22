# QuadSmash Codex Goal 完整实施与训练手册

本文件为各独立文件的汇总阅读版；Codex执行以独立文件和configs中的机器可读契约为准。



---

<!-- source: README.md -->

# QuadSmash：Codex /goal 实施与训练任务包

日期：2026-10-21。对象：Go2 + 标准 AgileX PiPER + 固定安装球拍；Isaac Lab；PPO。

**这是一份可交给 Codex 的实施契约，不是已经实现/训练完成的机器人项目。**
当前可执行内容只有 `tools/verify_goal.py`（证据结构与阈值校验）及其 CPU 单元测试。
`quadsmash` 仿真包和 `scripts/train.py` 等入口由 Codex 在 G00–G03 中实现。
没有附带训练权重，没有在当前环境运行 Isaac Lab，也没有宣称能保证达到学习成功率。

## 开始
1. 将本目录内容放入新的研究工作区；已有项目先建立分支，不覆盖原有 AGENTS.md。
2. 阅读 `configs/budget.json`。默认只允许 G00–G03 使用已分配的本地资源，合计上限 2 GPU-hours。
3. 在同一工作区打开 Codex CLI/IDE。先输入 `CODEX_GOAL_PROMPTS.md` 的“第一轮启动目标”。
4. G00–G03 通过后，再显式授权 G04–G07 的本地训练预算并输入第二轮目标。
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
第二条在没有真实 `reports/G00/evidence.json` 时应失败，这正是预期行为。
校验器检查证据文件、摘要、依赖、指标和必需产物；**不独立证明日志来自真实仿真**，
仍需保留原始逐回合数据、执行日志和可复现命令以接受审计。

## 最小可交付边界
G00–G07 = 能审计的跳跃扣杀仿真基线。
G08–G09 = 可恢复状态初始化与恢复约束事件选择研究分支。
G10 = 冻结测试、消融、多种子及结论审计；结果可以是不支持方法优势。


---

<!-- source: AGENTS.md -->

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


---

<!-- source: PLAN.md -->

# 分阶段实施计划

依赖链：G00 → G01 → G02 → G03 → G04 → G05 → G06 → G07 → G08 → G09 → G10。

G00–G03先建设；G04–G07训练最小基线；G08–G09研究增强；G10最终冻结比较。

| Goal | 目标 | 文档 | GPU-hours上限建议 |
|---|---|---|---|
| G00 | 审计环境并锁定兼容软件栈 | [G00](goals/G00_environment.md) | 0.5 |
| G01 | 组装真实耦合资产并审计执行器 | [G01](goals/G01_assets.md) | 0.5 |
| G02 | 验证羽毛球飞行、扫掠碰撞与事件 | [G02](goals/G02_physics.md) | 0.5 |
| G03 | 实现统一任务、训练评测入口与证据链 | [G03](goals/G03_env_ppo.md) | 0.5 |
| G04 | 训练完整负载下跳跃和落地恢复 | [G04](goals/G04_jump.md) | 8 |
| G05 | 训练按时完成空中虚拟击球事件 | [G05](goals/G05_virtual_hit.md) | 12 |
| G06 | 训练真实来球空中接触并恢复 | [G06](goals/G06_real_contact.md) | 16 |
| G07 | 训练完整扣杀、越网落点和恢复 | [G07](goals/G07_smash.md) | 24 |
| G08 | 研究分支：物理一致中间状态初始化 | [G08](goals/G08_reset_bank.md) | 12 |
| G09 | 研究分支：击球与恢复联合事件选择 | [G09](goals/G09_event_selector.md) | 16 |
| G10 | 冻结评测、多GPU实验与最终报告 | [G10](goals/G10_final_study.md) | 80 |

## 每个Goal的执行循环
读取已通过证据→查看本阶段未完成项→实现最小改动→运行单元/仿真测试→小规模训练→
独立开发评测→检查预算和错误类型→必要时一次有假设的修订→写报告和状态。
不能一次性生成所有文件后把阶段都标完成；不能在依赖失败时跳到下阶段掩盖问题。

## 条件失败处理
- 程序异常：保留最小复现，先修代码，不继续加大训练步数。
- 学习无进展：检查动作范围、奖励分项、事件链和执行器饱和；每次只改变有明确假设的项。
- 目标疑似不可行：做空间/速度/起跳与恢复能力图，不自动修改硬件参数。
- 达到预算：保存检查点与下一条resume命令，停止GPU作业并标BUDGET_EXHAUSTED。
- 无方法优势：保留完整结果；工程完成不等于研究主张成立。

## 状态报告模板
当前Goal / 新增文件 / 已运行命令 / 测试结果 / 学习指标 / 使用预算 / 已知阻塞 /
证据路径 / 下一条命令。禁止用“基本完成”替代可核查状态。


---

<!-- source: TRAINING_SPEC.md -->

# 算法与仿真实现规范

本文件中的数值是初始工程建议，不是已验证性能。模型真实值优先于建议；任何影响验收
定义的修改须记录并重新生成版本化协议，不能在看到最终测试结果后追改阈值。

## 1. 环境与版本
G00 决定主线，之后不混用 API：已有兼容环境优先。新建保守主线可评估 Isaac Lab
v2.3.2 + Isaac Sim 5.1 + Python 3.11，并使用该 tag 实际锁定的 RSL-RL，不单独升新版。
若直接复用 LeggedManip_Lab，必须检查其 Git SHA 与依赖；其当前 README 要求更新的
RSL-RL 接口，与 v2.3.2 的原样配置不兼容。不得照抄其 main 与今天的 Isaac Lab main
并假定兼容。可以只借鉴资产和任务结构，在锁定版本上适配。参考源见 docs/SOURCES.md。

选择 DirectRLEnv 风格为主（按锁定版本适配），避免同时维护 Direct 和 Manager 两套任务。
物理步、动作解码、事件检测和奖励必须分离。按实际源码检查子步钩子，不假设
_pre_physics_step 等于每一个物理子步；不随意复制并改坏整个 step()/自动 reset 逻辑。

## 2. 必须实现的代码模块
```text
source/quadsmash/
  assets/{combined_robot.py,manifest.py}
  physics/{shuttle.py,swept_contact.py,impact.py}
  envs/{smash_env.py,event_state.py,observations.py,rewards.py,resets.py}
  learning/{ppo_config.py,curriculum.py,reset_bank.py,feasibility.py,event_selector.py}
  evaluation/{metrics.py,protocol.py,report.py}
scripts/
  audit.py smoke.py validate_assets.py validate_physics.py
  train.py evaluate.py collect_reset_bank.py fit_feasibility.py run_study.py stop_run.py
```
这里的 scripts 命令是 Codex 应实现的本项目 CLI 契约，不是 Isaac Lab 原生命令。

## 3. 模型和执行器
一个浮动基座耦合系统：Go2 机身/腿 + 安装板 + 标准 PiPER + 工具。
禁止通过每步 set_world_pose 让独立机械臂跟着机身；必须传递惯性与约束反作用。
预期动作是 12 腿 + 6 臂 = 18 维；实际关节名/索引从模型核对，夹爪固定或移除后
如实记录质量差别，mimic joints 不进入独立动作向量。

动作先采用受限关节目标：q_des=q_nominal+scale*a。
PD/执行器模型逐组定义刚度、阻尼、速度与力矩限制、延迟；不能使用无限力矩。
未知参数可以用于标注为 assumed 的仿真研究，但不能声称实机参数已标定。
保存受限前命令、受限后命令、实际状态和饱和统计。

起点：physics_dt=0.002 s，policy_dt=0.010 s，decimation=5。
用 0.001/0.0005 s 做数值复核；维持 policy_dt 不变，相应改变 decimation。
不单纯增大步长节省算力以牺牲高速接触正确性。渲染独立、训练通常关闭。

## 4. 羽毛球与碰撞
第一版：p_dot=v，v_dot=g-k*||v-wind||*(v-wind)，k 有单位 1/m。
气动参数、球拍质量和碰撞参数分别标注测量/厂商/估计来源。支持 k=0 解析验证。
首选 CPU 可独立测试的向量化飞行模块，再提供 GPU 版本。旋转气动是后续扩展。

实际接触点速度 = 工具参考点线速度 + 角速度 × 接触点偏置。
对每个物理子步做移动/转动拍面的扫掠检测；必要时自适应子划分或求根，不能仅比较
两个采样时刻的距离。接触必须在实际有效拍面边界内，使用接近方向与分离滞回。
一个接触只算一次。检查恢复系数定义、运动拍面坐标系和相对法向速度。

自定义接触和引擎接触二选一处理同一次碰撞，不得双计冲量。
若使用自定义响应，处理与验证机器人反作用冲量；若先采用无限质量拍面近似，必须
显式标注并量化其误差，不得拿它证明完整全身碰撞正确。若无法实现该版本中可靠的
冲量耦合，阻塞物理验收，而非悄悄忽略。网与地面首次越界事件也使用插值/扫掠。

## 5. 状态、动作和事件目标
世界坐标固定 x 朝对场、y 横向、z 向上。明确 quaternion order，写往返变换测试。
策略坐标使用 yaw 对齐任务系或机身系，所有速度变换考虑参考点与角速度项。
目标 g=(time_to_hit, position_hit, normal_hit, velocity_hit)。所有阶段使用统一观测形状；
无球阶段用 mask+零值，而不是改网络输入尺寸。阶段 one-hot 由真实事件更新。
阶段：prepare / push_off / flight_pre_hit / flight_post_hit / recovery。

Actor：可观测/可估计本体状态、球拍状态、接触、当前/历史来球、命令目标、上次动作。
Critic：可附加真实质量、延迟、摩擦和无噪声状态。未来真实接触时刻或未来真轨迹不得
泄漏到 actor；目标时刻可以作为命令输入，预测轨迹只能由当前估计生成。
第一版 MLP + 约 0.1 s 历史；后续再比较 GRU。

## 6. PPO 起始配置
Actor 隐层 [512,256,128]，Critic [512,256,128]；18维对角高斯策略，共享全身目标。
先用标准 RSL-RL PPO，不重写 PPO。对接对应版本的观测归一化和非对称 critic 接口。
学习率 3e-4，clip 0.2，epochs 5，mini-batches 4，gamma 0.997，GAE lambda 0.98，
初始 entropy coefficient 0.005，max_grad_norm 1.0，rollout_length 128。
网络和系数均为调试起点。统计 KL、clip fraction、价值误差、熵、梯度范数、饱和率。
num_envs 从 32/128 冒烟，再 512，显存允许再到 1024/2048，不固定强求4096。

所有时间奖励用秒定义：连续奖励按 dt 积分，接触/成功奖励只发一次。
GAE 正确区分 terminated 与 truncated；时间截断使用 reset 前 final_observation bootstrap。
保存 actor、critic、normalizers、optimizer、scheduler、curriculum、计数器和随机状态。
--init-checkpoint 表示迁移权重/兼容归一化，重置新阶段优化器与训练计数；
--resume-checkpoint 表示完整恢复同一实验。严格校验接口/资产/动作索引和配置兼容性。

## 7. 奖励与成功
无球阶段：在指定时刻附近按位置/拍面/接触点速度误差给时间窗奖励。
位置容差可从 0.12m 渐变到0.04m；方向从30度到10度；时间窗从80ms到20ms。
这是训练 shaping，不改变最终评测或物理拍面。速度目标先取可行范围内低值，逐步提高。
不可每一帧强奖靠近目标，导致提前停拍。奖励速度必须同时匹配方向和位置。

真实来球阶段：只在真实接触后给攻击回报，用碰撞后的出球速度/方向，不用指令速度替代。
最终主成功 = 空中有效触球 AND 合格出球 AND 越网落入目标区域 AND 稳定恢复。
实际足端离地、接触力和滞回决定 airborne，不由命令阶段决定。
触球后继续到机器人恢复窗口与羽毛球结果均判定。超时未判定为失败/删失单独记录，
不能默认成功；发球命中身体、二次触球等无效事件要独立记录。

代价：力矩、能量、动作变化、自碰撞、危险机身接触和落地冲击。
不能全程强罚竖直速度、倾角、偏离站姿；这些可能是完成任务所必需。
硬执行器限制始终存在；平滑正则可后期增强。报告真实越限而不只报告命令限幅。

## 8. 课程
G04 带完整机械臂跳跃恢复 → G05 无球空中事件跟踪 → G06 真实来球空中接触恢复 →
G07 真实扣杀和恢复。每个阶段继承上一阶段初始化，但不永久冻结腿或臂。
逐级扩大事件分布，不同时放开所有几何/动力学扰动。先固定名义动力学，后做轻扰动，
最后 G10 测组合未见扰动。不得在最终 holdout 上调参。

可恢复状态库在 G08 做正式消融；若G05长期没有有效事件，可在G05当前预算内提前实现
该模块作为明确标记 `ppo_resetbank_rescue` 的分支，而不是偷偷替换 `ppo_plain` 基线。
必须先通过以下恢复一致性测试，不能把G08整阶段标为通过。正式比较仍在G08从匹配
起点重训，计入采集成本。
状态来自真实 rollout，保存 base/joints/工具/球、命令、阶段、
时钟、延迟缓冲、历史观测和执行器状态，以及来源模型/策略哈希。优先采集无足地接触
的空中状态，避免假装能够恢复引擎不可访问的接触求解器内部状态。
用连续执行 vs 恢复执行对照检查漂移容差；不要求跨硬件 bitwise 相同。
只在 episode 开始 restore，不允许回合中传送。最多50%辅助重置，后期降至20%或0%；
所有验收从地面正常初始化。旧 rollout 不直接作为 on-policy PPO 的训练数据。

## 9. 事件可行性与选择（G09）
冻结某一低层策略采集 (state, goal, success)；标签必须覆盖击球+恢复。
按整条 episode/初始场景分组划分 fit/calibration/validation，不能按相邻帧随机切分。
建议小型 ensemble 估计成功概率，做独立校准并报告 false-safe rate 和 coverage。
策略改变后刷新数据或标注策略版本；旧模型不能作为新策略的绝对能力界限。
沿预测来球轨迹生成约64个候选，先做物理必要条件筛选，再按攻击收益、代价和保守
成功估计选择。没有合适候选时明确 abstain/保守回球；不能靠全部 abstain 虚假安全。
不声称学习成功概率是严格安全证明。低层收敛前不引入另一个高层 PPO。

## 10. CLI 契约（由 Codex 实现）
```bash
python scripts/train.py --stage jump_recover --seed 0 --num-envs 1024 \
  --max-env-steps 200000000 --max-gpu-hours 8 --run-dir runs/G04/seed0
python scripts/evaluate.py --stage jump_recover --checkpoint artifacts/G04/policy.pt \
  --suite dev_jump_v1 --episodes 200 --reset-mode ground --output reports/G04/eval
python scripts/stop_run.py --run-dir runs/G04/seed0 --save-and-stop
```
--max-env-steps 计所有环境、所有rank、所有重试累计交互；不是每个环境步数。
GPU-hour = 实际占用GPU数量 × 墙钟小时；计训练、采样、评测与失败重试。
训练入口默认headless，render/video只在明确评测请求时开启。
训练自带截止检查并定期checkpoint，SIGTERM可保存退出；SIGKILL不能保证即时保存。
启动前记录PID/作业ID和设备，重连时先查旧作业，禁止双重启动。

## 11. 评测与研究
开发验收集在G03冻结，G04/G05仅允许使用来自预先规则的可行目标采样器。
最终holdout在G10之前不读取结果。最终任务只从地面初始化，不启用训练辅助。
报告数值均是实测；没有学会时保留检查点、曲线和失败分布。
实验基线共享任务、观测、课程、硬件约束；计入离线可行样本与状态库采集成本。
G10至少比较：统一PPO；+状态库；+事件选择无恢复；完整方法。
允许研究结论为无优势或样本不足，不允许为了positive result更改协议。


---

<!-- source: ACCEPTANCE.md -->

# 验收协议

## 三类结果必须分清
1. 工程就绪：接口、测试、数据流可运行。
2. 策略达标：冻结检查点在指定开发验收分布达到数字门槛。
3. 研究主张成立：与预算/信息公平的基线比较，有多种子证据。
G00–G03只证明第1类；G04–G07要求第2类；G08–G10不能以“必须有正向提升”操纵实验。
所有阈值是本项目初始目标，不是文献结果、硬件保证或统计显著性标准。

## 固定条件
- 数据源版本、任务suite ID、初始化规则、网高、球拍尺寸、动力学和控制限幅必须入哈希。
- 每个学习Goal至少200个地面初始回合；G04–G07建议两个预先固定开发面板各200回合，
  验收指标使用两个面板较差的比例，sample_count为总回合数≥400。
- 频繁开发评估允许用于调参；不得将其说成未见最终测试。
- 最终G10至少5个训练种子，每个方法/种子至少1000个固定holdout回合。
- 视频展示固定案例/随机抽样成功与失败，禁止仅挑选成功片段代替数值验收。
- NaN、非法维度、重置串扰、虚假接触、重复冲量及未限幅执行器都是硬失败。
- G01–G07都使用完整负载；debug资产/缩放球网只能标为独立实验，不可冒充主线。

## 主要目标
G04：总质心离地后上升至少0.08m、四足实际腾空至少0.10s、稳定落地恢复，
     两开发面板成功率均≥0.90。高度指总质心相对起跳瞬间，不是机身坐标的假跳跃。
G05：指定时刻球拍位置误差≤0.04m、法向误差≤10°、速度误差≤max(0.5m/s,0.2|v_goal|)、
     事件时间误差≤20ms，且空中发生和随后恢复；联合成功率≥0.70。
G06：真实羽毛球接触发生于腾空、随后稳定恢复；联合成功率≥0.60。
     本阶段不要求攻击性，不得把它命名为“已学会扣杀”。
G07：空中击球 + 净出球速度≥8m/s + 出球vz≤-1m/s + 实际越网 + 落入目标区域 + 恢复，
     联合成功率≥0.50，同时空中触球率≥0.70，P(恢复|空中触球)≥0.85。
     8m/s等是初始研究目标，不是PiPER实测能力。如果能力审计不支持该要求，应提交
     单独变更提案，不能自动把阈值降到本次恰好达到的值。

稳定恢复：任务后接触恢复，连续0.8s满足已冻结的姿态、根速度、关节速度和非机身着地
限制。允许落地瞬间姿态较大，不能用持续水平姿态约束阻止合法动态动作。
默认恢复 roll/pitch≤15°、水平根速度≤0.3m/s、机身角速度≤1rad/s；实际阈值在G03验证。
足端腾空判定使用接触力阈值、离地距离与滞回，阈值由G01接触标定冻结，不用单帧信号。

G07主线球网：中心高1.524m、侧边1.55m，网面几何及位置独立配置；早期低网仅debug。
若在该几何和实际执行器能力下尚未找到可行攻击，报告受限区域与失败证据，不能抬升
地面、移低球网、延长球拍、减重或增加力矩后仍使用原实验名称。

## 证据目录
每个阶段：
```
reports/Gxx/
  evidence.json
  REPORT.md
  commands.log
  ...逐回合指标、测试和评测配置...
artifacts/Gxx/policy.pt    # 需要训练的阶段
```
`evidence.json`使用`docs/EVIDENCE_SCHEMA.md`。由真实测试/评测程序生成，不能手写成功指标。
`tools/verify_goal.py`仅作结构、哈希、依赖与门槛检查，原始数据仍须可重算。

## 状态与停止
PASSED需满足本阶段全部门槛和证据。FAILED用于已执行但未达标；BLOCKED用于依赖/资产/
技术缺失；BLOCKED_PERMISSION用于未授权；BUDGET_EXHAUSTED用于预算不足。
缺少实测文件不能被“代码编译通过”“应该可以运行”代替。
同一性能失败经过3轮有记录修订仍无进展就提交诊断；不要求无限尝试。

## 研究验收
G08的可恢复状态库和G09事件选择即使没有提升，也可以在功能正确、实验完整时工程通过。
报告必须分列 `engineering_status` 与 `research_conclusion`。
G10方法优势由均值、每种子结果、置信区间和训练成本决定，允许`not_supported`。
不得为让G10通过而要求一定优于baseline。没有足够算力完成预注册对照则标部分完成。


---

<!-- source: CODEX_GOAL_PROMPTS.md -->

# 直接粘贴进 Codex 的 /goal

先在GPU服务器或可访问该工作区的Codex会话中打开项目。不要在普通shell中执行/goal。
若Codex不识别此命令，按官方文档启用Goal或更新支持该功能的客户端。

## 第一轮：默认只实现与验证G00–G03
```text
/goal 在当前工作区实现 QuadSmash 的 G00–G03。先读取 AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md、configs/budget.json 和 status/STATUS.json，依次执行 goals/G00_environment.md 至 goals/G03_env_ppo.md。直接实现、运行测试并验证，不要只输出计划。只使用已分配的本地GPU，累计最多2 GPU-hours，不修改系统驱动、不创建云资源；没有GPU分配时先完成CPU代码与检查，再报告阻塞。每阶段生成真实证据并运行 tools/verify_goal.py；依赖未通过不进入下一阶段。所有G00–G03通过后停止，或遇到权限/技术阻塞/预算上限时保存状态并停止。不要启动G04训练，也不要把Mock或CPU测试当作Isaac仿真通过。
```

## 第二轮：显式授权第一版训练G04–G07
阅读并认可预算后再使用以下命令；这会授权最多新增60 GPU-hours，不保证达到指标。
```text
/goal 先检查G00–G03真实验收证据。现在授权在既有、已分配的本地GPU上依次执行G04–G07，更新configs/budget.json对应授权：新增累计60 GPU-hours，各阶段8/12/16/24 GPU-hours，同时只运行1个GPU训练进程，不得创建云资源。遵循TRAINING_SPEC.md和对应goals文件，实现并训练完整负载跳跃恢复→空中虚拟击球→真实空中接触恢复→扣杀越网和恢复。所有验收从地面初始化，阈值/球网/质量/力矩限制不得降低。每阶段实测通过tools/verify_goal.py后才继续；最多3轮有记录的修订，达到任一预算或发生阻塞即保存检查点、停止自己的GPU作业并报告。G07通过或明确阻塞后停止，不启动G08。
```

## 第三轮：研究增强G08–G09
```text
/goal 在G00–G07证据有效的前提下，执行G08–G09。授权新增累计28 GPU-hours：G08最多12、G09最多16；只使用已分配本地GPU，最多2个独立实验进程。依次实现物理一致状态库、同预算对照、击球与恢复联合可行性预测及候选事件选择。禁止使用最终测试集调参，禁止把旧rollout当普通PPO on-policy数据，禁止把学习成功概率称为安全证明。报告工程完成状态与研究结论，允许无提升。预算内完成对应goals文件所有证据才标通过，预算不足则保存状态停止，不自动开展G10。
```

## 第四轮：冻结实验G10
```text
/goal 先读取并冻结G10的实验协议、方法列表、5个训练种子和每种子1000回合holdout；禁止读取holdout结果后修改方法。授权新增最多80本地GPU-hours与2个并行实验进程，先运行dry-run预算清单，不创建云资源。按goals/G10_final_study.md完成全部预注册实验和报告，计入预训练、状态库和可行性数据成本。研究结果不必为正，但不得减少种子或修改成功条件来伪装完成。预算不足或技术阻塞时保存可重现部分结果、标未完成并停止。
```

## 单阶段推进（更易定位）
使用下面命令不自动新增预算；阶段必须已在budget.json授权，否则报告BLOCKED_PERMISSION。

### G00 审计环境并锁定兼容软件栈
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G00_environment.md，完成G00，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G00/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G01 组装真实耦合资产并审计执行器
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G01_assets.md，完成G01，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G01/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G02 验证羽毛球飞行、扫掠碰撞与事件
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G02_physics.md，完成G02，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G02/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G03 实现统一任务、训练评测入口与证据链
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G03_env_ppo.md，完成G03，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G03/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G04 训练完整负载下跳跃和落地恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G04_jump.md，完成G04，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G04/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G05 训练按时完成空中虚拟击球事件
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G05_virtual_hit.md，完成G05，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G05/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G06 训练真实来球空中接触并恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G06_real_contact.md，完成G06，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G06/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G07 训练完整扣杀、越网落点和恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G07_smash.md，完成G07，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G07/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G08 研究分支：物理一致中间状态初始化
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G08_reset_bank.md，完成G08，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G08/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G09 研究分支：击球与恢复联合事件选择
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G09_event_selector.md，完成G09，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G09/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G10 冻结评测、多GPU实验与最终报告
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G10_final_study.md，完成G10，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G10/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

## 会话中断后的继续
```text
/goal 读取status/STATUS.json、status/DECISIONS.md、status/RUNS.jsonl和最后一个Goal报告，先查询已记录PID/Job ID是否仍在运行，禁止重复启动。校验预算余额与checkpoint哈希，在现有授权内继续首个未通过的Goal。已通过且哈希未变化的阶段不重复训练。达到该Goal验收或阻塞/预算边界后保存证据停止。
```

## 查看、暂停、恢复
```text
/goal
/goal pause
/goal resume
/goal clear
```
训练进程另由 scripts/stop_run.py --save-and-stop 或集群调度器管理；先让本项目进程保存
检查点，再暂停会话。不要对共享服务器使用killall/pkill python。


---

<!-- source: goals/G00_environment.md -->

# G00：审计环境并锁定兼容软件栈

前置：无。证据task_id：`environment_audit`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

检查当前工作区、Git改动、GPU分配、驱动、Python、Isaac Sim/Lab、RSL-RL与资产路径。
不要自动占用他人的GPU，不先执行远程一键安装。选择兼容版本并锁定SHA与完整依赖。
优先运行锁定版本的官方简单任务：32环境、至少200策略步、PPO至少2次更新、保存和
重载策略。建立项目包及scripts/audit.py，登记代码和依赖许可。禁止用FakeEnv通过。
本阶段checkpoint只是训练链路冒烟，不代表机器狗学会任何动作。

命令契约：`python scripts/audit.py --output reports/G00/environment.lock.json`
再运行对应官方任务，将原始命令、日志和检查点重载结果写入证据。

无GPU、无分配权限、缺Isaac许可证/依赖或导入失败时BLOCKED，并给可复现错误。
若参考仓库依赖不兼容，迁移少量资产/配置或选择兼容提交，不循环升级整个服务器。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `gpu_smoke_ok` | `eq True` |
| `dependency_lock_ok` | `eq True` |
| `checkpoint_reload_ok` | `eq True` |

## 必须交付

- `reports/G00/REPORT.md`
- `reports/G00/commands.log`
- `reports/G00/environment.lock.json`

另外生成 `reports/G00/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G00`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G01_assets.md -->

# G01：组装真实耦合资产并审计执行器

前置：G00。证据task_id：`asset_audit`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

加载Go2+标准PiPER+固定工具；核对18个主动关节，所有刚体质量/质心/惯量、关节限制、
工具安装位置和实际球拍接触点。夹爪处理必须与后续资产一致。组合为一个正确耦合的
浮动基座系统；记录模型来源与哈希，逐项区分真实参数和假设参数。
运行100次、每次10s站立及低速摆臂；至少95%站立成功。对腿/臂分别做小幅关节映射测试。
无地面接触的诊断中，确认机械臂动作会引起机身反应，并检查总质心/角动量数值一致性；
不把姿态传送当作动力学连接。标定足接触阈值、碰撞几何和关节限幅。
能力审计先覆盖可达空间、实际拍点速度、负载与静态力矩；动态起跳能力由G04验证。

`python scripts/validate_assets.py --num-envs 32 --episodes 100 --output reports/G01`

无法核实安装板/球拍参数时允许写入显式估计继续仿真，但必须标注未实机校准。
若官方模型缺质量/惯量，不得填零或无限小值，报告并请求合法参数/建模决策。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `joint_mapping_ok` | `eq True` |
| `coupled_dynamics_ok` | `eq True` |
| `asset_manifest_ok` | `eq True` |
| `stand_success_rate` | `ge 0.95` |
| `sample_count` | `ge 100` |
| `stand_episode_seconds` | `ge 10.0` |

## 必须交付

- `reports/G01/REPORT.md`
- `reports/G01/commands.log`
- `reports/G01/assets_manifest.json`
- `reports/G01/asset_tests.json`

另外生成 `reports/G01/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G01`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G02_physics.md -->

# G02：验证羽毛球飞行、扫掠碰撞与事件

前置：G01。证据task_id：`physics_validation`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

先在独立CPU单元测试实现重力+二次阻力、运动/旋转拍面扫掠碰撞和冲量响应，再对接GPU。
定义不少于50个解析接触/不接触/擦边/旋转/离开后再接触案例。对无阻力轨迹与解析解
比较；无重力有阻力时检查相对空气速度衰减；旋转拍面验证接触点速度。
禁止重复冲量；自定义模式处理等量反作用并与引擎接触互斥。球网和落地也做扫掠事件。
用dt=2/1/0.5ms、同policy_dt做收敛性检查，明确近边界不稳定案例。
非边界接触基准建议出球速度差≤5%、方向差≤5°、接触时刻差≤2ms；解析单元测试使用
更紧容差并公开。不能靠改球拍碰撞体让测试通过。

`python scripts/validate_physics.py --num-envs 32 --output reports/G02`

超高速穿透时先修检测/时间步，不用更大的隐形拍面；反作用API不可用则修接口或阻塞。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `analytic_flight_ok` | `eq True` |
| `swept_contact_ok` | `eq True` |
| `no_double_impulse` | `eq True` |
| `reaction_impulse_ok` | `eq True` |
| `timestep_convergence_ok` | `eq True` |

## 必须交付

- `reports/G02/REPORT.md`
- `reports/G02/commands.log`
- `reports/G02/physics_tests.json`

另外生成 `reports/G02/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G02`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G03_env_ppo.md -->

# G03：实现统一任务、训练评测入口与证据链

前置：G02。证据task_id：`environment_ppo_smoke`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：2,000,000（0表示本阶段无正式学习预算）。

实现TRAINING_SPEC中的Direct任务、统一18维动作/观测mask、真实事件机、正确自动reset、
PPO包装器、train/evaluate/stop_run入口和预算检查。32环境运行至少2000策略步，涵盖
部分reset、超时、失败、触球事件、羽毛球未落地但机器人已恢复等情况。
PPO至少10次更新验证梯度有限、checkpoint保存/加载；测试中断恢复和正常终止bootstrap。
固定dev验证规则、采样种子、目标分布版本及未来holdout生成规则。
所有训练CLI接收--stage/--seed/--num-envs/--max-env-steps/--max-gpu-hours/--run-dir，
评测接收--checkpoint/--suite/--episodes/--reset-mode/--output。配置缺失应fail-fast。
实现从原始CSV自动生成evidence.json，记录所有配置、代码、资产与权重哈希。

`python scripts/smoke.py --num-envs 32 --policy-steps 2000 --output reports/G03`
`python scripts/train.py --stage smoke --num-envs 32 --seed 0 --max-env-steps 2000000 --max-gpu-hours 0.5 --run-dir runs/G03/smoke`

不要把本阶段随机策略reward上升解释成学会动作。默认预算授权到此结束。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `env_contract_ok` | `eq True` |
| `partial_reset_isolation_ok` | `eq True` |
| `timeout_bootstrap_ok` | `eq True` |
| `checkpoint_resume_ok` | `eq True` |
| `evaluation_protocol_frozen` | `eq True` |

## 必须交付

- `reports/G03/REPORT.md`
- `reports/G03/commands.log`
- `reports/G03/interface_contract.json`
- `reports/G03/evaluation_manifest.json`

另外生成 `reports/G03/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G03`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G04_jump.md -->

# G04：训练完整负载下跳跃和落地恢复

前置：G03。证据task_id：`jump_recover_ground_v1`。
累计预算建议：8 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

无羽毛球，从地面正常初始化。PiPER/工具质量完整保留，初期臂姿态软正则随后减弱。
学习准备、蹬地、总质心升高、四足腾空、腿部落地准备与恢复。先不随机大范围质量/延迟。
目标逐步从短小跳跃到验收要求：起跳后总质心升高≥0.08m、腾空≥0.10s、恢复≥0.8s。
奖励避免全程惩罚竖直速度/倾角。真实关节扭矩/速度限制不随课程放宽。

`python scripts/train.py --stage jump_recover --seed 0 --num-envs 1024 --max-env-steps 200000000 --max-gpu-hours 8 --run-dir runs/G04/seed0`
`python scripts/evaluate.py --stage jump_recover --checkpoint artifacts/G04/policy.pt --suite dev_jump_v1 --episodes 400 --reset-mode ground --output reports/G04`

两个固定开发面板各200回合，记录较差面板成功率≥90%，不能使用空中reset计入此门槛。
若不离地先查实际力矩/负载/动作空间和站姿惩罚；若落地失败查终止时机、恢复奖励和
足端准备；若没有可行起跳证据，先做搜索与负载诊断，不增加力矩或减轻PiPER。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `jump_recovery_success_rate` | `ge 0.9` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G04/REPORT.md`
- `reports/G04/commands.log`
- `reports/G04/episodes.csv`
- `reports/G04/eval_manifest.json`
- `artifacts/G04/policy.pt`

另外生成 `reports/G04/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G04`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G05_virtual_hit.md -->

# G05：训练按时完成空中虚拟击球事件

前置：G04。证据task_id：`virtual_hit_recover_ground_v1`。
累计预算建议：12 GPU-hours；所有尝试累计环境交互上限：300,000,000（0表示本阶段无正式学习预算）。

由G04权重初始化，不冻结腿。仍无羽毛球，给定目标(time,position,normal,velocity)。
目标来自已验证可达区域及动态rollout，不能独立随机采样出大量明显不可能的组合。
先学位置，再方向，再接触点速度与精确时序；训练时间窗80→20ms，位置12→4cm。
先用可行低速目标，后扩大；最终目标分布必须在训练前固定，不按失败情况删难题。

`python scripts/train.py --stage virtual_hit --init-checkpoint artifacts/G04/policy.pt --seed 0 --num-envs 1024 --max-env-steps 300000000 --max-gpu-hours 12 --run-dir runs/G05/seed0`
`python scripts/evaluate.py --stage virtual_hit --checkpoint artifacts/G05/policy.pt --suite dev_virtual_v1 --episodes 400 --reset-mode ground --output reports/G05`

联合成功至少70%：空中满足位置/方向/速度/时间全部容差并稳定恢复。
若连续多次评估没有有效事件，可在本阶段预算内提前实现TRAINING_SPEC的状态库救援，
先完成恢复一致性单元/仿真验证，使用`ppo_resetbank_rescue`独立分支并记成本；最终仍
全部地面初始化，不能将它当作`ppo_plain`基线或宣称G08对照已完成。
若摆到目标却无速度，检查奖励时间窗及速度方向；若臂挥得好却落地失败，检查恢复
任务是否真的参与完整episode；不得永久冻结腿以回避协调问题。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `virtual_hit_recovery_success_rate` | `ge 0.7` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G05/REPORT.md`
- `reports/G05/commands.log`
- `reports/G05/episodes.csv`
- `reports/G05/eval_manifest.json`
- `artifacts/G05/policy.pt`

另外生成 `reports/G05/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G05`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G06_real_contact.md -->

# G06：训练真实来球空中接触并恢复

前置：G05。证据task_id：`airborne_contact_recover_ground_v1`。
累计预算建议：16 GPU-hours；所有尝试累计环境交互上限：300,000,000（0表示本阶段无正式学习预算）。

将虚拟事件替换为从episode开始独立积分的真实羽毛球。通过初始发球条件采样与前向
预测生成经过可行区域的球，不能在接近球拍时修正球的位置、速度或时间。
从G05初始化。先小范围来球，第一次真实接触奖励+恢复奖励；尚不要求攻击性越网。
记录预测误差、接触时间误差、真实拍点速度、冲量与落地结果。

`python scripts/train.py --stage real_contact --init-checkpoint artifacts/G05/policy.pt --seed 0 --num-envs 1024 --max-env-steps 300000000 --max-gpu-hours 16 --run-dir runs/G06/seed0`
`python scripts/evaluate.py --stage real_contact --checkpoint artifacts/G06/policy.pt --suite dev_contact_v1 --episodes 400 --reset-mode ground --output reports/G06`

两个开发面板的空中接触+恢复联合成功率均≥60%。无真实接触的shaping不能计成功。
失败时先区分：虚拟事件执行误差、来球预测误差、物理接触误差；不可把三者都归为PPO差。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `airborne_contact_recovery_success_rate` | `ge 0.6` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G06/REPORT.md`
- `reports/G06/commands.log`
- `reports/G06/episodes.csv`
- `reports/G06/eval_manifest.json`
- `artifacts/G06/policy.pt`

另外生成 `reports/G06/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G06`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G07_smash.md -->

# G07：训练完整扣杀、越网落点和恢复

前置：G06。证据task_id：`airborne_smash_recover_net1524_v1`。
累计预算建议：24 GPU-hours；所有尝试累计环境交互上限：500,000,000（0表示本阶段无正式学习预算）。

从G06初始化。使用真实出球、球网与落点，优化向下出球≥1m/s、总出球速度≥8m/s、
实际越网、目标区域落点，并在机器人恢复和球结果均可判断后结束回合。
主线网高中间1.524m，侧边1.55m；低网debug不得通过本Goal。目标是攻击与恢复而非跳高。
允许训练先使用攻击收益渐进权重，但评测始终同一固定协议。回合超时未判定不计成功。

`python scripts/train.py --stage jump_smash --init-checkpoint artifacts/G06/policy.pt --seed 0 --num-envs 1024 --max-env-steps 500000000 --max-gpu-hours 24 --run-dir runs/G07/seed0`
`python scripts/evaluate.py --stage jump_smash --checkpoint artifacts/G07/policy.pt --suite dev_smash_net1524_v1 --episodes 400 --reset-mode ground --output reports/G07`

完整成功≥50%，空中触球≥70%，触球条件恢复率≥85%；用每个开发面板分别计算，取较差。
如果触球点低于净空要求导致向下球无法越网，做几何/动力学诊断；不得自动降网高、
增大力矩或只选特别容易来球。在真实能力限制下尚未达标须如实报告。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `sample_count` | `ge 400` |
| `ground_init_fraction` | `eq 1.0` |
| `complete_smash_success_rate` | `ge 0.5` |
| `airborne_contact_rate` | `ge 0.7` |
| `recovery_given_airborne_contact` | `ge 0.85` |
| `numerical_failures` | `eq 0` |

## 必须交付

- `reports/G07/REPORT.md`
- `reports/G07/commands.log`
- `reports/G07/episodes.csv`
- `reports/G07/eval_manifest.json`
- `artifacts/G07/policy.pt`

另外生成 `reports/G07/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G07`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G08_reset_bank.md -->

# G08：研究分支：物理一致中间状态初始化

前置：G07。证据task_id：`reset_bank_ablation_v1`。
累计预算建议：12 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

从G04/G05 rollout提取物理一致中间状态，不从随机空中摆姿态生成。保留历史、执行器
延迟、目标时间和球状态；不能假设引擎内部接触缓存可无损恢复，优先无接触空中状态。
建立恢复执行与连续执行的数值对照。训练可在episode开始抽取中间状态，最多50%，
末期降至20%或0%；评测全部地面初始化。旧数据不直接混入on-policy PPO更新。
以G05相同初始权重/目标分布/种子和总采样预算做with/without reset bank试验；至少3种子
小规模pilot，采集状态库成本必须计入。此阶段是从匹配起点重训，不是拿G07强策略对比
G05弱策略。若预算不足3种子则未完成对照，保留部分结果。

`python scripts/collect_reset_bank.py --checkpoint artifacts/G04/policy.pt --output artifacts/G08/reset_bank`
`python scripts/run_study.py --study reset_bank --seeds 0 1 2 --budget-config configs/budget.json --output reports/G08`

工程验收要求接口正确、数据与对照完整，不要求一定提升。结果为no_gain也应保留。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `restore_consistency_ok` | `eq True` |
| `ground_only_eval_ok` | `eq True` |
| `on_policy_update_ok` | `eq True` |
| `ablation_report_complete` | `eq True` |

## 必须交付

- `reports/G08/REPORT.md`
- `reports/G08/commands.log`
- `reports/G08/reset_bank_manifest.json`
- `reports/G08/ablation.json`

另外生成 `reports/G08/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G08`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G09_event_selector.md -->

# G09：研究分支：击球与恢复联合事件选择

前置：G08。证据task_id：`recovery_aware_event_selection_v1`。
累计预算建议：16 GPU-hours；所有尝试累计环境交互上限：200,000,000（0表示本阶段无正式学习预算）。

冻结G07低层策略或固定训练版本，采样(state,event_goal,complete_success)。标签必须包含
实际击球+攻击/回球+恢复，失败也保留。按整条episode/场景分组划分训练、校准和验证。
训练小ensemble或单模型校准成功概率，报告Brier/ECE、false-safe、precision/coverage。
沿预测轨迹采样约64候选，使用物理必要条件筛选和可恢复成功概率选择，不强加高层RL。
允许abstain/保守回球并报告覆盖率，不能通过拒绝全部任务获得漂亮安全数据。
比较固定事件、只击球评分、击球+恢复评分；信息与预算相同。低层策略变更后，能力模型
必须刷新或重新评估。概率不是安全证明，不强求本阶段方法必优于baseline。

`python scripts/fit_feasibility.py --policy artifacts/G07/policy.pt --split-by scenario --output artifacts/G09`
`python scripts/run_study.py --study event_selector --seeds 0 1 2 --budget-config configs/budget.json --output reports/G09`

若没有足够正样本或校准不可靠，报告并阻塞在线使用，不把训练集准确率当泛化能力。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `grouped_split_ok` | `eq True` |
| `probability_calibration_reported` | `eq True` |
| `abstention_coverage_reported` | `eq True` |
| `policy_version_matched` | `eq True` |
| `selector_ablation_complete` | `eq True` |

## 必须交付

- `reports/G09/REPORT.md`
- `reports/G09/commands.log`
- `reports/G09/dataset_manifest.json`
- `reports/G09/calibration.json`
- `reports/G09/selector_ablation.json`

另外生成 `reports/G09/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G09`；通过后才更新STATUS为PASSED。


---

<!-- source: goals/G10_final_study.md -->

# G10：冻结评测、多GPU实验与最终报告

前置：G09。证据task_id：`frozen_holdout_study_v1`。
累计预算建议：80 GPU-hours；所有尝试累计环境交互上限：5,000,000,000（0表示本阶段无正式学习预算）。

预注册并冻结比较：统一PPO；+状态库；+事件选择无恢复；完整方法。至少5训练种子，
每方法每种子1000固定holdout回合。必须计入预训练、库采集和可行性数据成本，报告两种
口径：端到端成本，以及共享预训练已给定时的增量成本。记录研究结论，不强求正提升。
未见来球组合、目标、延迟、工具惯量与摩擦分开测试；所有部署actor不能看真实随机化参数。
给出均值、种子间分布/置信区间、条件成功率、攻击质量、安全约束、样本及GPU-hour。
优先多GPU独立种子；单任务稳定后才分布式。每rank独立seed和env，统计global steps，
验证一GPU与多GPU batch规模变化，不承诺线性加速。
做dt减半冻结策略测试；跨引擎机器人验证是可选扩展，不把共用羽毛球代码当作独立验证。

`python scripts/run_study.py --study final --protocol configs/study_frozen.json --budget-config configs/budget.json --output reports/G10`

study_frozen.json需在本阶段启动前生成审核冻结。预算默认80 GPU-hours仅建议上限，
不保证完成全组训练；预算不足时报告未完成，不能减少种子后仍声明正式验收通过。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `training_seeds_per_method` | `ge 5` |
| `test_episodes_per_seed` | `ge 1000` |
| `matched_compute_accounting_ok` | `eq True` |
| `holdout_not_used_for_tuning` | `eq True` |
| `numerical_robustness_reported` | `eq True` |
| `all_registered_runs_completed` | `eq True` |

## 必须交付

- `reports/G10/REPORT.md`
- `reports/G10/commands.log`
- `reports/G10/results.csv`
- `reports/G10/protocol.json`
- `reports/G10/research_conclusion.json`

另外生成 `reports/G10/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G10`；通过后才更新STATUS为PASSED。


---

<!-- source: docs/EVIDENCE_SCHEMA.md -->

# evidence.json 契约

文件路径：reports/Gxx/evidence.json。由真实评测程序写入，不手工编造结果。
`contract_sha256`是configs/acceptance.json的SHA256。改协议将使旧证据失效，须重新审计。

```json
{
  "schema_version": 1,
  "goal_id": "G04",
  "task_id": "jump_recover_ground_v1",
  "contract_sha256": "实际验收配置文件SHA256",
  "evidence_kind": "observed",
  "run": {
    "command": "实际执行命令",
    "git_commit": "真实commit；dirty差异另存",
    "started_at": "实际ISO时间",
    "finished_at": "实际ISO时间",
    "gpu_hours": 0.0,
    "env_steps": 0
  },
  "metrics": {},
  "artifacts": [
    {"path": "reports/G04/REPORT.md", "sha256": "真实文件SHA256"}
  ]
}
```
示例故意没有成功指标和有效哈希，不能用于通过验收。所有required_artifacts都要列出。
每个artifact必须在工作区内，非空，哈希一致。raw CSV、配置和checkpoint还要在报告里
建立对应关系，指标必须能由CSV重算。工具不是远程真实性证明；SHA只证明文件未变。

验证命令：
```bash
python tools/verify_goal.py --root . --goal G04
```
只做结构/预算之外的门槛核验：预算实际限制由训练runner实施；不能只靠此工具控制GPU。


---

<!-- source: docs/SOURCES.md -->

# 来源与版本核对

查询日期：2026-10-21。参考仓库不是本任务包的已运行依赖。

## S1 OpenAI：Using Goals in Codex
`https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex`

Goal生命周期与证据化完成条件。

## S2 OpenAI：Follow a goal
`https://developers.openai.com/codex/use-cases/follow-goals/`

/goal及启用features.goals。

## S3 OpenAI：AGENTS.md
`https://developers.openai.com/codex/guides/agents-md`

持久项目指导与加载边界。

## S4 Isaac Lab v2.3.2 Direct RL environment
`https://isaac-sim.github.io/IsaacLab/v2.3.2/source/tutorials/03_envs/create_direct_rl_env.html`

锁版本API参考；不是所有版本通用。

## S5 Isaac Lab v2.3.2 Multi-GPU
`https://isaac-sim.github.io/IsaacLab/v2.3.2/source/features/multi_gpu.html`

旧版分布式启动参考。

## S6 Isaac Lab current Multi-GPU
`https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html`

新版命令发生变化，必须按锁定版本使用。

## S7 LeggedManip_Lab（社区）
`https://github.com/zzzJie-Robot/LeggedManip_Lab`

Go2–PiPER参考；当前README指出与v2.3.2原样配置不兼容。未在此运行。

## S8 AgileX 官方URDF
`https://github.com/agilexrobotics/agx_arm_urdf`

标准piper目录与H/L/X分开。

## S9 AgileX 官方仿真资源
`https://github.com/agilexrobotics/agx_arm_sim`

核对PiPER模型资源。

## S10 Unitree 官方资产与训练
`https://github.com/unitreerobotics/unitree_rl_lab`

Go2基础配置参考，实际兼容性由G00验证。


---

<!-- source: docs/PACK_VALIDATION.md -->

# 任务包本地验证记录

日期：2026-10-21。

- Python标准库单元测试：18项通过。
- 缺少G00实测证据时，verify_goal.py返回FAILED与退出码2：符合预期。
- 11个Goal文档、依赖引用、JSON配置与初始TODO状态：一致性检查通过。
- 未运行Isaac Sim、Isaac Lab、GPU物理仿真或机器人PPO训练。
- 未获取/封装外部机器人物理资产，未生成任何机器人策略权重。
- 校验器只能检查结构/哈希/依赖/门槛；不是独立的仿真真实性或预算执行证明。

## 单元测试原始结果

```text
test_artifact_tamper_fails (test_goal_gate.EvidenceCheckerTests.test_artifact_tamper_fails) ... ok
test_bool_not_accepted_as_rate (test_goal_gate.EvidenceCheckerTests.test_bool_not_accepted_as_rate) ... ok
test_contract_tamper_fails (test_goal_gate.EvidenceCheckerTests.test_contract_tamper_fails) ... ok
test_dependency_cycle_rejected (test_goal_gate.EvidenceCheckerTests.test_dependency_cycle_rejected) ... ok
test_empty_artifact_fails (test_goal_gate.EvidenceCheckerTests.test_empty_artifact_fails) ... ok
test_invalid_time_fails (test_goal_gate.EvidenceCheckerTests.test_invalid_time_fails) ... ok
test_low_metric_fails (test_goal_gate.EvidenceCheckerTests.test_low_metric_fails) ... ok
test_missing_dependency_fails (test_goal_gate.EvidenceCheckerTests.test_missing_dependency_fails) ... ok
test_missing_metric_fails (test_goal_gate.EvidenceCheckerTests.test_missing_metric_fails) ... ok
test_missing_report_fails (test_goal_gate.EvidenceCheckerTests.test_missing_report_fails) ... ok
test_missing_required_artifact_fails (test_goal_gate.EvidenceCheckerTests.test_missing_required_artifact_fails) ... ok
test_mock_kind_fails (test_goal_gate.EvidenceCheckerTests.test_mock_kind_fails) ... ok
test_nan_metric_fails (test_goal_gate.EvidenceCheckerTests.test_nan_metric_fails) ... ok
test_numeric_not_accepted_as_bool (test_goal_gate.EvidenceCheckerTests.test_numeric_not_accepted_as_bool) ... ok
test_path_escape_fails (test_goal_gate.EvidenceCheckerTests.test_path_escape_fails) ... ok
test_unknown_goal_rejected (test_goal_gate.EvidenceCheckerTests.test_unknown_goal_rejected) ... ok
test_valid_synthetic_fixture (test_goal_gate.EvidenceCheckerTests.test_valid_synthetic_fixture) ... ok
test_wrong_task_fails (test_goal_gate.EvidenceCheckerTests.test_wrong_task_fails) ... ok

----------------------------------------------------------------------
Ran 18 tests in 0.030s

OK
```
