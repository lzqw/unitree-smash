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
