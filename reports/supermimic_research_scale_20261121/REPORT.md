# 双 Head 在线训练与采集吞吐

2026-11-21，RTX 5080，128 个原生环境。研究组 G1 已加载组合 checkpoint，启动时的固定面板 WM 复核通过。首块已经把 128 个控制 tick、16,384 条新 transition 写入 Replay。本报告发布时尚无该组 128 或 256 次更新的固定评价成绩。

## 训练和评价节奏

每个 Head 块以当前策略采集 128 tick，使用共享 Replay 更新 WM，再以冻结的块内 WM 更新对应 Head。256 次保留更新由 16 组 R8、A8 组成。在第 128、256 次更新处分别安排一次组合策略完整采集，并独立评价对应 checkpoint。固定评价复用已核对初始策略身份的基线，使用相同的 128 个案例、seed 和 800 tick。训练侧 RTG 仅来自从起点到终止都由同一冻结策略执行的回合。

接触监督拟合与策略侧接触梯度分开记录。128 次离线接触拟合得到的候选在签名开发数据上未过门槛。G1 的受限解析接触辅助属于研究配置。模型身份、训练参数、开发集拟合记录和实际加载的 WM 复核结果随本报告保存。

## 冻结采集和固定评价

历史冻结采集用 1,584 个控制 tick 凑齐 128 条完整回合，两个产物时间戳相隔 4,839.8 秒，包含仿真、整理和序列化。固定评价 800 tick 用时 1,788.1 秒。按各两次折算，单组历史预算为 3 小时 41 分钟，不含常规训练块和 WM、Critic 拟合。[历史计时](../supermimic_r16_and_contact_fit_20261121/runtime_cost_snapshot.json)

- 每次归档允许一个不可变 CPU 数据块在后台序列化、压缩、落盘并生成校验和。队列有界，保存 checkpoint 前等待写入完成，写盘异常会阻断保存。冻结回合收集器直接读取本 tick 已编码的 Replay 数据。
- 完整评价将门控计数保存在 GPU，结束后取回。加载策略可按需映射 checkpoint，文件哈希分块计算。原生子步复用固定拍面几何，用已选 lane 计数，并在 GPU 上按掩码更新首次接触记录。
- 两块已有的真实 CPU 归档各含 32 tick、128 lane。交替顺序重复两次，同步写入占用前台时间中位数为 1.451 秒，异步提交为 0.211 秒。立即等待最终落盘时分别为 1.451 和 1.487 秒。四次配对输出的压缩字节哈希均相等。该结果测量的是可与下一段仿真重叠的工作。[写盘基准](archive_write_benchmark.json)
- 在 RTX 5080 上，合成 128 lane 的局部子步运算重复 480 次，中位时间由 0.119 秒降至 0.038 秒，输出值、NaN 位置及计数一致。测试期间训练进程也在使用 GPU。端到端冻结回合和固定评价的提速须以新进程的实测为准。[GPU 局部检查](substep_kernel_check.json)

G1 启动于上述性能补丁之前。后续独立评价和新训练进程将加载补丁。新冻结采集清单记录仿真墙钟、归档排空前总时间、后台写入时间、前台等待和 env-step/s。后台写入与仿真可以重叠，两项用时不可相加。

清理移除了旧 canary 两个已完成刷新的回退快照，以及旧零更新失败运行的三个临时 checkpoint，共释放 7.21 GiB。当前组合 checkpoint 引用的历史 Replay、固定评价案例和已测策略均保留。[清理清单](storage_cleanup.jsonl)

## G1 首次实际策略更新

首个 Recovery 块采集了 16,384 条新 transition，执行 8 个共享 WM minibatch。4 个事件监督 minibatch 均实际更新了接触残差。批次中的 49、50、49、50 是有放回采样窗口内的事件数。首批交互与历史 Replay 共同用于训练。固定窗口 64 步球速 RMSE 从加载时的 1.001 降到 0.983 m/s，根角速度 MAE 从 0.170 降到 0.169 rad/s。

第一次提案覆盖 6 个起点记录的 16 条有效 lane，步长回退至 0.5 后保留了 Recovery 的实际 Adam 更新。目标从 4.112381 降到 4.112313，6 个 Recovery 参数张量均发生改变，Actor 阶段的 WM 哈希保持一致。采集用时 191.8 秒，8 次 WM 更新合计 197.9 秒，Actor 提案和候选复算 487.8 秒，Critic 更新 9.7 秒。采集和 WM 拟合仅在块首执行。评价对应学习收益的固定面板仍在后续里程碑。[首步记录](first_retained_recovery.json)

TensorBoard 中 G1_learning 与后续 G0_learning 使用本组新增交互量、保留更新和训练墙钟三个横轴。起点 checkpoint 已有 17 次保留更新与 4,990 个原生 tick，均从本组计数中扣除。训练墙钟从初始 Replay 重建完成后开始，整个进程耗时另由 launcher 记录。

针对 Actor 提案耗时，后续进程将完全截断的球分支直接与 Body 输出重新组合，避免反向遍历零梯度的碰撞积分器。G1 显式出球辅助保留对输入的导数。在 2 个记录时刻、每个 3 条 lane 的实际组合 WM 上，H4 的 CPU 检查覆盖辅助开启和关闭，两种实现的末态与动作梯度最大绝对差均为零。这项检查未测量完整 GPU H64 提速。[实测对齐](actor_graph_parity.json)

在同一张 RTX 5080 上复核上述 H4 对齐，覆盖两组记录起点和两种梯度设置。末态和动作梯度均与修改前逐元素相等。关闭显式出球梯度时，两个起点的同步反向时间分别从 0.600、0.584 秒降到 0.165、0.176 秒。与主训练共卡测量，各计时只对应这一次短轨迹。主训练原进程继续使用其启动版本。[GPU 梯度与计时](actor_graph_gpu_parity.json)

后续冻结采集与固定评价共用的球子步审计现按近拍 lane 计算上一帧间隙，并在 GPU 上直接更新接触时间、扫掠漏检计数和最近距离。128 lane 合成传感器输入覆盖全远、部分近拍、全近拍及待确认事件。18 帧逐帧核对全部状态张量后，两版均记录 328 次力确认接触、11 次扫掠漏检和 459 次未收敛。该测试特意降低迭代上限以覆盖未收敛分支。96 帧局部计时中位数在 RTX 5080 上由 0.484 秒降至 0.210 秒。正式评价沿用登记的物理时间步与扫掠迭代上限，完整 128×800 耗时仍待里程碑评价测量。[子步审计对照](ball_frame_audit_comparison.json)

核对 G1 原生 Replay 清单与零更新评价的任务合同后，两者使用相同的 TRAIN128 案例文件（SHA256 前缀 `2f9bf380`），原生环境种子均为 882072。G1 原始登记的 `train_case_seed=20261114` 对应学习器初始化；续训在环境 reset 后恢复组合 checkpoint 的 CPU/CUDA 随机状态。原登记保留供追溯。固定评价运行产生单独的轨迹，128 案例结果衡量训练分布内的配对变化，阶段末独立案例才衡量泛化。[种子与案例核对](case_seed_audit.json)

首个 Recovery 块的四个在线数据块已逐一校验 SHA256 并解压核对。原生 tick 4990–5117 连续，16,384 条 transition 全部标为 stream 8、mixed_v17、Recovery。原进程的归档顶层继承了初始冻结采集的策略名、Actor 哈希和旧提交，实际采集行为由当前训练登记和各行的策略版本标识。新进程把源归档身份单独保存，每行记录行为 Actor 权重哈希，顶层保存采集进程的启动提交和脚本哈希。成功轨迹索引与单策略价值校准按这一规则识别新旧在线归档。[首块逐行核验](first_block_replay_identity.json)

1024／2048 阶段末的独立评价已提前固定 512 个新球路。原协议生成器的 A、B 分布各取 256 个，种子为 97124701、97124702；全部球路的初始位置与速度组合都与 TRAIN128、正式 A200 和 B200 分离。单 GPU 按四批 128 环境、每批 800 tick 评价同一 checkpoint；按全部合格触球数汇总条件恢复率。运行入口为 `scripts/evaluate_supermimic_independent512.py --checkpoint <checkpoint> --g07-execution-root <root> --output <directory>`。面板在 G1 首个 Recovery 块期间登记，目前尚未执行该评价。[固定案例](independent512.json.gz)

在首个 R8 完成前，为 G1/G0 预先固定了开发评价专用的 128 条球路。训练和冻结数据采集继续使用 TRAIN128。研究评价使用同一生成器、相同 A/B 各半的构成及相同的 800 tick 评分，A/B 球路种子改为 97124801、97124802。G1/G0 在 128、256 更新的 checkpoint 上使用这一面板，共用初始组合的一次零更新原生评价，由监督程序在主训练后串行执行。历史 TRAIN128 零更新成绩单独保留，两套球路不跨面板做差。新面板与 TRAIN128、A200、B200 及阶段末 512 条球路均无重合。基线额外消耗一次 128×800 评价，两个策略组共用。[开发评价案例](fixed128.json.gz)

G1 的 R6 断点在旧配对入口会因当前 Actor 已更新而被拒绝。现在续接同一计划时从原策略快照或经过原评价哈希校验的 checkpoint 恢复比较用的 Actor。新的 checkpoint 同时保存零更新比较策略和最近一次冻结比较的记录与策略。在 CPU 上读取实际 R6 断点，两种旧格式恢复路径均核对了 128 条基线记录，恢复期间 R6 Actor 哈希不变。验证范围为比较状态的恢复，主训练继续运行。[实测结果](paired_resume_validation.json)

首个 Recovery 块 8 次提案全部保留，每次使用 16 个有效起点。最初 128 个真实控制步新增 16,384 条 transition。逐数据块检查原生合格触球和拍框的一次性事件锁存，二者均无新增边沿。WM 首块的接触监督来自此前的 Replay。对固定历史恢复状态重放 R0 与 R8 的 G07 执行器，门控开启的 748 条记录全部改变最终关节目标，变化 RMS 0.00501 rad，最大 0.01719 rad；门控关闭的 148 条记录变化为零。仅 Recovery 的六个权重和偏置张量发生变化。新策略真实得分须等待固定评价。[训练与采集](first_recovery_block.json) · [逐起点组的命令变化](recovery_r8_commands.json)

后续研究进程在 Approach 逐控制步预检前，先筛掉留出 lane、掩码外 lane 和尚未发球的状态。对 R8 的实际 Actor、WM 和 128-lane 原始记录做 CPU 对照，五个起点时刻的完整布尔掩码逐项相同。四个发球后时刻分别得到 44、39、41、29 条有效起点，压缩预检只推进各时刻的 45 个训练 lane。四次预检合计从 29.31 秒降至 21.63 秒。当前 G1 继续使用启动时的代码。[逐时刻计时与掩码](r8_approach_preview_parity.json)

对起点库里的记录，预检也可以先取四个候选类别的 lane 并集。回退到普通起点时另用完整训练掩码。R8 的四类起点各取第一条可用记录，去重后为三条；CPU 重放比较了每条记录的完整类别并集，所有合格 lane 一致。三条记录的预检合计从 28.78 秒降至 12.17 秒。这是起点选择的局部计时。[起点库对照](r8_bank_preview_parity.json)

首个 Approach 更新已全步长保留。16 个有效起点全部匹配参考并产生 Tracking 梯度。对冻结历史起点重放 R8 和 R8+A1 的执行命令，Approach 门控大于 0.25 的 152 条记录全部改变最终目标，变化 RMS 为 0.000338 rad，最大 0.001305 rad。门控为零的 252 条记录均保持原目标。仅 Approach 的六个参数张量发生变化。前两个块累计新增 32,768 条原生 transition；策略成绩将在固定面板上另行测量。[首个 Approach 更新](first_approach_update.json) · [执行命令审计](first_approach_commands.json)

冻结采集器现在直接接管控制步开始时已独立保存的状态与执行器记忆，步末仍独立保存终态、最终命令和子步事件。用 G1 第七个真实数据块的 128 lane、32 tick 做 CPU 重放，32 行序列化字节逐行一致，四次交错顺序测得步末封包合计时间中位数从 0.166 秒降至 0.085 秒。这部分节省只占完整采集的一小部分。SuperMimic 回归集 671 项通过。[数据身份与封包计时](transition_seal_parity.json)

首个 Recovery 块从 R1 结束到 R8 结束共 2,853 秒，其中 2,430 秒用于七次 Actor 提案的 BPTT 与复算。同期 568 个资源采样的整卡利用率中位数为 23%，显存占用中位数 5,138 MiB、采样最大值 5,557 MiB，训练进程 CPU 使用量中位数为 1.03 核。这段时间的负载有明显显存余量。每步的 16 条有效起点按原记录分别执行 H64，后续优先优化小批次的执行效率。第一个 Approach 块已保留两次全步长更新，均使用 16 条有效 lane。[原生训练资源记录](r2_r8_resources.json)

为后续独立训练重复增加了 `run_supermimic_research_scale.py --training-seed`。显式指定时，程序从相同的组合权重与历史 Replay 出发，在原生环境 reset 后重新初始化学习器、Replay 窗口和长飞行窗口的随机流。G0 继承配对 G1 的训练种子。正常断点续接沿用各随机流保存的位置。使用本次 G1 的实际初始化 checkpoint 做 CPU 重放，新的训练种子改变了三个随机流的抽样顺序，重复指定同一种子可复现三组抽样。672 项回归测试通过。当前 G1 沿用原登记的种子，第二个训练 seed 的策略实验留到扩展阶段。

新增累计扩展入口 `PYTHONPATH=.:scripts python scripts/extend_supermimic_research_scale.py --parent <已完成的组目录> --total-retained 1024 --output <新目录> --tensorboard <原TensorBoard根目录>`，后续终点可取 2048。入口使用上阶段冻结采集、WM 和 Critic 刷新全部结束后的 latest.pt。调度器核对原计划与回执，直接进入下一周期的 Recovery，沿用真实交互计数、学习器和采样器状态，完成扩展后执行独立 512 案例评分。新进程从新的原生回合开始；已有 Replay 通过归档恢复。新 checkpoint 保存训练墙钟，首个旧版 G1 的续接计时使用原 launcher 的进程耗时，包含其启动与 Replay 重建。

在实际 R8+A4 checkpoint 上做了只读 CPU 检查。两个 Head 各有 6 个 Adam 参数状态，WM 有 64 个、Critic 有 14 个，Replay 随机状态与四份归档清单已保存。恢复出的调度指向首个 Approach 块的第 4 次保留更新。CPU 回归覆盖 256→1024→2048 的累计预算、保留的周期及冻结回执、使用刷新前 checkpoint 的拒绝路径，以及接触事件跨阶段累计。675 项通过。主训练仍在 256 阶段，尚无原生扩展成绩。[实际断点检查](cumulative_resume_inspection.json)

在 R8 权重和两个真实原生记录上检查了 G07 执行器状态的分支复制。对只含浮点数的计划观测列表保留独立副本，避免为每个历史标量递归分派。每个记录取相同三条 lane，四组交错计时的复制中位数分别从 38.52 降至 5.41 ms、从 42.59 降至 7.64 ms。H4 时逐步最终关节目标、55 维物理状态和执行命令梯度均逐元素相等。676 项 CPU 回归通过。该局部优化供后续进程使用，完整 G1 墙钟按实际运行版本统计。[分支复制基准](g07_branch_clone_parity.json)

在 G1 最初保存的 256 个真实控制 tick（128 个环境）逐一读取越网检查输入。40,960 个物理子步里，每步所有 lane 都能由“未在回合中或球拍仍完全在己方”排除。冻结采集与固定评价的原生环境现在直接跳过这些空查询，有候选时仍执行原来的整批几何和见证判定。RTX 5080 上，每 128 次合成空查询的中位数从 0.0460 秒降至 0.00175 秒；有 4、64、128 个合格 lane 时，四组交错计时的中位数增加 3%–6%，返回 lane 序列全部一致。覆盖比赛后半程的整体收益将由原生里程碑计时给出。682 项 SuperMimic 及越网几何测试通过。[真实子步分布与 GPU 局部计时](net_broadphase_parity.json)

已核对磁盘清理日志及仍在用的归档路径。清理过期 checkpoint 共释放 7.21 GiB，内容哈希相同的已关闭 Replay 数据块去重共释放 43.69 GiB，原读取路径仍可用。18:54 所在文件系统可用约 70 GiB。[空间清单](storage_reclaimed.json)

19:16 首个 R8→A8 完成，16 个提案全部保留。Recovery 用时 63.9 分钟，Approach 用时 69.6 分钟，合计 133.4 分钟。两块的 Actor BPTT 和提案复算合计 95.6 分钟（71.6%）。每次 Approach 使用 16 条有效 lane，第 5、8 次提案以 0.25 步长保留，其余六次保留全步长。将 R8 与 R8+A8 的策略在同一组记录状态上重放，强门控的 152 条记录全部改变实际关节目标，RMS 为 0.00567 rad、最大 0.01240 rad。252 条零门控记录的变化为零，仅 Approach 的六个参数张量改变。任务成功率将在固定面板上测量。[八次提案](first_approach_block.json) · [最终执行命令](first_approach_block_commands.json)

按首个完整循环外推，G1 剩余 15 个常规循环约 33.36 小时。另有两次冻结采集，历史合计约 2.69 小时，以及两次固定评价和两组共用的一次零更新基线，历史合计约 1.49 小时。G1 单组剩余工时的点估计为 37.54 小时。冻结后的额外拟合、后续接触辅助的反传和回退次数都会改变实际耗时。G0 对照组单独计时。[首个循环墙钟与预算](first_pair_budget.json)

A8 后第二个 Recovery 块继续真实交互。重新读取归档块 8–11 并核对哈希，原生 tick 5246–5373 连续，共 16,384 条 transition，均标为同一组合策略版本 33。65 个 lane 出现合格拍线触球，44 个出现拍框触球，合计 109 个独立 lane-episode。109 个均进入碰后分离门控，合计 12,085 个有效 lane-tick，其中 8,592 个有足端支撑。无足端支撑转为有支撑共 137 次，包含重复支撑转换。[新数据明细](post_a8_native_block.json)

首两个块的 16 个 WM 短窗批次中，记录的最晚窗口起点和 16 步终点均早于本轮起始 tick 4990。其 recent_online 配额由历史在线数据满足。后续进程按实际抽中的每个 transition 保存原始采集策略版本与拍线／拍框事件使用次数，保留在窗口中途发生的策略切换，也能识别被抽为 contact_event 的在线数据。TensorBoard 按块显示采集策略对共享 WM 和专用接触拟合的使用次数。旧日志保留原有可核验范围。684 项 CPU 回归通过。

19:34 已保留 R9，累计 17/256。第二个 Recovery 块的八个 16 步 WM 批次均实际抽到了 A8 后策略版本 33 的 transition。每批最晚窗口起点为 tick 5296，另有历史锚点。四次接触拟合各使用了 38、37、42、34 个采样事件，计数涵盖各来源且允许窗口重叠。三次固定 WM 面板上，H64 Body RMSE 为 0.5521、0.5515、0.5520，球速 RMSE 为 0.9826、0.9721、1.0066 m/s。接触当步的一步球速 RMSE 为 0.3188、0.3142、0.3126 m/s。R9 使用 16 条有效 lane，全步长保留。另用新采的含触拍数据检查越网几何预筛，20,480 个实际物理子步都可跳过空查询。完整 128 案例固定评价成绩仍待里程碑。[R9 拟合与前向复核](post_a8_wm_sampling.json)

固定案例评价现在将合格触球、球侧完成、Complete5 的败转胜和胜转败数、双方共同合格触球案例的恢复翻转与分母，以及真实 Return 配对差的均值、中位数和描述性 95% bootstrap 区间写入 TensorBoard。每个点复用已评分的同一批案例，按保留更新、真实交互量和训练墙钟记录。使用历史 R16 后的固定 128 案例档案回放，三个二元指标的逐案例名单和精确配对 p 值、Return 均值及区间与独立保存的分析逐项一致。687 项 CPU 测试通过。目前 G1 的新面板仍待 128/256 里程碑评分。首块 Approach 两次四分之一步长来自较大步长未保住原本的 TOI 覆盖，两次的命令预算均合格。

检查首个 Approach 短块的起点请求，5 次 gate_onset、3 次 pre_contact；累计 128 次 lane 起点使用中有 57 个不同的 (lane, episode, tick) 标识。旧版用块内提案编号索引长度 20 的类别表，若每块八次即完成，会反复使用前八项。在本次 checkpoint 中，failure_pre 与 success_near 分别已有 80 和 48 个 lane 快照标签。这些数是标签支持数，尚需通过实际预接触可用性筛选。新启动的 G1 和匹配 G0 按已保存的 Approach 提案数连续轮换，拒绝、跨块和中途恢复都保留位置。现有进程及其匹配对照沿用原登记的顺序；双方进入累计扩展后使用连续顺序。688 项 CPU 测试通过。该修正的真实任务收益待实测。

对实际起点库的执行命令预算检查，块开始时可以从每条记录自己的控制器记忆预先求一次 G07 基础命令和限速器状态。候选 Actor 的残差及最终关节目标仍逐次重算。在 tick 272 和 384 的两条 128-lane 记录、两套真实 Actor 权重上，RTX 5080 预热后每轮预算预览的中位数从约 181 ms 降到 3.4–3.7 ms，首次准备两条记录耗时 194 ms。预热后请求、归一化命令、关节目标逐元素相同，冷启动比较曾观察到 7.45e−9 差异。不同记录的控制器时钟仍各自推进。该局部计时没有覆盖 H64 反向或整步训练。689 项 CPU 回归通过。[命令对齐与局部计时](fixed_command_preview_parity.json)

将首轮起始策略与第 19 次保留更新（R11+A8）的组合策略在相同归档状态、执行记忆和关节限幅下重放。准备击球窗口中，仅 Approach 门控激活的 221 条有 166 条改变最终关节目标，RMS 0.00418 rad。恢复窗口中，仅 Recovery 门控激活的 668 条全部改变最终目标，RMS 0.00872 rad、最大 0.02553 rad。两类窗口中双门控关闭的 373 条记录均无变化。参数变化限于两支 Head 的 12 个张量。固定评价程序将在两个正式研究里程碑按同一方式重放执行命令，同时写入 RMS、改变记录数及截断前高斯 KL；三个横轴沿用对应策略在采集和学习时的计数。第 19 步的记录是 CPU 执行重放，正式 PhysX 面板成绩仍待评价。690 项回归测试通过。[逐组执行命令](diagnostic_commands_0019.json)

固定 128 案例的里程碑评价现在按训练 checkpoint 已完成的冻结事件读取采集凭证，同时记录 Actor 保存时与全阶段结束时的真实训练交互量。收齐各 lane 完整回合期间实际推进的所有 lane-tick 计入冻结开销，进入同策略 RTG 的步数单列。候选的独立评价及两组共用的一次零更新评价另计。用历史 1,584 tick、128 lane 的冻结归档核对，实际推进 202,752 个 env-step，其中 100,192 个属于选定完整回合。当前 G1 的 R12＋A8 checkpoint 已产生 49,152 个常规在线 env-step，冻结采集为 0。692 项回归测试通过。

用实际 G1 的 R8＋A8 权重、tick 272 的两条匹配起点回放完整 H64，击球到达损失只在索引 6、7 产生监督。新代码对其余区间保留标量值并省去该零损失的反向图，回退时一次检查整段参考覆盖。按旧、新、新、旧顺序测量回放与 Tracking／anchor 参数梯度，两版在 CPU 和 RTX 5080 上分别逐项对齐，记录的最大绝对差为 0。单线程 CPU 中位数 10.49→7.23 秒，与当前训练共享 GPU 时为 18.94→13.61 秒。该计时只涵盖这两条起点，未来新进程采用修正，已启动的 G1 按原代码运行。693 项回归测试通过。[实际 H64 的数值与计时](sparse_toi_backward.json)

在 R8＋A8 策略新采的 128 tick 中，从本轮 WM 排除的 lane 选取触球分离后首次由无足支撑转为有支撑的记录。68 个不同 lane／回合均有连续 64 步，其中 38 个在着地时已有合格触球。以着地后的实测状态启动，使用逐步记录的实际关节目标，比较初始化 WM、第三块拟合前以及拟合后的完整 WM。三者的 H64 姿态平均误差为 0.1485、0.1506、0.1504 rad，机身角速度 MAE 为 0.2316、0.2307、0.2315 rad/s。拟合后 H64 高度 MAE 为 10.9 mm，足支撑位错误率为 10.7%，姿态误差 P90 为 0.3052 rad。直接保持着地状态的 H64 姿态误差为 0.6254 rad。此次刷新在该组动作上未出现早期实验的长时漂移。该检查衡量自然执行命令下的前向误差，动作扰动方向和原生恢复率另行评价。TensorBoard 的 G1_postlanding_WM 已绑定首次实际使用该版 WM 的提案，是累计第 17 次保留更新、49,152 个新 env-step。695 项回归测试通过。[逐时域、模型身份与窗口](post_a8_landing_wm.json) · [复核程序](../../scripts/audit_supermimic_online_landing.py)

在首组 R8＋A8 冻结策略上提前回放预热后的 G1 配置。tick 272 有 30 条 TOI 合格训练 lane，按 lane 顺序等距取 12 条，分三批执行 H64。12 条均在有效门控内出现一次可用的模型预测拍击。按登记的 0.1 出球权重计算，三批的出球／TOI 参数梯度范数比分别为 0.63、1.21、0.024，方向余弦为 −0.337、0.469、0.525。该结果说明辅助目标可以产生有效参数梯度，其贡献随案例而变化。这次 CPU 回放没有更新 Actor，也没有测量 PhysX 出球收益。新启动的训练进程记录实际参与辅助目标的 lane 数，并将加权范数及比例写入 TensorBoard。旧日志缺少覆盖数时保留缺失。696 项回归测试通过。[分组与模型身份](prospective_outgoing_shadow.json)

20:36 第二个 Recovery 块完成，累计 R16＋A8。8 次更新均全步长保留，128 次起点使用涉及 110 个不同的 lane／回合／tick、32 个不同的 lane／回合，Actor 提案与复算合计 63.05 分钟。已从新归档确认随后 Approach 块的首 32 tick 由更新后的版本 41 采集。此次切换在原登记的 Critic 单批 MSE 门槛下额外进行了 4 轮 Critic-only 想象拟合，MSE 依次为 135.23、118.79、98.89、40.15，计算合计 94.0 秒，另有 checkpoint 写盘。以后新登记的研究组及累计扩展在满足 Critic 预热并得到有限 loss 后，记录该 MSE，继续常规 GAE 和冻结回合 RTG 校准。当前 256 次的两组沿用彼此匹配的原登记，正式 profile 保留原门槛。697 项回归测试通过。[Recovery 块、Critic 重试和新采集身份](second_recovery_block.json)

### Replay sampling cost on the registered native archive

The episode-capped WM sampler previously ran an uncapped flow and discarded its solution, constructed eight unused event pools, and sorted (stream, lane, episode) triples across the replay. The capped path now runs its integer allocation directly and encodes the same episode partition with scalar keys. On the 800-tick, 128-lane initialization archive, the median CPU draw fell from 0.426 to 0.218 seconds for Recovery and from 0.346 to 0.145 seconds for contact windows. Across four seeds for each distribution, the sampled tensors, lane order, sampling reports and final RNG state matched byte for byte. The current G1 process uses its startup code. These timings measure CPU sampling alone. [Benchmark](replay_quota_cpu_parity.json)

### Coverage across consecutive quota-constrained WM batches

On the real initialization archive, eight consecutive draws used the same 16 Recovery lanes and the same 17 contact lanes within their respective distributions. The integer allocator repeatedly returned the same optimal episode allocation while window selection varied within those episodes. Using the checkpointed replay RNG to shuffle the allocation groups before solving increased coverage to 88 Recovery lanes and 82 contact lanes across eight batches. All draws met the source and event quotas and episode cap. Sampling time was similar. This measures training-data coverage, with no policy outcome measured. Both matched 256-update arms keep their registered sampling settings. The cumulative extension enables the shuffled order for both arms. [Coverage measurement](wm_quota_lane_coverage.json)

### R16 + A9 and native reward accounting

At retained update 25, 65,536 new native lane-transitions had been collected. The fourth block sampled 1,024 short WM windows. The latest window started at tick 5275, so even its 16-tick rollout ended before the new version-41 collection began at 5374. The eight WM optimizer steps used older replay, including the preceding version-33 collection. Four contact optimizer steps each used 59 measured event occurrences; their independence across draws is unverified. The composed WM passed the formal anchor panel. Relative to the initial deployed panel, some short body errors increased slightly and the post-contact one-step velocity RMSE increased from zero to 5.44e-8 m/s. The accepted Approach update used 16 effective lanes and the warmup TOI objective. [Update record](second_approach_first_update.json)

The weighted components from the first five completed online episodes reconstruct each recorded Return to within 1e-6. All five incurred a body-ground penalty of 40. Case A-029 also incurred 4 for a racket-frame event and 8.55 for post-deadline time without qualified contact. Case A-009 earned 20 for qualified contact and zero for crossing and target. The baseline summary lacks stepwise reward components, so a term-by-term baseline difference is unavailable. [Native reward components](first_five_reward_components.json)

### Pooled H64 tracking gradients and fixed-panel preparation

The latest completed Approach proposal at 26 retained updates used 16 eligible lanes from four records (1, 1, 1 and 13 lanes), with a separate G07 execution state per record. Its proposal computation took 301.6 seconds. The preceding Recovery block processed five to seven distinct records per proposal. In that path, the tracking diagnostic and the Actor derivative previously traversed the same imagined trajectory separately. Both derivatives now come from one autograd traversal per record. On a two-lane recorded Recovery start from the frozen R16+A8 checkpoint, a CUDA comparison took 22.37 seconds before and 18.66 seconds after. The measured loss, gradient vectors and Critic tensors matched. These are two-lane measurements on a shared GPU; the running G1 was not restarted. [Comparison](pooled_tracking_backward.json)

The registered initial checkpoint resolves the measured native source and task contract. The 128 training, 128 fixed development and 512 independent cases have disjoint case IDs. This preflight does not score a policy. The first paired independent development scores remain pending.

### Online return and Approach update cost through proposal 28

Across the first 12 Approach and 16 Recovery proposals, all were retained. Mean wall time per proposal including block collection was 528.8 and 539.5 seconds respectively. Median Actor proposal computation was 300.8 and 385.3 seconds. Between collections, the median wall times were 407.7 and 469.3 seconds. At R16+A8 versus R16+A10, only six Approach tensors changed. Across the 221 Approach-only archive states, 166 final executable commands changed with joint-target RMS 0.0017083 rad. All 668 Recovery-only states in the Recovery panel had identical commands. [Same-state command replay](within_approach_0024_0026.json)

TensorBoard now tracks cumulative return for completed first online episodes and the initial frozen policy on those exact TRAIN128 cases. It also displays the paired difference and net ball-side, physical recovery and Complete5 case counts. The first five mixed-policy episodes have mean difference −2.67. Case order follows termination time. The independent frozen-policy evaluation remains the policy comparison.

The learner reuses the fixed command references within a proposal and reduces budget deltas on-device. Two candidate checks on two recorded 128-lane G07 states gave identical budgets and final-target effects with the original and optimized routines. CPU preview time for this isolated comparison was 0.0268 versus 0.0210 seconds. G1 continues in the original process. [Command comparison](sm_budget_within_proposal.json)

### Shared free-flight prediction for Body and ball

The composed physical transition was integrating the same 160 native drag substeps for the body and ball priors. The shared calculation preserves the parked and launch transitions, keeps the robot derivative during ball-detached BPTT, and keeps independent backward traversals when both consumers differentiate the incoming state. On 16 recorded starts and commands, one GPU transition with the ball gradient detached took 80.99 ms before and 64.74 ms after. A forward-only step took 59.27 ms before and 52.89 ms after. Across compared outputs and input derivatives the largest absolute GPU difference was 1.87e-9. The active G1 process continues with the loaded implementation. [Conditions and CPU/GPU measurements](shared_flight_transition.json)

### First 128 resolved TRAIN128 flights

All 128 first flights have ended in the first 65,536 native lane-transitions. Ground contacts and rule faults both end a flight. All 109 observed racket releases occurred under the combined R8+A8 policy (global version 33). The approach to the shot crossed policy versions. Five full robot episodes had terminated at this snapshot.

The frozen initial actor hit the shuttle in 110/128 cases and the online actors in 109/128. Qualified airborne string contact fell from 74 to 65 and frame contact rose from 36 to 44. All 13 cases losing a string hit became frame hits. Four cases gained string hits. Among the 61 cases with an airborne string hit on both sides, 52 landed in the target on each side. Among the 13 losing the string hit, target landings fell from 11 to zero. The four gaining a string hit all reached the target. Target landings across the full panel fell from 63 to 56.

The qualified contact, scored crossing and scored target bonuses fell by 180, 50 and 135 points in total, or 2.85 points per case combined. The initial values use the unchanged native one-shot rules on the frozen terminal latches. The online totals agree exactly with the archived weighted reward components across all 128 cases. This excludes recovery, body contact and time-dependent rewards. Full mixed episode Return and independent frozen policy results remain separate. [All paired cases, flips and reward sums](first_128_online_flights.json)

In case A009, the measured outgoing speed fell from 5.69909 to 4.99928 m/s. The shuttle contacted the net, crossed its plane, and landed at y=1.02181 m outside the 1 m target half-width. The 5 m/s requirement suppressed the crossing bonus. The earlier reward summary alone could not distinguish a missed crossing from a crossing without its bonus. The new per-episode output retains the observed flight measurements and terminal behavior version separately from the head-local update count. [Five native event timelines](first_five_native_flight_events.json)

### R16+A16 execution and native outcomes

After 32 retained updates both head parameter sets changed. Replaying the CPU actor on all 16,384 observations collected in the following block reproduced the normalized native requests with maximum absolute error 1.2e-7. Four lanes launched again and none reached a new first racket contact in this block. The 32-update WM scored 1.047 m/s on the 64-step development velocity panel (formal anchor 1.200 m/s). [Checkpoint and native replay](milestone_0032_verified.json)

For the first 128 flights, all 13 switches from qualified strings to frame were frame contacts on the first native contact substep. Twelve landed on the same lateral side of the racket. The median signed distance inside the string polygon was −3.33 mm for these 13 versus +32.86 mm for the 61 retained string hits. The distance uses the measured world contact point and the native boundary poses interpolated to the observed substep. From native starts 40 ms before impact the 32-update WM predicted solid racket contact for all 13. From 80 ms it predicted solid contact for seven and strings for six. Both rollouts used the actual executed command sequence through the native contact tick. The shots occurred at R8+A8. [Case geometry](first_contact_frame_geometry.json) · [Forecast by lead time](first_contact_frame_leads.json)

Among the first 12 terminated episodes, mean Return on the matched initial cases was −7.13 and online Return was −25.38. Three additional body-ground cases cost 10.00 points per case averaged over all 12. Qualified contact, scored crossing and scored target together cost another 6.25. The other reward terms accounted for the remaining 2.00. Each measured online Return agrees with the stepwise archive. Baseline shot and body contributions use native terminal flags and one-shot weights. These early terminations span multiple behavior versions. [Episode-level accounting](first_12_reward_sources.json)

### Task-memory encoding throughput

The first 32 retained G1 updates took 274 minutes after initial Replay reconstruction. Actor proposals and candidate rollout checks accounted for 197 minutes. The last completed Approach block took 60 minutes, including 38 minutes in the proposal path. Using contiguous fields of the same dtype for validity masking reduces kernel dispatch in both the imagined task transition and Critic context. On the same 16 archived lanes, the CUDA encoder median fell from 2.27 to 0.23 ms for imagined memory and from 2.02 to 0.88 ms for native memory. Four CPU rollout steps reproduced the physical state, requested final targets, reward, done, Critic input and action derivative exactly. The running G1 process uses the previously loaded encoder. New processes receive this optimization. [Measurements and live block timing](task_memory_encoding.json)

### Step size and command budget through retained update 33

All 41 candidate comparisons satisfied the configured execution budgets. Of 16 Approach proposals, three used a reduced step to preserve TOI lane coverage and two used a reduced step after their full-step imagined objective increased. One of 17 Recovery proposals used a reduced step after its imagined objective increased. All 33 proposals retained a parameter change. The current run's `final_command_max_change` measures the executed-target preview against the pre-proposal policy. Its older `budget_report.relative_last_accepted` was computed after moving the accepted reference, which produced zero for accepted steps. New training processes retain the pre-proposal budget reference in the report and include measured budget deltas for each trial. [Trial metrics](first_33_backtracks.json)

### 身体触地前的 Recovery 起点与线床边缘修正

累计第 33–35 次保留更新中，在线窗口内可核对的 5 次即将身体触地的起点，全部只提前 1 个控制步。三个新增身体触地案例的罚分出现在首次落地后 0.67、2.10、2.72 s。新采样器用实际身体触地一次性罚分的上升沿挑选提前 20 个控制步的因果状态，保留当时的 G07 记忆、观测和执行时钟。回放 R16＋A16 后的 128 个原生控制步得到 9 个这样的 lane，3 个可用于在线训练，6 个属于原 WM 留出集。短块内起点不足时仍可用现有的多阶段起点。

Approach 在已有 TOI 运动学损失中加入线床边缘软惩罚。使用实际注册球拍资产的线床多边形，在预测到达时刻把来球投影到拍面坐标系，内侧余量达到 5 mm 后该项为零。对实测首次接触投影进行 GPU 复算，13 个由拍线转框的案例均有向内的下降方向，61 个持续拍线命中的案例该项梯度全为零。固定 R16＋A16 权重，在 4 个训练 lane 的真实初态上进行 H64 想象，新增损失对 Approach 参数的梯度范数为 1.17935。本次为离线计算。修正后策略的原生触球率、身体触地率和 Return 仍待冻结配对评价。

两项改动随新的研究主线配置注册，G0 复制对应 G1 的开关，原已注册的 256 次实验继续按其配置运行。累计扩展入口为两个 Head 对应的实验组同时启用修正。TensorBoard 记录 Recovery 实际抽中的提前起点数和原生 Return 滑动窗口的回合数。[实测样本、模型身份与梯度结果](recovery_prefix_and_string_margin.json)

### R20＋A16 冻结复评与训练切换

在相同的 128 个初始状态及原生执行合同下完成 800 tick。对照为此次研究登记的配对基线，seed 为 882072。相较基线，合格空中拍线 74→55，球侧成功 51→26，触球后恢复 35/74→18/55，Complete5 22→8，身体触地 60→61。25 个案例从合格拍线变为拍框先碰，8 个反向改善。19 例新增身体触地，18 例避免原有触地。平均 Return 从 −12.089 降到 −28.093，逐案例均差 −16.005，配对 bootstrap 95% 区间为 [−24.973, −7.371]。该结果评估了修改前的策略。

新训练从基线恢复两个 Head 及其 Adam，保留在 R20＋A16 时收集的真实 Replay 和已更新的 WM、Critic。共享 WM 在续训入口通过原生 forward 面板。新阶段重新计算 256 次保留更新预算，旧阶段 36 次另列。两个对照组使用同一份交接 checkpoint 和同样的修正。原生评分、学习率、动作预算及 H64 沿用原登记。

从 640 个已写盘的新增控制步重建 Recovery 起点，排除 83 个留出 lane，找到 8 个训练案例的 24 个身体触地前窗。在一次 16 起点抽样中，8 个来自失稳或触地前窗，4 个来自落地转换，4 个来自稳定维持，覆盖落地后 1.3–2.6 秒的风险状态。Approach 的 H64 冻结权重回放覆盖 1 个预测拍线首碰、1 个预测拍框首碰，TOI 和首次接触组合梯度范数为 4.247，这两条的出球辅助梯度范数为零。上述回放用于检查训练信号，策略收益需用修改后的 checkpoint 再做原生评价。[配对结果、起点与梯度证据](r20_a16_revision_evidence.json)

修正版 G1 的第一次 Recovery 参数更新已实际保留。128 环境新增 16,384 条真实转移，共享 WM 更新 8 次后通过固定原生前向复核。八个批次各分配 20 个落地后失支撑窗口和 16 个身体触地前窗，两类配额均满足。Actor 实际获得 16 个有效起点，风险、落地转换、稳定维持各 8、4、4 个。风险组包含落地后 2.46 秒仅一脚支撑的状态，以及距离身体触地 20 和 40 tick 的起点。六个 Recovery 参数张量均改变，执行器预览中最终关节目标最大改变 0.00183 rad，Actor 更新期间 WM 冻结。该提案采集耗时 173 秒、WM 更新 201 秒、H64 提案与回溯检查 336 秒。块内后续提案共用本次采集与 WM 刷新。修正后的策略仍待冻结原生配对评分。

### 修正版的固定 128 案例零更新基线与训练回报监控

G1 和 G0 共用的零更新 Actor 在独立固定面板上完成 128 个原生回合、800 个控制步，耗时 31.88 分钟。合格拍线触球 72/128、球侧完成 48/128、触球后恢复 26/72、Complete5 为 21/128、身体触地 76/128，平均 Return 为 −17.675，数值异常为零。该组案例独立于训练采集，后续 128、256 次更新均与它逐案例配对。训练侧固定案例使用另一套初态。三个学习曲线横轴的零点已写入 TensorBoard 的 shared_fixed128_zero。[零更新结果与原生记录哈希](revised_shared_fixed128_zero.json)

旧 G1 的在线 Return 从 −21.27 降至 −25.38。用已结束的首回合对齐其训练侧冻结案例，前 5 例均差为 −2.67，随后 7 例均差为 −29.38，后 7 例新增 3 次身体触地并净损失 2 次球侧完成。修正版独立监控每批新增完整回合。每批至少覆盖 4 个不同 lane，首回合连续两批都落后于相同案例基线，且第二批的配对差、批均值和滚动 Return 同时下降时，暂停实际训练进程并记录案例与指标。缺少同初态对照的后续回合，连续三批的批均值和滚动均值均下降时暂停排查。该判据用于发现退化，冻结配对评价用于判断策略收益。旧 G1 的日志回放在第 12 个回合触发了预期的暂停判据。

### 修正版首组策略交接与共享 WM 样本来源

首个 Recovery 块保留 8 次参数更新。在冻结记录状态上重放 G07，Recovery 激活的 748 条状态上最终关节目标 RMS 变化约 0.00665 rad；对应未激活的 148 条变化为零。随后 128 个原生环境使用 R8 行为策略各执行 128 个控制步。首个 Approach 提案已保留，两块共新增 32,768 条真实转移。第一批完整回合的 Return 仍待回合结束后计分。

逐批审计显示，首个 Recovery 与 Approach 块各进行了 8 次 WM 更新，每批均抽到 24 个近期在线窗口，但两次刷新中刚采集的行为版本分别贡献了 0 条训练转移。对 R8 的四个新原生数据块排除 83 个留出 lane 后，有 5,085 个普通窗口，只有 2 个 Approach 激活窗口，触拍事件为零。该段可补充普通动力学，接触监督继续来自先前真实事件。

在后续累计扩展入口中，每批以最新在线策略的 6 个完整窗口为目标。分配器先最大化已满足的来源与事件配额，再选择最接近该数量的解，同时维持 episode 上限。连续版本或流边界不混作最新窗口。使用两个真实采集块离线重放近期来源的 24 个槽位，8 批各抽到 6 个最新窗口，同时保留两个激活窗口和 22 个普通配额。该重放只覆盖近期来源。当前 G1/G0 的首轮 256 次预算沿用已登记的采样配置。[实际刷新、窗口覆盖与采样重放](revised_first_pair_sampling.json)

### R8＋A8 的真实首碰与出球速度

第 257–384 个控制步共 16,384 条原生转移，由同一 R8＋A8 策略采集。固定这些观测和因果门控，用该 Actor 重放动作请求，最大绝对误差为 5.22×10⁻⁸。相较 R8 权重，A8 改变了 6,763 条动作请求，Approach 门控激活时变化 RMS 为 0.00472（归一化请求单位）。该段检测到 111 个首次球拍接触。

到第 384 个控制步，128 个案例中已有 123 个首球结束。这 123 个相同训练案例与冻结基线配对后，合格拍线为 74→73，其中 3 个拍线转框先碰，2 个反向改善。计分过网 58→52，计分目标落点 52→45。8 个失去落点分的案例中，6 个仍然合法触球、正向过网并落在目标区，唯一失去的条件是出球速度达到 5 m/s。这 6 例的原生速度由 5.02–5.25 m/s 降到 4.84–4.96 m/s，均来自 A 类球路。另有 1 例转框、1 例落点越界，1 例新获得落点分。逐例原生计分与归档奖励一致。该阶段 G1 的出球辅助权重为零。完整回合的 Return 和恢复率仍待终局计分。[逐例速度、奖励与动作复核](revised_first_384_ticks.json)

紧随 A8 的 Recovery 更新已全步长保留，累计 R9＋A8。共享 WM 的 8 个训练批次实际使用了该 R8＋A8 行为版本的真实转移 3,705 次，包含 10 次球拍事件边沿使用。其中 4 次进入了执行接触残差优化的批次。计数允许窗口重复，独立采集量和独立触拍案例另报。8 批的 fallback 都是零。相同原生面板上，64 步球速 RMSE 为 0.9849→0.9837 m/s，根角速度 MAE 为 0.1681→0.1672 rad/s。Recovery 的 16 个有效起点含 8 个风险、4 个落地和 4 个稳定窗口。完整回合结束数为零。[逐 WM 更新的数据回流](revised_r09_post_a8_fit.json)

用 A8 实际使用的共享 WM（第 152 次更新）从原生首碰前一帧开始，按每个案例的真实执行命令连续预测三个控制步。111 个原生首碰与释放案例中，模型在正确控制步预测了 110 个球拍接触，漏检 1 个。73 个拍线首碰全部对齐到控制步，其中 20 个实际出球速度低于 5 m/s，模型却将其中 9 个预测到门槛之上。6 个仅因速度不足失去落点计分的案例中，WM 识别了 3 个。按训练中已有的 5.25 m/s 速度裕量，5 个会收到速度不足代价，A-035 仍会漏掉（预测 6.057，原生 4.900 m/s）。从真实首碰当帧重新开始单步预测，A-035 的步末速度仍高估约 1.02 m/s。该批结果用于观察出球辅助能否识别失分原因；当时辅助权重为零，其梯度对学习的实际贡献需在启用后评价。[同策略、同命令的逐案例出球复核](revised_post_a8_release_speed.json)

### R16 后复核同一批真实击球的 WM 刷新效果

保持 111 个案例的起点和三步执行命令完全相同，对比使用本批在线数据前后的 WM（第 152 与第 160 次更新）。两版都预测到 110 个首碰；在这些相同案例上，出球速度 MAE 为 0.34170→0.34151 m/s，实际低于 5 m/s 却预测达标的案例均为 11 个。按本轮训练分区，其中留出 lane 上的 69 个共同命中案例为 0.35190→0.35106 m/s。6 个仅因速度失去落点分的案例中，两版均漏报 3 个。共享 WM 确实消化了新数据，这一次刷新对速度门槛的识别尚无可测改善。TensorBoard 的 G1_release_audit 按 WM 实际更新数记录两版结果。[同命令、同案例配对](revised_release_after_wm_refresh.json)

修正版到第 512 个在线控制步共新增 65,536 条真实转移，R16＋A8 已保留。首批 6 个完整回合平均 Return 为 −21.123，同案例冻结基线为 −26.282，逐例平均差 +5.159；4 例增加、2 例减少。合格触球 5→5、球侧完成 3→3、身体触地 6→6、Complete5 为 0→0。这 6 例均在 512 步内结束且记录到身体触地，尚有 122 个首回合未结算。当前还不能判断固定策略的恢复率是否提高。[逐例终局评分](revised_first_completed_cohort.json)

首批 6 例中，Return 上升的 4 例在冻结基线中都运行到了 800 步，本轮分别在 440、452、505、512 步提前结束。A-046 两次都未合格触球。本轮 505 步归档奖励之和与终局 Return 一致，未触球持续扣分合计 −10.25。相对基线少经历的 295 步减少了 14.75 分扣罚，解释了该例 +18.42 分变化的约 80%。这项核对只归因 A-046 的未触球扣分。TensorBoard 同时记录配对回合时长变化，以及由跑满变为提前结束和反向变化的案例数。[A-046 归档及计分依据](revised_case46_episode_duration.json)

### R16＋A16 后的第二批终局

第 513–640 个原生控制步新增 8 个首回合终局。这 8 例的 Return 均值从同案例冻结基线的 −5.15 降至 −26.03，配对差 −20.88。合格触球 6→6、球侧完成 3→3、身体触地 4→8、物理恢复成功 4→0、Complete5 1→0；8 例在基线均跑满 800 步。新增的 4 次身体触地按原生每步扣分追溯，实际发生于第 462、474、494、504 步，当时执行 R16＋A8（版本 77），结算时才换成 R16＋A16（版本 85）。触地前一帧 4 例的 Recovery 门控均已开启，机身倾角为 38.2–40.0°。两例没有足端支撑，另两例仍有同样的两个足端报告接触。至 640 步，14 个已结算首回合总体配对差为 −9.72；尚有 114 个未结算。这批早结束回合显示真实的落地失分，尚不能估计固定策略在 128 例上的恢复率。[逐例终局与首次身体触地归档](revised_second_completed_cohort.json)

### 真实失稳前的 Body WM 与支撑边沿覆盖

从新增身体触地的 4 个案例各自的实测状态起步，按实际执行目标自由预测至首次身体触地前一帧。交接权重 W136 与第 32 次策略更新保存的 W168 使用完全相同的窗口。W168 从 20、40 步前预测，4 例均正确预测倾角增加，但平均分别低估 3.31°、6.71°；W136 分别低估 3.25°、6.53°。排除在线拟合的 lane 86 和 97 在末步同时失去两脚支撑。即使从前一帧的真实状态起步，W168 给两个即将卸载的足端继续接触的概率也分别为 0.992/0.964、0.991/0.942。另两个训练 lane 保留两脚接触的结果预测正确。这四个按事后失败选出的窗口用于定位误差。

同一行为采集块的训练 lane 有 4,140 步符合已分离、已落地、尚未身体触地的条件，没有从有支撑降到零支撑的转移；两个实际边沿都在留出 lane。初始冻结归档的训练分区有 6 次此类边沿，来自 5 个 lane。为后续扩展加入按实测转移本身取窗的 WM 配额：从已有 20 个失支撑窗口中划出 4 个，保持 batch 与真实交互预算相同。对真实初始化归档试采 128 个 16 步窗口，满足请求、无 fallback、未使用留出 lane。当前 G1 和匹配 G0 的首轮 256 次登记仍沿用同一配置。扩展配置的策略收益需在实际训练中验证。[窗口、权重与覆盖结果](revised_prefall_forward.json)

### Recovery 想象过程中发生的首次着地

旧的落地后纠偏项只在想象起点读取着地标记。从触地前一帧开始，即使随后恢复支撑，整段 rollout 也不会计算该项。新选项在已触球且分离的飞行中读取下一控制步的支撑预测，从着地当步开始计入倾斜、角速度、高度与支撑代价。候选复评保留原提案已有的监督时段。两个匹配实验的扩展配置均启用这一改动。

用第 32 次更新的实际 Actor、WM、Critic 和两个训练 lane 的原生触地前状态做 8 步 CPU 复算。新增 15 个有效 lane-step，Recovery 梯度差范数 0.05546。人为取消候选的预测着地后，有参照权重的候选损失仍为 7.24550，与原提案一致。该检查的优化器步数为零，真实任务收益待部署后评价。[起点、权重与复算结果](revised_rollout_landing_shadow.json)

### 相同 128 个初态的第 640 步核查

对上一轮 main_02 和本轮 G1 的相同首回合累加真实逐步奖励，提前结束的案例保留终值，重置后的奖励单列排除。两轮的初始 55 维状态逐元素一致，Actor 起始权重、任务、执行合同和种子一致。已结束的 12 / 14 个回合均复现各自的终局 Return。前 256 步两轮的逐案例累计回报完全一致。在第 384、512、640 步，本轮相对上一轮的累计均分差分别为 +0.764、+0.627、−0.398。第 640 步，本轮身体触地项每例多损失 2.8125 分，相当于在全部 128 例上净增 9 次。击球和漏球相关奖励抵消了一部分失分。该对照比较两个在线训练版本在相同回合时点的前缀，仅用于观察失分何时发生。

相对同案例的冻结起始策略，本轮第 640 步已有 63 例受过身体触地扣分，冻结策略全程有 60 例。13 例是冻结策略全程未发生的新触地，其中 9 例尚未结束。即使剩余回合全部不再新增触地，本轮首回合净增加数也至少为 3。这里的 63、60 借助整组 128 案例计算，与只比较已结束回合的 Return 分母不同。[逐步奖励、事件与源数据身份](revised_first_640_paired_prefix.json)

Recovery 想象着地修补后的 SuperMimic CPU 测试为 716 项通过，1 项第三方接口弃用警告。运行中的首轮对照继续使用其启动时的实现。

### 第 768 步暂缓策略更新

R24＋A16 最新采集结束后，6、14、16 个已结束回合的 Return 均值分别为 −21.12、−23.93、−27.26。与同案例的冻结基线比较，累计差为 +5.16、−9.72、−8.95。最后一批只有两个新回合。全部 128 个首回合已有 67 例出现原生身体触地扣分，超过冻结策略全程的 60 例。16 例是新增的身体触地，12 例尚未结束。基于连续下降的原生曲线与已不可逆的净增触地，提前暂停并启动同一 R24＋A16 行为权重的固定 128 案例复评。

A17 没有保留策略更新。停止处理器保存了最后 128×128 条转移；续训准备已在真实断点验证，只从中断快照接回 Replay、起点与物理交互计数。模型和优化器取自复评的块入口。新的匹配训练入口启用想象着地监督与实测失支撑转移配额，继续使用原学习率和动作预算。策略是否回退由冻结配对结果决定。[暂停证据](revised_native_return_pause_768.json)
