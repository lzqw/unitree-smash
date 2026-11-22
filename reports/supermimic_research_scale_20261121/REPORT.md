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

### R24＋A16 同策略冻结复评与恢复训练

用种子 882072 和相同的 128 个初始状态完成冻结复评。相对本次规模实验的起始 Actor，平均 Return 从 −12.09 到 −14.89，逐案例均差 −2.80，配对 bootstrap 95% 区间 [−10.80, +5.24]。合格拍线触球 74→65，球侧完成 51→36，Complete5 22→16。16 个原来合格的案例变为拍框先碰，7 个原来拍框先碰的案例转为合格。身体触地 60→49，新增 16 例、避免 27 例。以两种策略都合格触球的 58 个案例为分母，成功恢复 25→21。在线已结束回合的身体触地失分与单一冻结策略的全案例结果应分别解释。[逐项指标、案例与权重身份](revised_R24_A16_frozen128.json)

保留已评分的 R24＋A16 权重及对应优化器，接回暂停前的完整真实 Replay。该点的 G1 出球辅助有效更新次数为零。在下一次 Approach 提案启用辅助之前，修正其首次接触资格。只有首次击中拍线的选中 lane 可以从出球质量获得收益；首次打框和未触球均承担固定代价，始终使用所有选中 lane 做分母。拍线边界与拍框竞争的连续几何项继续提供纠偏梯度。新增回归检查验证了混合批次的分母、合法首碰的梯度和拍框、未触球的零出球梯度。修补版 SuperMimic CPU 回归 719 项通过。全仓 CPU 另有 24 项缺少旧机械臂 URDF，10 项依赖未提供的旧 PPO 权重，1 项旧速率上限断言失败，11 项跳过。修补后的冻结策略收益需要后续原生评价测量。

首批续训在写完 128 个原生控制步后，监控计数器因局部变量作用域错误退出，当次 Actor 更新为 0。修正累积计数后，回放暂停前全部 768 个原生控制步，复现第 640、768 步的 63、67 个身体触地案例；新流前 128 步为 0。用同一已评分 Actor 和 WM 在 RTX 5080 上从 8 个真实训练起点跑 H64，5 个首碰拍线、2 个首碰拍框、1 个没有预测到首碰球拍。拍线限定的出球项 Actor 梯度范数 0.26645，拍面跟踪梯度范数 5.71492。该检查不更新策略。修正后 SuperMimic CPU 回归 720 项通过，使用相同的已评分策略和优化器重试真实训练。[原生重放与 CUDA 检查](revised_first_string_preflight.json)

重试后 G1 已实际保留 A17，累计 R24＋A17。128 环境完成 16,384 条真实转移，共享 WM 更新 8 次，通过相对正式锚点的 forward 检查。16 个有效训练起点中，预测首次触线 3 个、先碰框 9 个，另 4 个没有预测到球拍首碰。A17 的出球辅助与 TOI 主项的加权梯度范数比为 0.001935；6 个 Approach 参数张量发生变化，同状态执行预览中的最大关节目标变化为 0.005624 rad。训练目标从 2.04615 到 1.99829。当前新流的完整原生回合为 0，策略真实收益仍待测量。[续训实际记录](revised_R24_A17_resumed.json)

A17 之后，第 18、19 次 Approach 抽样连续选中 success_near。两个批次实际使用相同的 15 个 lane／episode／起点 tick，记录顺序不同。Actor、Adam 和 WM 在两次尝试间相同；5 个缩放都满足执行预算和 TOI 覆盖，但同批目标均高于提案前的 0.673209。两次均未保留，BPTT 与候选复评分别用时 416.6 和 424.9 秒。完成 256 次后的匹配扩展将四类起点交错排列，每 20 次仍请求 5/7/5/3 次，避免相邻提案连续请求同一稀疏类别。首轮 G1 和 G0 继续使用原登记顺序。新增首球报告可绑定本次恢复点的冻结评价，并记录每例首次触拍时的行为策略版本。[两次提案的试探步长、目标与起点](revised_success_near_retries.json)

### 11 月 22 日续训的回退耗时与原生 Return 监控

G1 在保留 A17 后，第 18、19、20 次 Approach 抽样全部选中相同的 15 个成功附近起点（按 lane、回合、tick 去重）。三个提案分别耗时 416.58、424.90、418.53 秒，合计 21.00 分钟。每个提案的 5 档更新均使同批想象目标升高，均恢复了 Actor 参数与 Adam 状态。该区间没有新完成的原生回合。

起点汇集器现在先检查候选记录中是否还有本批尚未选中的 lane／回合。整条记录都已重复时可省去 TOI 预演；只要还有一个新回合，就保留原有预演和轮换顺序。当前已启动的 G1 继续使用启动时的汇集器。尚未测得该优化的端到端 GPU 提速。

原生 Return 监控已绑定至后续训练入口。当前 G1 的独立监控已切换至新版本，GPU 训练进程保持运行。监控识别持续下降并暂停后，会保留同一批尚未消费的回合边界，恢复后继续逐批判断。两项修改后的 SuperMimic CPU 回归共 725 项通过。

### 同一批成功起点上的 Adam 更新方向

使用已评分的 R24＋A16 块入口 Actor、WM、Critic、Approach Adam，在 CPU 重放线上连续三次回退使用的 15 个 lane／回合／tick（5 个控制器快照，H64）。出球辅助权重为 0.1。零位移复算完整目标与原值 0.641085 相同。原 Adam 位移与损失梯度的内积为 +0.023502，1/16 和全步损失分别为 0.645945、0.669053。单独清零一阶动量、保留二阶矩和步数后，内积为 −0.015582，对应损失为 0.640205、0.622929。各次预演均为 11 个拍线、4 个拍框。训练中的 A17 和在线刷新后的 WM 权重不同；这项复算定位的是块入口在这些状态上的方向问题。真实成功率与 Return 收益仍待冻结原生评价。[逐项结果](scored_handoff_adam_direction.json)

修复保留独立 Adam。当 Approach 的全步位移与实际评分损失梯度内积非负时，可用当前梯度重建一阶矩，再按原步长梯度回退及执行预算评分。拒绝候选仍恢复原 Actor 和两个 Adam 矩。后续扩展自动采用该规则；配对修订入口可同时为 G0/G1 启用起点交错和该规则。对候选重复评分已去掉中间 Critic 训练特征构造，保留末态价值。两个原生 Replay 着地起点的 8 步 CPU 复算中，Critic 编码调用 18→2，任务、跟踪、锚定、总损失和回报逐值相同。该小样本计时没有显示稳定的端到端提速。

05:08 运行交接已核实。旧 G1 连续五次回退的 BPTT 与候选评分共耗时 46.54 分钟。请求正常保存退出后，在途提案又保留了 A18，因此旧运行最终为 R24＋A18，完整的 interrupted.pt 与 16,384 条新增 PhysX 转移保留在原目录。修复版 G1 已在 RTX 5080 启动，注册参数同时启用起点交错和上坡动量修复，G0 将由同一调度器使用相同修复。两组从已评分的公共 R24＋A16 断点计 40/256；旧运行的两次 Approach 更新单列。新启动时处于模型和 Replay 恢复阶段，尚无修复后冻结策略的原生 Return。原生 Return 监控已跟随新训练进程，TensorBoard 运行组为 paired_adam_direction_0040。此次 SuperMimic CPU 回归为 728 项通过。

### 05:39 修复版 G1 首次在线 Approach 更新

公共 R24＋A16 断点恢复后，首个提案实际抽到 success_near，5 个控制器快照共覆盖 15 个有效 lane。原 Adam 位移沿损失上升，内积为 +0.02257。重算一阶动量后降至 −0.01481，全步通过原执行预算和 TOI 覆盖检查，只评分一次即保留 A17。想象损失 0.65087→0.63360，预览的最终关节目标相对上次更新最大变化 0.001351 rad。修复版当前为 R24＋A17，共 41/256 次；旧试运行单列。[实际在线提案](paired_adam_first_online_A17.json)

本次恢复后采集 128×128 条真实 PhysX 转移，8 个共享 WM 批次共使用 1,024 个窗口，包含 192 个近期在线窗口与 64 个较早在线窗口。在本次采集策略版本 93 上记录到 208 次 transition 使用，可能包含相同版本此前的记录。更新后的 WM 通过固定基准前向检查。采集、WM 更新与 BPTT 分别用时 164、192、196 秒。当前新流完整原生回合数为 0，A17 的冻结 Return 与 Complete5 待评。

新版采集入口在每次研究块收集结束即发布原子归档清单，首球和接触曲线可在下一提案前读取已写盘的数据。已启动的 G1 仍按原检查点节奏发布归档清单，独立 Return 监控照常跟随逐块回合记录。SuperMimic CPU 回归 729 项通过。

05:45，修复版连续保留了 A18，累计 R24＋A18。16 个实际 pre_contact 起点均匹配参考。原 Adam 位移的损失斜率为 −0.003729，沿用其一阶动量。全步使想象损失升至 1.52129，半步将 1.49342 降至 1.49227，最大最终关节目标变化 0.000878 rad。16 个起点的半步预演有 6 个拍线首碰和 7 个拍框首碰。该提案复用块内固定 WM 和 Replay，尚无本次恢复后的原生完整回合。

后续训练入口在真实控制步有回合结束时，将最近最多 128 个完整原生回合的平均 Return 和分母写入 TensorBoard。同一控制步的各 lane 一起计入；块内反复评分 Actor 不会再写重复的 Return 观测。独立暂停监控的判据保持一致。

出球代价在一批球全部落地后结束剩余积分。使用冻结 R24＋A16 原生评价中 108 条已分离首球的接触位置和分离速度作为输入，与原 100 步实现逐项比较。CPU 上代价和对六维输入的梯度最大绝对差均为 0。5 次交错计时的前向＋反向中位耗时从 26.2 ms 降至 17.0 ms。这是出球代价的局部计时；真实训练吞吐仍以块级墙钟记录为准。上述两项改动随下一训练进程加载，本次 G1 持续运行。SuperMimic CPU 回归 730 项通过。

### 多记录候选评分与实际起点覆盖

修复版 A17～A20 连续保留，依次用到成功附近、接触前、门控开启和失败前四类起点。A19 将 12 个控制器快照中的 16 个起点汇成一批，BPTT 和候选评分耗时 443.5 秒。A20 用到 4 个快照的 16 个失败前起点，耗时 156.5 秒，目标从 2.04836 降至 2.00910。两步都在全步长通过。A21 与 A22 也已保留。在线 Return 尚待新流回合结束。

CPU 采样定位到候选评分会逐步复制 G07 的历史计划。训练起点使用归档原有的 64 位浮点数值，在私有想象状态中直接将公开观测矩阵存为连续数组，控制器仍逐步复制自己可修改的状态。检查当前断点的 193 条不同起点快照，EarlyLateral 和 SharedTarget 的游标均已读到计划日志末尾；MomentumStrike 的请求增益在这 193 条中均为零。球模型还省去了无梯度评分时整批无延迟球拍冲量的一段步末积分。

冻结 R24＋A16 的 7 个真实起点、两份不同控制器记忆上，H64 回放逐项比较 8,064 个最终目标分量和 111,186 个 Actor 梯度分量，最大绝对差均为 0。单独改变计划矩阵存储后，无梯度候选评分中位数 9.50→8.51 秒；含反向的用时 29.35→29.69 秒，未测到反向提速。省去无用积分在单独对照中将两条 H64 的 prior.advance 调用数从 512 降至 390。两项计时均来自 CPU。在线 GPU 进程继续按启动版本执行。SuperMimic CPU 回归 732 项通过。[全部计时与逐值复核](compact_plan_scoring_parity.json)

修复版 G1 已保存 R24＋A24，首轮累计保留 48/256 次。A17～A24 八个提案全部保留，A18 采用半步，其余七个全步通过。127 个起点样本来自 33 个 lane、55 个不同的 lane／回合组合，实际类别为接触前 3 次、门控开启 2 次、失败前 2 次、成功附近 1 次。八次 BPTT 和候选评分合计 32.8 分钟。下一 Recovery 块已用此 checkpoint 的策略完成 128 个控制步、16,384 条 PhysX 转移。本次续训两批共 32,768 条，两个批次结束时均无完结回合，尚无本版本的完整原生 Return。[逐提案与原生采集记录](paired_adam_A24_block.json)

### 第 25 次 Recovery 更新的数据来源

在新策略 R24＋A24 的 4 个落盘数据块中核对了全部 16,384 个原生控制样本。128 个环境均已发球，本段合格触拍、拍框和身体触地事件均为 0。在 45 个可训练 lane 上有 5,085 个连续的 16 步窗口，其中没有落地、失支撑或稳定恢复事件。此次 WM 实际使用了 192 个近期在线窗口及 64 个较早在线窗口，版本 101 的 transition 使用次数为 0。首轮登记的近期来源池覆盖多个策略版本，未为最新版本在普通窗口中单独保留份额。

在同一批观测和原生门控上分别重放 R24＋A16 与 R24＋A24。新策略与记录请求的最大绝对差为 5.96×10⁻⁸，66 个 lane 中共 289 个观测有超过 10⁻⁶ 的请求变化。门控大于 0.25 的 46 个观测上，新旧请求的 RMS 差为 0.01233（归一化动作单位）。这些数据覆盖接触前的控制，尚无终局评分。R25 的 16 个有效起点中有 3 个来自身体触地前 20 或 40 tick，另有一个来自首次落地 2.24 秒后、足端全部无支撑的状态。全步保留，想象目标 5.872→4.505。[采集身份、起点与逐来源统计](r25_online_windows_and_requests.json)

候选评分现在为每份真实起点复制一次 G07 控制器，在该候选的 64 步评分内推进私有记忆。CPU 用两个碰后失稳前起点、两个 Approach 控制器快照分别交错复算，全部关节目标、总损失、各损失项和末态价值参与的回报逐值相同。单次无梯度评分的中位耗时在这两组分别从 6.68 降至 6.06 秒、从 8.60 降至 7.62 秒。当前运行中的 G1 仍沿用启动版本，本次计时范围只覆盖 CPU 候选评分。SuperMimic CPU 回归 733 项通过。[各次计时](candidate_controller_scoring_parity.json)

### 扩展阶段的 Recovery 起点覆盖

用 R25 块入口的同一份真实起点库、相同的 83 个排除 lane，在 CPU 上逐项复现了 R25–R29 的 80 次实际起点选择。固定起点库并模拟连续四个 R8 块，每块重置采样序号共覆盖 84 个不同的 lane／回合／tick，跨块连续计数覆盖 163 个；独立 lane／回合组合从 37 增至 44。两次模拟均抽取 512 个起点，其中风险、着地、稳定样本分别为 256、128、128。正式累计扩展入口使用按 Recovery 历次提案累计的序号，随 checkpoint 保存采样模式，并将每次使用的序号写入训练历史。当前 G1／G0 首轮保留原采样设置以维持配对。该对照仅衡量起点覆盖，策略收益需看 PhysX 完整回合。SuperMimic CPU 回归 733 项通过。[逐项统计](recovery_block_start_rotation.json)

### R32＋A24 之后的 384 tick 首球对照

本次 R25–R32 八步均以完整步长保留，每次使用 16 个有效起点。128 次抽样覆盖 84 个不同的 lane／回合／tick。其中 31 次在身体触地前启动，分别有 15、8、8 次提前 20、40、60 tick；28 次发生于落地后 1–3 秒。该 Recovery 块的 BPTT 计时合计 47.4 分钟。

新一批 16,384 条 PhysX transition 落盘后，续训累积 49,152 条。用恢复点 R24＋A16 的冻结 TRAIN128 记录配对，至第 384 tick，123 个案例的首次球路已结束。其中合格触拍 65→68，拍线转先碰框 5 例、先碰框转合格拍线 8 例。符合计分条件的越网 44→43，目标落点 37→37；丢掉目标得分的 9 例中有 4 例仅出球速度未达标。123 例的三项实际击球奖励合计比冻结对照多 55 分。105 例已记录的分离事件均使用行为版本 109。实时数据包含多个块的前缀，不能代替冻结评价。

128 个首回合均未结束。15 例已经身体触地，其中 14 例在恢复点冻结评价中也触地，A-720401-009 为已确认新增。其余案例仍可改变最终净差。完整 Return 暂无新观测。首球结果已写入 TensorBoard 的 G1_first_flights_since_R24_A16，横轴为本次续训环境步数。[逐案例、奖励与原生数据块身份](paired_adam_first_flights_384.json)

### 首球采集进入 A25 的 WM 与接触监督

R32＋A24 的行为版本 109 在 A25 块的八次 WM 更新中贡献 2,819 次 transition 使用，其中有 25 次拍线／拍框事件边沿使用。八批合计 1,024 个 H16 窗口，允许窗口重叠。接触模型执行四次额外监督优化，使用各来源合计 240 个接触事件样本。A25 使用 14 个有效起点，完整步长保留；加权出球辅助梯度范数约为 TOI 主梯度的 2%，两者余弦为 −0.232。

用采集时的 WM192 和记录的三个连续执行目标复算 110 个原生首次触拍窗口。从接触前一个真实状态开始，模型在三个控制步内预测到全部 110 次拍击。30 例原生出球低于 5 m/s，其中 12 例被预测为达标。在仅因速度失去目标分的 4 例中，模型均在原生接触当步预测先碰拍线。A-007 和 A-043 预测为 5.448、5.324 m/s，原生为 4.886、4.575 m/s，两例的预测值都超过训练裕量 5.25 m/s。另两例的预测低于裕量。由此能定位短窗口代理对这两例的速度惩罚为零；A25 实际 BPTT 使用自己的起点，逐案例策略梯度未测量。此次复算使用 CPU，GPU 持续训练。[版本、训练批次与逐案例预测](paired_adam_A25_native_WM.json)

### 出球速度高估来自哪一段

对 WM192 的 12 例“预测过 5 m/s、原生分离未达标”做同案例速度差分。模型提前一帧启动带来的平均差为 +0.064 m/s；从原生接触当步的状态和执行目标出发，WM 步末相对 PhysX 步末的平均误差为 +0.510 m/s；PhysX 步末与原生计分锁存的平均差仅 +0.011 m/s。12 例均预测到正确接触控制步。

在相同起点、执行目标和接触模型下，将 Body WM 预测的 49 维机器人步末状态替换成实测值。110 个窗口的步末球速 MAE 从 0.295 降至 0.202 m/s，在训练排除的 70 个 lane 上从 0.315 降至 0.229 m/s。A-043 的预测从 5.288 变为 4.328 m/s，实测 4.576；A-007 从 5.361 变为 5.172 m/s，仍高于实测 4.896。机器人步末轨迹解释了部分误差，实测步末仍未重现完整子步接触。此项 CPU 事后归因可以用出球审计程序的 `--robot-endpoint-oracle` 重放；两次单独执行的 110 个案例结果吻合。CPU 回归 733 项通过。[分项数据](paired_adam_WM192_body_endpoint.json)

### R32＋A32 里程碑与首批完整回合

首轮累计保留 64/256 次更新，已核验里程碑 checkpoint 的 Actor 和 WM200 权重。A25–A32 八步均全步保留，每步 14–16 个有效起点。用相同的 110 个原生窗口和三步执行命令比较 WM192、WM200，两者在窗口内都预测到 110 次拍击。出球速度 MAE 为 0.3151→0.3169 m/s，5 m/s 门槛的高估误判 12→13；训练排除的 70 个案例上高估误判 7→7。A-025 的预测从 4.9992 变成 5.0025 m/s，原生为 4.7930 m/s。八次 WM 更新已经吸收本轮数据，这次更新没有减少近门槛的速度误判。[配对复算及块内更新](paired_adam_WM200_refresh.json)

本次续训采集达到 512 个控制 tick、65,536 条 PhysX 转移。全部 128 个首球均已结束。相对续训起点 R24＋A16 的同案例冻结策略，合格拍线 65→71，计分越网 44→45，计分落点 37→37；先碰框转合格拍线 11 例，反向 5 例。到此时 26 例已发生身体触地，其中 5 例在冻结对照的完整回合中未发生身体触地。待其余案例结束后才能比较最终净差。

最先结束的两个完整回合平均 Return 为 −28.685。相同两例在续训起点冻结策略下为 −29.657，在实验起点冻结策略下为 −29.144；配对平均差分别为 +0.972 和 +0.459。两例均发生身体触地，完成顺序有选择性。本批只有一个非空完整回合观测点，尚无持续下降趋势，训练继续。[原生事件、逐案例配对与行为版本](paired_adam_native512.json)

### R33 的真实落地数据回流

本轮在 R32＋A32 采集后的首次 Recovery 提案以完整步长保留，累计 R33＋A32。16 个不同回合起点中，风险纠偏 8 个、着地转换 4 个、稳定维持 4 个，16 个均有有效门控和非零 Tracking 梯度。8 批 WM 更新共使用 1,024 个 H16 窗口，其中包含行为版本 117 的 286 次 transition 使用。窗口可以重叠。对 Actor 起点复核 lane、episode、tick、支撑脚数、倾角和落地时钟，3 条与该版本新采轨迹吻合，其中一条只有一足支撑，距离着地约 0.94 秒。采样日志没有逐条保存原生流 ID，因此上述匹配只表明记录状态一致。

截至本地 tick 512，相对同案例的完整冻结续训基线新增 5 个身体触地案例。3 例在触地前 0.6 至 0.2 秒的取样点及触地前一帧均已触球并分离，最后 0.2 秒倾角从约 28–30 度升至 36–39 度。另 2 例直到身体触地时仍未触球，Recovery 门控关闭。5 例中 3 例属于留出 lane。当前只有 2 个完整回合，尚不能据此判断修补改变了完整回合成功率或 Return 趋势。[逐例原生状态和 R33 采样](paired_adam_R33_prefall.json)

### 触地前的机身倾角和实际支撑脚

用 WM200 从实测状态开始，逐步执行归档的最终关节目标和上一动作，自由预测身体、球及足支撑。对 3 个碰后新增身体触地案例，提前 20 tick 预测最后一帧机身倾角为 38.45、36.59、36.06 度，PhysX 分别为 38.63、36.34、37.31 度。3 例的支撑脚数均预测为 2，但归档顺序下的实际支撑均为 0101，预测均为 1100。2 例是 WM 训练排除的 lane，另外 1 例来自较早且允许参与拟合的行为轨迹。误差衡量实际执行命令下的前向轨迹。

另取 B-034 身体触地前一帧（本地 tick 412），对从 tick 392 连续到该帧、已触球分离且尚无身体触地的全部 96 个案例做相同复算。H20 的支撑脚数正确 49/96、四脚状态全部正确 47/96；训练排除的 61 个案例上分别为 37/61 和 35/61，直接沿用起点足状态的完整匹配为 33/61。这个时点按故障案例选取。该结果揭示支撑脚数指标遗漏的错误，尚不能将 Return 变化归因于足接触预测。落地窗口审计已补充完整支撑状态准确率和逐脚误差，TensorBoard 已写入本次复算的分母和结果。落地和真实 Return 监控的 11 项针对性回归通过。[逐案例、时域及接触模式计数](paired_adam_WM200_prefall_support.json)

### R34 回退原因与复算时间

R34＋A32 共保留 66 次更新。R34 完整步在相同起点上的想象目标由 −9.47555 降至 −9.78148，但相对上次接受策略的归一化执行命令变化为 0.02071，超过预算 0.02000。半步变化为 0.01024，目标为 −9.62950，实际保留。两次候选评分之间没有新增原生回合。该提案的 BPTT 和复算合计 521.96 秒。

上一完整 R8、A8 分别耗时 54.82、33.16 分钟，其中 BPTT 和复算共 69.71 分钟，占该组墙钟的 79.2%。后续训练入口对执行预算已明确排除的有限值中间候选跳过 H64 目标复算，记录预算和未评分原因。仍逐一检查执行预算，按原判据评分可接受的候选，全部失败时保留最后一次诊断评分。正在运行的 G1 使用启动时实现。新代码的整步节时尚未实测。[计时与两次候选记录](paired_adam_R34_budget_timing.json)

TensorBoard 增加当前 G1 与已评分 R24＋A16 同案例的逐回合 Return 曲线，目录为 `G1_returns_since_R24_A16`。该只读进程校验冻结评价的 checkpoint、Actor、任务、执行合同和 seed。首两例累计配对差为 +0.97225。原实验起点对照继续单独记录。SuperMimic CPU 回归共 736 项通过。

### 球拍几何前向及反向耗时

球模型几何特征和 TOI 球拍目标只读取拍面位姿与速度。两处现复用原全身 FK 的相同逐关节运算，沿资产中通向球拍的 9 条连接求值，原全身 FK 处理 27 条连接。需要足端和整机质心的调用继续计算整机几何。用原生状态在 RTX 5080 上交错测量两项特征与一次输入梯度，batch 16 中位数由 19.90 ms 降为 7.49 ms。batch 1、16、128 的特征和输入梯度最大绝对差均为 0。计时与在训进程共享 GPU。

另用 WM200 在 12 个真实接触窗口上从接触前一步连续预测三个控制步，使用逐步记录的执行命令，CPU 两条路径都在中间一步预测到 12 个球拍接触。55 维状态和出球速度加权和对三个执行目标的梯度最大绝对差均为 0。SuperMimic CPU 回归 737 项通过。当前 G1 仍在原进程中，本次耗时为局部测量。[时域、案例与计时记录](racket_chain_parity.json)

### R40＋A32 后的真实 Return

G1 累计保留 72/256 次。R33–R40 共八次 Recovery 更新，每次有 16 个有效起点，首个全步保留，后七次半步保留。使用 R40＋A32 新采 128×128 条 PhysX 转移，本次续训累计 81,920 条。本地第 640 个控制步已有 3 个首回合结算，128 个首回合中已有 29 个发生身体触地。

新结束的 B-049 将最近已结束回合的 Return 均值从 −28.685 拉到 −37.627。B-049 在线为 −55.513，实验最初冻结策略为 −31.694，续训起点 R24＋A16 冻结策略为 −56.752。原生奖励归档显示，身体触地扣分发生于该回合 tick 477，执行 R32＋A32，Recovery 门控关闭；回合在换用 R40＋A32 后结算。最初冻结策略在该案例未触地，续训起点已经触地。

相同三例的在线平均 Return 相对实验最初策略低 7.633 分，相对续训起点高 1.061 分。逐步奖励求和与三条终局 Return 吻合。逐例与续训起点相比，身体触地、合格触球、计分越网和目标落点的奖励差均为零。新增案例进一步揭示了较早阶段已存在的损失，当前样本不足以判定 R40＋A32 冻结性能。训练继续，原生 Return 暂停监控在运行。奖励归因程序现支持经 checkpoint 哈希核实的续训基线，并记录触地扣分当步的行为策略、Recovery 门控和结算策略。737 项 SuperMimic CPU 回归通过。[原生配对、逐项奖励与触地时序](paired_adam_native640.json)

### R40 的动作回放与 A33–A40 接触监督

G1 已保存 R40＋A40，累计保留 80/256 次。将 R32＋A32 和 R40＋A32 分别用于同一批原生观测和当步门控。两份权重只在 Recovery 的 6 个参数张量上有差异。13,952 条门控开启的首回合观测上，归一化请求差的 RMS 为 0.06457；2,069 条门控关闭的观测上请求逐项一致。R40＋A32 重放与 PhysX 采集时记录的请求最大绝对差为 1.79×10⁻⁷。该比较固定了观测，衡量进入 G07 前的策略请求。

本地第 513–640 步没有新触拍。此段新增 3 条首回合身体触地记录，三例在续训起点的冻结评价中也都曾触地。A-013、A-008 在扣分时 Recovery 门控开启，B-058 关闭。A33 前的 8 个共享 WM 批次共包含 1,024 个 H16 窗口，最新采集版本 125 有 244 次 transition 使用。接触模型另有 4 次监督更新、243 次接触事件抽样，最新版本的触拍边沿使用数为 0。相同验证规格下，WM208→216 的 H64 球速 RMSE 为 0.98362→0.95713 m/s，接触当步球速 RMSE 为 0.30689→0.30673 m/s，H64 根部角速度 MAE 为 0.16537→0.16658 rad/s。

A33–A40 每次有 14–16 个有效起点，6 个提案全步保留。A37 经四次同起点评分保留八分之一步，A40 保留半步。本块原生采集 3.24 分钟，共享 WM 刷新 3.34 分钟，八次 BPTT 与候选评分合计 20.77 分钟。A33 虽然总损失下降，预测首碰拍线从 8 条减至 7 条，预测首碰拍框从 4 条升至 7 条。TensorBoard 的 G1_learning 已补入 A17–A40 同起点上实际保留候选的首碰计数变化；相同起点可跨提案重复。这些是模型预测。738 项 SuperMimic CPU 回归通过。[原生动作回放、WM 对比与逐提案评分](paired_adam_A40_execution.json)

R40＋A40 随后采集了另一个 128 tick。本次续训累计 98,304 条真实转移，达到本地 tick 768。期间没有新增首回合终局，已有身体触地的首回合从 29 增至 30。按相同 128 个初始案例在 tick 768 截尾，续训起点的冻结策略已有 11 例结束，在线为 3 例。逐例有 8 例运行更久、3 例更短，其余相同；平均已运行时长为 748.44→761.90 tick。最初冻结策略在该时刻已结束 12 例。终止时间提供了独立于结算顺序的中途比较，完整 Return 和 Complete5 仍需等各回合结束。

### R41 的晚期失稳起点与原生数据回流

G1 已保留 R41＋A40，共 81/256 次。R41 的 16 个有效起点含 8 个风险状态、4 个落地状态、4 个稳定状态。5 个风险起点距实际身体触地还有 20 或 40 tick，位于落地后 1.59–2.78 秒。全部 16 个 lane 均产生 tracking 梯度。半步候选的同起点想象目标由 −11.1466 降至 −11.2185。WM 刷新至第 224 次，8 个批次抽取了 1,024 个 H16 窗口，最近原生策略版本 133 的 transition 计入 150 次，含重复使用。

在本地 tick 640–767 的四个原生文件中，能按 lane、episode、tick、足端数量、倾角及落地时钟复核 2 个 Actor 起点。两者位于落地约 4.44 秒后。旧进程没有按起点记录行为 Actor 哈希。后续版本会在 Recovery 批次中写出原生流、tick、行为策略及权重身份，已有归档的一条真实起点已完成回读核验。R41 沿用 tick 768 时的 3 条已结算首回合，尚无新增终局用于衡量此次更新的任务收益。

新代码在无需反传的候选评分中复用同一下一状态的全身正运动学。两个真实 Recovery 起点的 H64 交错计时中，RTX 5080 上中位数从 11.77 秒降至 11.08 秒，五项评分逐项一致。该测量与主训练共用 GPU，仅覆盖这段评分。当前进程继续使用启动时实现。SuperMimic CPU 回归 740 项通过。[起点、采样及计时](paired_adam_R41_start_coverage.json)

### 同一物理时刻的 128 案例累计奖励

在相同 seed、初始物理状态、控制和评分合同下，将当前 G1 与上一次暂停的 G1 逐 lane 对齐。每例从第一个控制步开始，累计真实逐项奖励，遇到首回合终局即停止。到第 768 步，两轮可用的首回合终局分别为 3 和 16 条，均已与对应的逐步求和核对。两轮起始 Actor 及后续更新历史不同。

| 首回合截点（tick） | 上一轮平均累计奖励 | 当前轮平均累计奖励 | 当前减上一轮 | 身体触地案例，上轮→当前 |
|---:|---:|---:|---:|---:|
| 384 | 4.58 | 5.93 | +1.35 | 23→15 |
| 512 | −4.70 | −1.49 | +3.20 | 43→26 |
| 640 | −13.63 | −5.38 | +8.25 | 63→29 |
| 768 | −18.65 | −9.30 | +9.35 | 67→30 |

第 768 步，71 例累计奖励提高、57 例下降，逐例差的中位数为 +0.064 分。减少身体触地平均贡献 +11.56 分，目标落点奖励平均贡献 −0.94 分，按 15 分/例折合净少 8 例。全部奖励项相加与直接观测的差一致。该结果说明本轮混合策略在这些案例上的累计身体触地损失减少；最终恢复率、Complete5 和当前单一 checkpoint 的收益仍需终局及冻结评价。TensorBoard 新增 `G1_first_episode_prefix_vs_paused_run`，横轴为本地控制 tick。[六个截点、原生归档哈希和逐项奖励](paired_adam_native768_prefix_comparison.json)

### 首球目标落点翻转与想象中策略请求的开销

对比上一轮暂停前与当前轮的全部 128 条首球。两组首球有实测分离的分别来自策略版本 69、109。按逐例评分条件和原生一次性奖励复核，目标落点从 45 例降至 37 例，15 例失去得分、7 例新增得分。按第一个未满足的条件归因如下。条件可同时失败，表内每例只计一次。

| 首个失分条件 | 本轮失分 | 本轮改善（上轮在此失分） |
|---|---:|---:|
| 合格拍线触球 | 9 | 5 |
| 缺少有效分离或出球速度不足 5 m/s | 4 | 2 |
| 满足计分越网但落点未入目标区 | 2 | 0 |

9 个失去合格触球的案例中，8 个记录了拍框接触，1 个未触拍。两轮都满足计分越网的 36 个案例中，目标落点为 32→30。该对比衡量了两轮不同策略历史下的首球。正式策略收益继续由共同案例的冻结评价衡量。

后续想象 rollout 可复用同一原生 Recovery 锁存，同时直接取策略分布的确定性请求，省去 Actor 和冻结参考在每一 tick 上的 log probability 和 entropy。RTX 5080 实测 16 lane 的双策略请求、门控及单次请求反向合计用时，Recovery 由 2.67 降至 2.33 ms，Approach 由 2.78 降至 2.50 ms。1、16、128 lane 的动作逐项相同，输入和参数梯度最大绝对差 2.92×10⁻¹¹。两个真实晚期失稳起点的 H64 CPU 总目标和梯度逐项一致；单次 H64 CPU 反向用时为 16.05→16.44 秒。局部 GPU 计时与在训进程共用显卡，整组训练吞吐尚待下一次启动实测。742 项 SuperMimic CPU 回归通过。[逐例失分、改善及计算路径核验](paired_adam_A40_first_target_and_requests.json)

### 未触球后的 Recovery 覆盖

截至本地 tick 768，首回合发生身体触地的 30 例中，21 例在扣分时已启用碰后 Recovery，另 9 例从未触拍，执行门控关闭。核对真实逐步记录后发现，这 9 例在扣分前 14–229 tick 已同时满足球已发射、已落地、已退出有效球路、无首次触球及拍框／拍线接触。18 个首回合曾进入这一可因果识别的未击中状态，在首回合运行且尚未身体触地时合计覆盖 5,057 个 lane-tick。观测范围内这些球路随后均未重新进入有效飞行或触拍。该数据给出后续未击中恢复分支可介入的时间范围，其策略效果需要同执行合同的零更新基线。当前 G1 和配对 G0 继续沿用已登记的碰后门控。[实际门控、原生扣分与提前量](paired_adam_native768_miss_recovery.json)

### 同状态执行核验与零更新对照

固定 R40＋A40 权重，在上述 9 例各取首次可确认漏球与身体触地前的两个真实状态，恢复当时的 G07 执行器记忆。18 个状态的最终执行目标均随门控切换而改变，单个状态的关节目标 RMS 差值中位数为 0.0889 rad，最大单关节变化 0.2136 rad。原生目标重放最大误差为 7.16×10⁻⁷ rad；同批 1,697 个门控相同的 lane 观测在两条执行路径下产生相同目标。[逐状态执行证据](native768_missed_recovery_command_preview.json)

新评价入口按当前球路的发射、落地、退出比赛与触拍锁存启用漏球恢复，并记录每个首回合的首次启用时间和累计控制步。另用已有完整评分的 R24＋A16 checkpoint 开展零更新 PhysX 配对评价，核对相同初态、权重和物理配置，以 128 个首回合的身体触地、Return 和任务成绩衡量效果。训练仍采用登记的执行合同。当前起点缓存支持保留真实 lane 身份的记录再次加载。745 项 SuperMimic CPU 测试通过。

冻结 R24＋A16、种子 882072 的 128×800 原生对照已完成。20 个首回合首次在第 305–319 控制步启用新增门控，平均 Return 变化 +3.772；其余 108 例逐例 Return 差为零。全组身体触地 49→47，3 例避免、1 例新增；平均 Return −14.8872→−14.2978，配对均值差 +0.5894，案例 bootstrap 95% 区间 [−0.5645, +1.9261]。合格触球 65/128、触球后恢复 24/65、球侧完成 36/128、Complete5 16/128 均持平，数值失败为零。这个变化来自同一策略在确认漏球后启用恢复动作。当前配对训练延续原登记门控，单独的 TensorBoard 分组记录本次零更新结果。[冻结权重、案例变化及配对统计](frozen_R24_A16_grounded_miss_pairs.json)

### G1 R48＋A40 与续训后的首批完整回合

G1 已保留 88/256 次更新。续训后的 128 个首回合在本地 tick 896 前全部结算，平均原生 Return 为 −7.7503。同案例冻结续训点为 −14.8872，配对差 +7.1369，案例 bootstrap 95% 区间 [1.58, 12.71]。身体触地 49→30，贡献每例 +5.9375 分；合格触球 65→71，贡献 +0.9375 分。球侧完成 36→35，Complete5 16→16。与实验最初的冻结组合相比，Return +4.3383，但球侧完成 51→35、Complete5 22→16。在线首回合包含多个行为版本，当前 checkpoint 的独立冻结评价仍待完成。

最近 R8 共用了 128 个有效起点，含 64 个风险、32 个着地和 32 个稳定起点。36 个身体触地前窗全部位于失败前 20–60 tick，覆盖 10 个 lane，其中 15 个已着地 1–3 秒。另有 8 个落地后失去支撑的起点。当前继续执行登记的 R8→A8 训练及原生 Return 自动暂停监控。[首回合配对成绩、原生奖励分解与实际采样](paired_adam_R48_A40_first128.json)

### 球侧完成的逐例归因与出球速度评分时刻

相对实验初始冻结策略，首批 128 个在线回合有 22 例失去球侧完成、6 例获得。按合法拍线、出球方向、速度、越网落区的顺序归类，22 例首个失分条件依次为 11、1、8、2。相对冻结续训点，9 例失去、8 例获得；失去的 9 例中，4 例缺少合法拍线、4 例速度不足、1 例越网落区未达标。上述首个条件计数互斥，文件中保留了每例全部失败条件。

原生评分要求分离时出球向前、竖直速度不高于 −1 m/s 且速率至少 5 m/s。71 例合格拍线中有 5 例分离时向下速度不足。在相同事件的首碰控制步末检查，只有 4 例低于所需向下速度。A-012 的分离速度为 −0.9345 m/s，到步末已达 −1.0117 m/s。原辅助目标检查步末速率、向前分量和后续飞行，遗漏了出球向下条件。

后续扩展阶段在现有接触辅助中，用同一 WM 在 TOI 施加解析及学习冲量后的速度计算速率及方向代价；位置、越网和落区继续使用完整预测步末状态。模型的瞬时冲量近似原生分离。已登记的 256 次 G1/G0 保持原有训练目标。R48＋A40 冻结权重的 GPU H64 验证中，首 4 个训练起点的出球辅助梯度范数为零，TOI 跟踪梯度范数为 9.92。另取同一时刻符合门控且真实首球曾因速度或方向失分的 11 个训练 lane，出球辅助梯度范数为 1.468，与跟踪梯度余弦为 0.623。这些是想象梯度，当前未用新目标更新策略。746 项 CPU 回归通过。[逐例原生条件、速度时钟与 GPU 梯度](paired_adam_R48_native_release_clock.json)

### G1 R48＋A48 与 H64 前向耗时

G1 已保留 96 次更新。本组 8 次 Approach 提案均被接受，其中 2 次缩小了步长，实际接触辅助／TOI 梯度范数比为 0.014–0.117。随后完成 128 个原生控制步，累计新增 229,376 个环境步，结束回合数仍是 128。续训起点的同案例平均 Return 为 −14.887，首批混合策略回合为 −7.750，Complete5 均为 16/128。

在冻结的 R48＋A48 上，用 4 个真实训练起点比较了 H64 想象。对无需飞行梯度的 CUDA 运算，将原来的 160 个物理子步合并在一次核调用中。四项目标和 6 个 Actor 参数张量的梯度逐元素一致。两次只评分的平均耗时由 8.00 秒降至 6.51 秒，带反传时由 21.11 秒变为 21.25 秒。计时期间主训练也在使用 GPU。该优化供两个研究组后续扩展阶段使用，当前 256 次对照继续使用原实现。746 项 CPU 回归和单独的 CUDA 比对通过。[本组更新及 GPU 逐次测量](paired_adam_R48_A48_flight_backend.json)

### 下一 Recovery 块的共享 WM

R48＋A48 后完成 128 个原生控制步和 8 次 WM 更新。WM240 在固定 32 个原生案例上的 H64 身体 RMSE 为 0.553，根角速度 MAE 为 0.167 rad/s，球速 RMSE 为 1.001 m/s。16 个独立 lane 的触球当步球速 RMSE 为 0.302 m/s。从 WM184 到 WM240 的四组相邻刷新中，Recovery 刷新后的 H64 球速误差均高于此前 Approach 刷新后的值。八次测量整体在 0.919–1.026 m/s 之间，H64 身体 RMSE 在 0.550–0.553 之间。这里比较同一组记录动作下的前向误差，尚无新终局用于归因策略收益。

另在 R48＋A48 的 4 个原生 Recovery 起点复核扩展阶段的 CUDA 实现。H64 中损失及 6 个 Actor 参数张量的梯度逐元素一致。仅评分的平均耗时为 6.87→5.19 秒，带反传为 14.06→12.50 秒，每种路径测量两次，主训练同时在运行。[固定面板误差、真实支持数和逐次耗时](paired_adam_WM240_forward.json)

### R49–R50 的数据回流

R48＋A48 策略的最新 128 个原生控制步覆盖 16,384 条转移。训练分区的 45 个 lane 可形成 5,085 个互相重叠的 H16 窗口，其中合格触拍、拍框、身体触地事件均为零，另有 1 次身体触地发生在留出 lane。本次共享 WM 的 8 个批次共 1,024 个 H16 窗口，实际使用了较早在线版本的数据，当前采集版本的 transition 使用数为零。4 次接触头监督共抽取 129 个事件，包含重复抽样。R49 和 R50 均在半步长保留，各自有 16 个有效起点。当前登记的 256 次对照保持相同采样设置，扩展阶段每批优先分配 6 个最新策略窗口。

两个扩展组另对 Body WM 无需飞行梯度的路径启用 CUDA 子步合并，保留原生阻力、质量除法及积分顺序。R48＋A48 在每个 Head 的 4 个真实起点上完成 H64 对照，4 项损失和各 6 个 Actor 梯度张量逐元素一致。Recovery 仅评分由平均 5.43 降至 5.02 秒，两条路径均已使用上一轮的冲量先验提速。746 项 CPU 回归及分开执行的 2 项 CUDA 回归通过。[逐块身份、采样计数和 H64 计时](paired_adam_R50_replay_and_body_flight.json)

### R50＋A48 后的首次提案拒绝

续训后的第 59 个提案耗时 858.3 秒。Recovery 取了 8 个风险、4 个落地和 4 个稳定起点，16 个 lane 都产生了跟踪梯度。全步越过命令预算，其余四个步长均在预算内。五次同起点评分均高于更新前的损失 −6.99267，其中最接近的是八分之一步的 −6.97600。训练器恢复了提案前的 Actor 参数快照，继续使用原 Replay。该提案期间没有原生回合结束，保留计数仍为 98。续训以来 Recovery 的 27 次提案耗时中位数为 492 秒，每次取自 14 份记录（中位数）；Approach 的 32 次分别为 156 秒和 4 份记录。[逐步损失、命令变化与恢复记录](paired_adam_R50_rejected_proposal.json)

### 回退后的 R51 和任务记忆解码耗时

下一次 Recovery 提案重选了 16 个有效起点，其中 11 个与被拒提案的 lane、回合、tick 组合不同。该提案在半步长保留，想象损失从 −7.26539 降至 −7.38863，累计达到 R51＋A48。期间新增原生完结回合为零。

想象中的任务记忆含 101 个字段。解码现在批量处理布尔值和浮点有效性，仅复制需要继承的原始状态。使用 R48＋A48 的冻结权重在 RTX 5080 上复算两个 Head 各 4 个原生起点，每次运行的四项 H64 目标和各 6 个 Actor 梯度张量均逐元素一致。Recovery 的 H64 候选评分平均 4.52→4.26 秒，Approach 为 5.40→5.18 秒。含反传的整段差距较小，分别为 12.08→12.02 秒及 20.45→20.26 秒。每种路径各测两次，与主训练共享 GPU。正在运行的 G1 使用启动时的实现。747 项 CPU 回归和另行执行的 2 项 CUDA 回归通过。[原生起点、逐次计时与参数更新记录](paired_adam_R51_task_memory.json)

### 100 次保留更新与后续扩展的 Return 监控

G1 达到 R52＋A48，101 个提案中实际保留 100 次。R52 在 8 个风险、4 个落地和 4 个稳定起点上取得跟踪梯度，半步更新使想象损失从 −7.50997 降至 −7.69516。预算检查中最终关节目标相对上次保留策略的最大变化为 0.01261 rad。真实线上交互累计新增 229,376 个环境步。此次提案期间新增完结回合为零。

第 128 次更新时使用同一独立固定面板作配对评价。该面板零更新基线的合格触球、球侧完成和 Complete5 分别为 72、48、21 例，平均 Return 为 −17.67489。在线训练病例和该固定评价面板分别统计。

在后续累计 1024／2048 次扩展中，冻结训练回合的 Return 持平只记录连续次数，持续下降仍触发暂停。判定沿用同一物理合同下 128 个冻结回合的真实分数和既有下降阈值。已登记的首轮两个 256 次组沿用原设置。748 项 CPU 回归通过，2 个 CUDA 专用用例在这次 CPU 运行中跳过。

### 40、64、96 次更新的 Critic 在同一批真实回合上的误差

三份检查点使用相同的 83 个留出 lane、Critic 输入、RTG、阶段掩码和行顺序。Recovery 有 1,918 个状态，覆盖 68 个独立 lane／回合组合。Approach 有 165 个状态，覆盖 73 个组合。Recovery 的预测均值减去实际 RTG，从第 40 次的 −2.70 变为第 64 次的 +5.04 和第 96 次的 +8.58。Approach 分别为 −2.02、+7.55、+6.47。第 96 次 Recovery 的 MAE 为 9.88，RMSE 为 15.42。同一时刻的回报排序准确率为 73.4%，Approach 为 58.9%。

这些回报由更早的冻结策略产生。对旧标签的估值偏高，当前策略的真实回报仍需用其自身完成的 PhysX 回合测量。后续启动的训练进程会在同一冻结标签及留出分区上同时记录 Critic 原生 RTG 校准前后的误差。正在运行的 G1 继续使用启动时的代码。训练学习率、预算及回报暂停条件沿用本轮配置。[三个检查点的数值与标签身份](paired_adam_critic_0040_0096.json)
