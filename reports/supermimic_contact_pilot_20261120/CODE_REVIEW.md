# 支线 B 代码审查记录

审查时间：2026-11-20

审查范围包括接触 pilot 抽取器、原生子步 trace、可恢复 probe launcher、Oracle A 审计入口，以及当前支线 B 的测试。支线 A 的未提交修改没有纳入本次代码变更。

## 已检查项目

- `quadsmash/rl/supermimic_contact_pilot_v1.py`
  - force matrix 的 manifold slot 全量保留并聚合。
  - contact 和 near-miss 可以出现在同一 lane 的不同窗口。
  - near-miss 记录 gap、接近速度、同一 trace 的 contact proximity。
  - threshold sensitivity 保留 `0.0001/0.001/0.005/0.01 N` 四档。
  - dedup 同时绑定 event、capture、lane、TOI frame、contact order，重复 capture 不会被错误折叠。
  - split 按已知 spec 原子分组，legacy unknown spec 按 capture 原子分组，行标签与分组分配一致。
  - source、spec、episode、capture、split group 的跨集合重叠显式审计。
- `quadsmash/rl/supermimic_native_substep_trace_v1.py`
  - 原始 contact matrix 和 slot 位置保留。
  - racket body、filter index、filter path 和 manifold normal 的可用性写入 metadata。
- `scripts/run_supermimic_event_aligned_native_probes.py`
  - 完整 probe 才计入 completed。
  - `--resume` 会跳过完整文件并归档不完整分支。
  - launcher 不执行模型更新。

## 验证命令

```text
python3 -m py_compile quadsmash/rl/supermimic_contact_pilot_v1.py quadsmash/rl/supermimic_native_substep_trace_v1.py scripts/run_supermimic_event_aligned_native_probes.py scripts/plan_supermimic_contact_pilot.py scripts/audit_supermimic_oracle_a.py
/home/autodrive/Isaac/env_isaaclab/bin/python -m pytest -q tests/test_supermimic_contact_pilot_v1.py tests/test_supermimic_native_substep_trace.py tests/test_supermimic_probe_aggregation.py
git diff --check HEAD~5..HEAD
```

结果为语法检查通过、`9 passed`、空白检查通过。多 spec 合成 split 审计通过，train、validation、test 都有行，source 和原子 group 没有重叠，并存在未见 spec holdout。

## 审查结论

支线 B 的 CPU 抽取和 split 逻辑可以继续运行。原生 GPU 采集仍然必须在每一批完成后重新构建 manifest，并以 coverage gate 决定是否准入。`policy_training_admitted=false`、`contact_bptt=detached`、`v13_residual_gradient=detached` 继续保持。

## cross-event 采集前审查

`/tmp/supermimic_contact_pilot_specs_cross_20261120` 的 15 个 spec 已逐个通过 `validate_spec`。所有 spec 都启用 `capture_substeps`，event tick 合法，lane 无重复，并覆盖 54 个 cross-event lane。manifest SHA256 为 `a3c4deef03f1160b59c47ef8dc7730d8fadd87f100408a32dcd09cd3063b2f72`。

该批次使用已审查的可恢复 launcher，启动前仍需确认当前 B1 fill 已结束且 GPU 只保留 A2 canary。采集完成后重新运行本文件中的测试和 pilot manifest 审计。

后续复核修正了 legacy `spec_unknown` 的审计边界。unknown 行继续按 capture 原子切分，并从已知 spec 的跨 split overlap 集合排除。修正后的测试和 unknown-spec 合成审计均通过，代码提交为 `71b9833`。

## 11 月 21 日复核与运行范围

逐行对照 FullRacketEnvV4 的 _consume_physics_frame 和 Isaac Lab ContactSensor 的点聚合实现后，撤回前文对原始五维矩阵的解释。球端第 0、1 个 filter 分别是球拍和地面。inverse_net 最后一个 filter 对应球，其余项属于机器人。刚体名 Racket 仅提供刚体身份。球网平面穿越不足以证明发生球网接触，原 cross-event 批次暂缓。

修复包括按对象取力和位置，单独保存 PhysX get_contact_data 返回的逐点法向、法向力、位置、分离距离和 filter 归属，分开保存刚体名与缺失的部件标签。新增计数和续跑检查覆盖 checkpoint、分支、规格、文件散列和完成的控制步。四组定向测试共 16 项通过，其中包括地面单独受力、地面和拍面同时受力、机器人碰网、稀疏接触点索引、损坏的续跑文件和独立的子步时钟。

短窗 GPU 首次试运行限 1 个进程、1 批、32 条 lane、seed 270080。核对了 teacher 的 128×18 关节目标、原始场景偏移、root link 到质心速度换算、1 个准备子步与 160 个测量子步、规定的空气阻力、新建输出目录和原子写入。使用 tick271 nominal_a 的球拍运动状态和原生铰接模型。采集器 SHA256 为 e2c58b0cfc04dbaca91d44abe80bbe461689a6a23119bad66ec286695a4d2b7b，trace 为 f178ddc89fadf1c476acafeb44933ac57ec3dc910cb52ccd2b2106724de8795d。此次结论限短窗试运行。扩量前须核对实际物理帧、逐点冲量与 filter 合力的一致性、重复事件及每个分桶的实测支持。

首批 32 lane 进程正常退出。160 个连续原生步内观测到 29 个接触，29 个均有逐点法向。逐点法向力合成结果与球拍 filter 最大分量差为 0.0002442 N，覆盖中心 4、网格 10、近框 10、边缘 5。重复接触为零。

复核短窗修改，对照本机 Isaac Sim RigidContactView.get_friction_data 的四元返回、起始偏移和计数，对单独的球拍 filter 累积摩擦，使用独立的地面和球网负例，缩小采集文件到球拍和球所需字段。17 项 CPU 测试通过。批准一次 2 批 × 128 lane 的实测，seed 270097–270098，单个 GPU 进程。采集脚本散列 182e93679973299a81afe2dee1ac6dd9bf65739da2f95298854ff8be0d243309，trace 散列 e74c020eb9b09744fdb5a14e66e089e1940ad378804a6861edfe76bd4ab665d7。放量条件包括实际逐点标签、球动量平衡及样本重复率审计。

两批试采实测共 217 个接触区间，其中 8 个为后续接触。两批逐帧冲量相对于球的动量变化（扣除重力、阻力）的相对误差 P95 分别为 0.00485% 和 0.00010%。两批均有独立测得的地面及球网负例。静态复核发现低拍速、高角速度、低切向速度覆盖不足。按 lane 控制根与关节的初始速度、目标以及入射切向速度，仍由原生铰接模型自由演化。复核 root/关节张量维度、seed 与现有批次互斥、时钟与输出目录后，批准单进程 17 批 × 128 lane，seed 270099–270115，目标约 1800 个新增接触。验证命令通过 17 项测试。采集脚本散列 ca626460f03e22d96a31023fc75d8d6e457c968a05c4c9cb5dfa0f8a0408171d。此后依据缺失桶定向补采。

CPU pilot 首次构建以 2 个 QA 批和 17 个 fill 批为唯一输入。核对 19 个独立 seed，完整 trace 均原子落盘；球拍 filter 分离、摩擦冲量接口和刚体速度换算经上述实测验证。抽取器对逐点几何部件保留来源，负例每个 capture/lane 最多选一条，三个 split 以独立 teacher lane family 分组。已有 17 项回归通过。首个输出目录全新，按实测分桶决定是否追加采集。

19 批共有 2083 个击球区间，97 个属于后续接触，其中 75 个来自擦边设置。首版抽样覆盖审计仅缺后续接触 6 条。审查擦边补采开关仅改变发球条件，关闭该补采中的地面和球网负例。保留事件间无拍力步数与期间最大几何间隙，可用于区分短暂断力和分离。批准一次 seed 270116、128 lane、1 批擦边采集；17 项测试通过，采集脚本 SHA256 为 719d1856cd6879a34d1c7b9bcbf55285bdbb313b83dbe1170f8d329740f1660c。

补采进程正常退出，共 106 条 lane 击中，143 个连续接触区间。复核 20 批输入目录与独立 seed 后启动正式 CPU 抽样。目标 2000/1000，负例每个 capture/lane 最多一条，源目录只读，单个 CPU worker，写入全新目录。

正式 pilot 的样本门槛通过。审查解析校准入口，参数搜索仅遍历 train，validation 与 test 各自按 archive 散列复验。自由飞行从帧 0 开始，独立原生球拍输入提供后续拍面姿态，预测首触时间不读取真值力；首触匹配误差、漏检、误检分别报告。检查无接触时钟哨兵、理想冲量和方向/幅值指标通过。批准一次 CPU 三集合首触评估，输出独立 JSON，策略训练继续关闭。

抽查测量审计入口使用无球拍 manifold 子步独立估计噪声，力与动量同钟对齐，切向力按球拍 filter 汇总，文件散列强制一致。许可 CPU 读取 20 批受控轨迹，汇总阈值、重复接触的分离和推断质量，不新增 GPU 作业。

训练集首触时钟复核。只读 seed 270097 lane 109 和 270102 lane 8，检查原生状态、预报自由飞行和各对象力，不读取旧 test 的 lane 身份。核对 advance_trajectory 的半隐式推进顺序和 observe_frame 在 sim.step 之后取样，比较输入为前一子步或当前子步的时钟。诊断仅使用 CPU 两线程，不改原始轨迹。

CPU 离散碰撞候选复审。校验首步与末步、整段无接触哨兵 len(gap)、估计球拍接触点的角速度项、负例分母与训练/验证的文件隔离。新预测只读每一子步起点的位姿与速度，用法向相对速度推进至该步末端，保存到 i+1 的力记录。固定接触包络为物理配置中单体 contact offset 的两倍。出球尺度和摩擦仅用训练集拟合。允许在已有 pilot 上运行一次 CPU 对照；此前已报告过的 test 作为历史对照，独立结论需新前缀。脚本 SHA256 e388aac2ae78fa318f0b6ace7e0c3d0850ca0c17090d2231815019b967e8b1db。

独立前缀验证预注册。使用归档 gpu_event_lanes_v2/spec_00_tick267/nominal_a，其 teacher 哈希 3c56f7ec1e51b241d77c550643d56ff1363791d90d72d5c82dac5c908d853f67。pilot 的 tick271 teacher 哈希为 7f23c6c50f2f298c28c54e7f439776a90daba227a84322776359ef18b79c7e59。固定模型参数为包络 0.0002 m、法向尺度 0.9282682538032532、摩擦 0.16。预先指定 seed 370201 至 370206，每批 128 条，全部计入，仅读取验证结果一次。预计约 130 个整段无球拍接触的负例。对照 64 控制步稳定性另行验收。复审采集脚本、160 个子步、单进程、资源和目录隔离，手动采集新增球端总法向力，复核 Isaac 单球传感器的 (lane,body,3) 轴；沿用已有对象过滤。拟运行时钟边界回归、原生 trace 回归及既有 pilot 测试，通过后启动一组独立验证采集。只保存这六个短窗，不重采 pilot。

12 项定向测试通过。采集入口 SHA256 719d1856cd6879a34d1c7b9bcbf55285bdbb313b83dbe1170f8d329740f1660c，子步 trace 为 285968922f840bcce7f61ee70d61d204388827e1792c8ae6ba12608e2067acba。批准上述一次六批运行。

独立评价入口复审。逐个核对六个预注册 seed、每组 128 条原生 lane、teacher 与 pilot 的来源哈希、模型和 pilot 哈希绑定、缺失批次报错。拟合参数从已冻结报告只读加载，所有整段负例进入分母；允许一次 CPU 评价。入口 SHA256 22ca675006a240e76b323f8ca44602c71a6a3a34a0b7c96ead74fbbf1eeb4e2b。

独立集评价完毕。批准对历史 pilot 做错误归因，检查所有错分 trial 的首帧 gap、原生球速改变量、manifold/filter、以及球拍力。使用旧 test 作开发诊断，记录其角色变化。新 tick267 的六组不用于本次错误归因。

三条历史错分均在 frame 0 已穿透。两条误报在前几帧存在原生球拍 manifold 且其法向力为零，球速在 frame 1 即变化。进一步批准仅使用 train 的初始穿透样本核对原生分离速度、法向力、切向摩擦和几何法向。训练与测试指标沿用原分母，禁止用首次穿透或是否有 manifold 剔除负例。

训练标签复核发现两条原先记为负例的整段都有测得的球拍切向力，法向力为零；训练反例首帧之后的球速变化主要沿切向。冻结原生任务 hit 标签和原评估结果。另外给球响应训练按任一原生球拍通道超过 0.001 N 定义响应区间，两个测量通道分别积分，单个对象滤波器才合并槽冲量，几何法向按接触点朝球心定向。零法向冲量时有效质量记为不可推断。负例排除窗口与敏感性扫描同样采用两个通道；近失的接近速度按最近拍面法向计算。检查历史无摩擦流的回退、独立训练源 20 批、正式 pilot 单独输出、不读 tick267 测试源。先运行含纯切向回归的测试，通过后仅在 CPU 重建，不补采。

13 项测试通过。批准从既有 QA、fill、repeat 原生目录单进程重建 2000+1000 个窗口，抽取器 SHA256 a8ab91c91cd5348470775d8db83f73b490169f764b1030784b37b99f3f491c5b。独立前缀保留为外部验证。

重新抽取后 2233 个候选中选出 2000/1000，各硬配额通过。复核测量审计读取新 pilot 的实际源哈希，逐帧比较合成法/切向冲量与原生 Δv，在无 manifold 的子步统计双通道噪声；批准一次 CPU 汇总。同时只读检查已归档 Body WM 检查点、64 控制步可用窗口与训练 lane 清单。

64 控制步 transfer 诊断预审。使用与现行 A2 同一文件的冻结 Body WM、训练时声明的整组验证 lane、预先指定起点 tick240、64 个连续 10 ms 控制步。核对按 episode 与终止位检查窗口、球已启动、训练和验证 lane 互斥、归档与发球配置哈希；使用记录的最终关节目标和上一动作，Body WM 分支下一步只接收自身机器人与球状态。另一分支仅给实测机器人端点以便分离误差。保留原记录的 rally/contact_count 作为控制步事件；真实子步时钟指标仍取原生独立短窗。按每步 160 个原生长度子步推进球，球拍使用冻结校准、球网和地面使用解析几何/恢复系数，位置 Hermite、姿态符号对齐。先用零碰撞、拍面正碰和原生窗口完整性检查做预检；通过后批准一轮双分支 CPU 评估，所有 32 条同时计入，不跳过失败窗口。脚本 SHA256 4956641de901ca7fc937e3ae3414f56995a147edf72bb10c6d940fa384177a2d。

运行预检通过：32/32 均已发球且 64 步无重置，直线飞行与现有积分器一致，静态正碰触发并反弹。预检发现 packed valid 是 float32，已将计数的有效位显式比较为 1 后再布尔求交。复审唯一改动，批准预注册运行。

对初始穿透的机理仅用训练源审查。以球拍摩擦、法向两个原生通道独立分类，汇总初态分离/接近与首次实测响应的关系；不使用原留出集决定修复参数。

纠正修订版 split。首次双通道试抽的行数变化使自动平衡更换了两个以上的 family，作废该次 split 以及基于该版本训练分组的穿透频数诊断。引入 --split-pilot，用原始 pilot 内的每个源哈希固定 train/dev/test，要求源集合严格相等且同一独立分组只有一种归属，把父 pilot 散列写入结果。预审本次重建只改变 split，未依赖任一留出集调整模型；准许 CPU 测试，验证真实分组身份与当前持有的 20 批轨迹后重新构建。

14 项回归通过。批准在全新 response_locked 目录重建；旧试抽目录在哈希、配额和分组验证后清理。

64 步首次评估完成，Body WM 相比实测端点漏检 3 条并额外误报 1 条。批准在同一批 32 条上做归因：增加每一控制步从记录的当前机器人状态预测下一步的诊断分支，球在所有分支中保持自回归，旧两种分支同样重跑以验证复现。实测未来机器人端点只用于评估位置误差与原本的端点上界分支。

最终锁定 split 的 measurement 和误差桶审计。检查原 20 批严格同源、从父 pilot 冻结模型的对应关系、预测只读真实起触前球/拍状态，整段积分的原生法向与摩擦向量独立用于评价。分子分母分别为预测与实测冲量幅值，向量误差单位 Ns。零分母只从相对比值中排除，仍统计绝对误差和桶计数。允许在 response_locked 上一次 CPU 复核两份报告。

仅针对已穿透、法向分离且仍有切向滑移的首段，加入有界滑动响应候选。上界取已有球滚动约束 2/7 乘滑移速度与已冻结摩擦系数乘穿透修正速度的较小值，两个上界单位均 m/s；测量审计中乘球质量转换 Ns。该项只作用于切向，不凭穿透制造原生法向力。连续 rollout 跨控制步携带重叠标记，每次进入该重叠只施加一次修正。评估保留原法向任务结果，另按实测任一力通道对全部 20+6 批比较预测规则；前次看过 tick267 的法向评价，标明本次是既有外部前缀的二次分析。所有系数沿用原拟合，未按这些结果调节。复核分组哈希、单位、无回读未来球速度、力门槛、负例分母与有限值。批准定向测试后分别在 CPU 上运行全窗响应、冲量桶、64 控制步分析，之后停止 B 采集。

最后复核原生 hit 定义。双通道标签解释了先前误报，但仍须实测修复是否能通过原来的负例分母。若预测到先滑动、后法向接触，按新滑动冲量重新推进该 lane 的球轨迹，并只把真实预测到的法向约束计为 task hit。沿用同样的球/拍力学参数、原 69 个历史法向负例与 149 个外部负例。候选 lane 选择仅依赖两个解析时钟；逐步推进只读取 frame 0 的球状态和原生球拍轨迹。手动复核首末帧、未触发 lane 复用原输出、连续滑动单次冲量和两个分母分别输出。准许 CPU 在原报告之外验证一次。

清理前比对临时试抽与锁定版本的全部行身份、正式父 pilot 的逐源 split、20 份文件 SHA256、旧双分支与三分支结果的共有字段、滑动和完整双标签报告共有指标、旧 stdout 与原 manifest；仅删除已证实被替代的派生产物。检查最终训练 pilot 中 1000 个负例的完整排斥窗口，两种实测接触力均需低于阈值。

最后增加真实策略工况的现存归档复验，无新增采集。选择 tick274 nominal_a 完整归档，与训练用 tick271 共享策略种子，属于不同的实测时段，不能算跨策略种子验证。预先指定保留全部 128 条 lane，冻结解析滑动修正后仅评价原生任务法向 hit 与 Δv，记录开窗时已有接触的 lane 数。归档缺少逐条摩擦时只使用已验证的球端法向滤波器。

交付前复核最终涉及的 B 入口和报告。两种标签的真值、源哈希、采集列表、锁定 split 和 final manifest 保持一致；逐源核查训练输入及实际 1000 个负例窗口通过。清理 3 个过期试抽张量 8,704,419 字节，另清理 7 个重复/已替代的派生文件 9,194,619 字节。批准仅运行现有解析先验、采集器、原生 trace、pilot 和时钟回归，不启动模拟器或训练器。

最终 26 项回归全部通过。支线 B 采集、解析评估和验证进程均已退出。
