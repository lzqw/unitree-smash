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
