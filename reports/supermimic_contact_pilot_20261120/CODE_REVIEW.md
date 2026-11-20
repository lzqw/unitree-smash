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
