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
