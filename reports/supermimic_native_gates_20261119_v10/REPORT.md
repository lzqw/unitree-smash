# SuperMimic native gate adjustment report

日期 2026-11-19

## Scope

本轮只执行代码调整、GPU PhysX 事件探针、动作响应标定和合同测试。没有启动 Recovery/Approach actor 训练、R8→A8→R8 闭环或大规模训练。

## Code

- 固定探针代码提交：`a4311850c2c563501fe6cf4bd21f901056f1e81b`
- 前置时序探针代码提交：`7e03755d3c6f40c09c7f08cb9a8077a00801a704`
- 本轮新增：V8 当前执行目标输入、V9 几何/模式门控 floor、V10 线性执行目标响应通道。
- 所有迁移均使用严格 shape 检查和 `strict=True`，没有使用 `strict=False`。

## Gate status

1. 候选安全筛选：通过“拒绝退化候选”合同。旧候选被拒绝，原因是留出集 pre-impact 4-step 指标由 2.46004 退化到 2.83483。该结果不是候选通过。
2. PhysX 事件对齐探针：已通过 GPU 运行。14 个独立 lane、nominal_a/nominal_b/minus/plus 四分支、lead-time action tick 265、event tick 267、epsilon 0.01，所有分支退出码为 0。
3. 原生动作响应方向/幅值：未通过。V10 256 次 GPU 标定后的 4-lane 支持验证集为：RMSE 0.0324127，最小方向余弦 -0.7736503，最大相对幅值误差 1.01634。
4. native direction-validation receipt：未签发。方向和幅值门槛未通过，且当前支持样本仍不足以证明跨接触模式泛化。

## Evidence

- 14-lane temporal probe bundle
  `.../gpu_event_lanes_v2/bundles/contact14_temporal_v4_t267.pt`
  SHA256 `aeb90a2f87da99c0b507f70edd9d1ed320db0e48203ca4de4c66105d84f6193a`
- Train split 7 supported lanes
  `.../contact14_train7_supported_temporal_v4_t267.pt`
- Validation split 4 supported lanes
  `.../contact14_val4_supported_temporal_v4_t267.pt`
- V10 calibration output
  `.../temporal_response_fit_v8_v10_t267_085535/calibration_result.json`
- V10 candidate SHA256 is recorded in its calibration output. It is not admitted to Actor BPTT.

## CPU contract

命令：

```bash
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic_*.py -q -ra --maxfail=1
```

结果：本轮 V8/V9/V10 代码后专项测试 `10 passed`。V10 后全量结果为 `441 passed, 1 warning`。

## Interpretation

GPU PhysX 探针链路已经运行，候选安全筛选会拒绝退化模型。模型仍不能在独立原生样本上同时复现球响应方向和幅值，因此不允许生成 native direction receipt，也不允许开启 contact-aware Actor BPTT、Recovery/Approach 更新或大规模训练。

下一次调整需要补充跨 episode 的可支持接触响应样本，并按接触前、接触当步、接触后分别核对实际执行命令、接触子步时序和球速度突变。不能通过放宽阈值或重复同一 lane 达标。

## Follow-up code adjustment

提交 `929f091` 将可用的 PhysX substep trace 摘要写入 probe bundle，并在 paired metrics 中报告 nominal 重放的首次接触子步、扰动分支的接触时序跨度和物理子步时长。它只用于事件对齐诊断，不作为世界模型未来输入，也不放宽方向/幅值门槛。由于本轮 14-lane 采集未开启 substep trace，该字段需要下一次固定版本探针采集后才能用于判定。

## Current-code GPU follow-up

在提交 `095842125e083ff3764f969671f5663635dad23b` 的代码下，重新完成了一个
9-lane、四分支、同前缀的 GPU PhysX 事件探针。该运行没有更新 actor、world
model 或 critic，四个分支均正常退出。

- 输出目录：`/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261118_233655/gpu_tick274_all9_lead2_temporal_current_v2`
- 分支：`nominal_a`, `nominal_b`, `minus`, `plus`
- 每个分支退出码：`0`
- 探针状态：`COLLECTION_FINISHED`
- actor/world/critic updates：`0/0/0`
- 当前 bundle：`gpu_event_lanes_v2/bundles/contact9_current_substep_temporal_t274.pt`
- bundle SHA256：`e8d8e994c249ce62fdcce9e0681a73695ebc9eb267d8ab9be4d5d72bd1e29e49`
- bundle 中的原生探针数：`9`
- `capture_substeps=true`，PhysX 子步 `dt=6.25e-05 s`

选定 9 个接触 lane 的 nominal 首次接触子步为
`[13,18,67,105,24,15,150,48,57]`。minus 与 nominal 的首次接触子步一致。
plus 仅在 lane 28 和 61 出现了相邻子步偏移，分别为 `67→68` 和 `150→149`。
四个分支的接触计数均为每 lane 2 次，lane 43 为 3 次。该结果说明当前探针
已经保存了可用于事件时序审计的子步证据，同时也说明扰动可能触发离散接触边界。

当前代码与旧 runtime identity 的模型代码哈希不同，因此没有复用旧 identity
强行打包。使用当前探针记录的执行代码哈希生成了新的
`runtime_identity_current_t274.json`，并完成了 bundle 构建。这个动作没有放宽
代码合同，也没有将 bundle 视为方向验证通过。

## Updated gate decision

当前四个准入环节仍为：

1. 候选安全筛选：通过拒绝退化候选的行为合同。旧候选仍为拒绝状态。
2. GPU PhysX 事件对齐探针：通过一次完整的 9-lane 当前代码采集，四分支退出码均为 0，且子步摘要可追溯。
3. 原生动作响应方向、幅值和局部 Jacobian：仍失败。已完成的 V10 结果为
   RMSE `0.0324127`、最小方向余弦 `-0.7736503`、最大相对幅值误差 `1.01634`。
   新的当前代码 bundle 目前只有原生子步和同前缀证据，尚未产生新的通过指标。
4. native direction-validation receipt：未签发。

因此仍禁止 contact-aware actor BPTT、Recovery/Approach 更新和大规模训练。
当前 GPU 进程已结束，未启动训练进程。阈值没有放宽，也没有通过重复同一 lane
或忽略离散接触偏移来宣称验证通过。
