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
