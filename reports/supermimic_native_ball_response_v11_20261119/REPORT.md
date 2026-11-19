# Native ball-response calibration V11

日期 2026-11-19

## 结论

本轮只完成原生球响应候选拟合、指标拆解和合同测试，没有更新 actor、critic 或完整 world-model，也没有启动 R8→A8 或大规模训练。

原生动作响应准入仍失败。候选 V11 仅在固定的 12 条训练、8 条验证响应上拟合，不能签发 contact-validation receipt。

## RMSE 的定义与来源

标定脚本比较的是三步原生配对 rollout 的球速度响应：

```text
native = native_plus[:, 52:55] - native_minus[:, 52:55]
predicted = imagined_plus[:, 52:55] - imagined_minus[:, 52:55]
RMSE = sqrt(mean((predicted - native)^2))
```

因此该 RMSE 只针对物理状态 55 维中的球速度三维字段，不是 reward、终止概率、完整状态 55 维或机器人关节误差。它测量的是正负最终命令扰动在接触窗口内产生的球速度变化是否被模型复现。

V10 混合验证集的分量结果如下：

| 分量 | 原生响应 RMS | V10 误差 RMS |
|---|---:|---:|
| 球速度 x | 0.04079 | 0.04515 |
| 球速度 y | 0.05046 | 0.05176 |
| 球速度 z | 0.16008 | 0.17246 |

z 分量占主要误差。V10 预测响应 RMS 为 `[0.01844, 0.00558, 0.05902]`，明显低估原生冲量。零响应基线 RMSE 为 `0.09973`，V10 为 `0.10718`，因此 V10 没有超过零响应基线。

## V11 修正结果

V11 增加了因果原生响应头，并冻结原有 V10 几何和命令头，只拟合新增响应头。这样避免用 12 条样本重写已拟合的基础模型。V11 固定验证集的最佳迭代结果：

- 输出：`/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261118_233655/temporal_response_fit_v11_nativehead_mixed_train12_val8_e/action_ball_candidate.pt`
- SHA256：`c49d4cd006112ea2b948d5261fe771a8914479243ef30a8e5a29a38f4dfb7d61`
- 训练样本：12
- 验证样本：8
- GPU 更新：256
- RMSE：`0.0999513`
- 零响应基线：`0.0997264`
- 最小方向余弦：`-0.8769`
- 最大相对幅值误差：`1.2303`
- 最佳验证迭代：16
- V11 预测响应 RMS：`[0.00597, 0.00107, 0.00045]`

V11 的 RMSE 只比零响应基线略高，方向余弦仍有负值，说明新增头在留出 lane 上仍趋向预测接近零的响应，而没有恢复真实离散冲量。它不能用于 actor BPTT。

## 根因定位

1. 指标本身确实主要来自球响应，具体是三步 rollout 后的 `state[:, 52:55]` 球速度差。
2. 原生接触响应在不同 lane 上幅值差异很大，验证集范数为约 `0.0129` 到 `0.4618`。小范数样本会导致相对误差很大，z 分量同时具有最大绝对误差。
3. 接触是离散碰撞事件。相同物理前缀下，0.008 rad 左右的扰动可能改变冲量大小或相邻子步接触时刻，不能被当前平滑几何残差仅靠 12 条样本稳定外推。
4. 现有样本主要来自一个策略版本和有限 episode/lane。它们不足以覆盖接触模式、拍面方向、相对速度和冲量幅值。
5. 旧 V10 的响应头受几何/模式 gate 约束，碰撞发生而 learned geometry 未进入 gate 时会输出接近零。V11 解除了该新增头的 gate，但在当前数据量上仍无法泛化。
6. 当前 native probe 已记录子步时序。部分扰动会造成相邻子步的接触边界变化，这种模式切换必须单独分类，不能与平滑 Jacobian 混在一个回归目标中。

## 已完成的代码修复

- V11 迁移现在支持从 V7/V8/V9/V10 候选严格构造，保留原有张量并只新增零初始化原生响应头。
- 原生标签复用只允许 actor、task、policy execution、policy version 和所有原生执行代码哈希一致。预测代码版本可以变化，但会写入 provenance，不会被当作新的 PhysX 证据。
- V11 标定默认冻结继承的 V10 参数，只更新 `native_response_head`。
- 标定结果增加球速度各分量 RMSE、原生/预测响应 RMS 和零响应基线 RMSE。
- 增加 V10→V11 严格迁移和指标字段测试。

## 测试

```bash
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic_*.py -q -ra --maxfail=1
```

结果：`444 passed, 1 warning`。

V11 专项测试：`14 passed`。

## 下一步

要真正降低 RMSE，必须增加独立 episode、lane、接触模式和动作邻域的原生配对数据，并按接触子步是否改变分层评估。下一次数据采集应同时保存实际执行命令、动作生效 tick、首次接触子步和球速度突变。不能通过放宽方向阈值、重复同一 lane、或把零响应基线当作成功来签发 receipt。

在原生方向余弦、幅值误差和局部 Jacobian 均通过前，不启动 contact-aware Actor BPTT、Recovery/Approach 策略更新或大规模训练。
