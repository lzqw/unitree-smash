# Hybrid WM verification update

日期 2026-11-19

本轮只修改验证和离线响应校准代码，没有更新 Actor、Critic、在线 WM，也没有启动大规模策略训练。

## 代码修复

- 原生 temporal Jacobian 现在对完整 minus/plus 执行命令序列做同一标量插值。后续控制 tick 同样使用插值后的执行命令，不再只扰动第一个 tick 后回放 nominal 命令。
- temporal rollout 在 probe region 和 native response fitter 中传播 `present`，避免跨发球边界使用旧的球存在状态。
- 接触模式分组增加 PhysX 首次接触子步比较。接触标记相同但首次接触子步变化的样本标记为 `contact-time-switch`，不会进入 smooth Jacobian receipt。
- native response calibration 增加 component-scaled Huber 响应损失、预先声明的方向响应阈值、三分法 final holdout 和逐样本 Hybrid 分解报告。
- V11 原生响应头增加显式 `native_response_gate_mode`。`impact_probability` 保留历史行为，`present_only` 仅用于离线校准诊断，要求后续加入真实非接触负样本后才可考虑准入。

## 测试

```text
460 passed, 1 warning
```

## 三分法离线拟合

训练集 12 个样本，development validation 5 个样本，final native holdout 9 个样本。三组 sample identity 完全不重叠。

### Robust loss，保留 impact-probability gate

输出 checkpoint SHA256 为 `186dffc88eb3dcbff39e961481857c6ed6badbc276b60f4fbd9c3ad4727f969f`。

- development validation RMSE `0.0338478`，零响应基线 `0.0337355`
- final holdout RMSE `0.1395846`，零响应基线 `0.1402166`
- final holdout 最小方向余弦 `-0.8617`
- final holdout 最大相对幅值误差 `3.1542`

RMSE 有轻微改善，但方向和幅值仍未通过。

### `present_only` 校准模式

输出 checkpoint SHA256 为 `e63a9558d61b4c1c88fabf4ed7e61aff4b8fb5fea6780d045db23a5148f938af`。

- development validation RMSE `0.0337767`，零响应基线 `0.0337355`
- final holdout RMSE `0.1403685`，零响应基线 `0.1402166`
- final holdout 最小方向余弦 `-0.9475`
- final holdout 最大相对幅值误差 `1.1246`

解除解析接触门控后，留出集方向没有改善，最终 RMSE 仍高于零响应基线。当前训练数据缺少足够的非接触负样本和跨接触模式覆盖，不能把 `present_only` 候选接入 Actor。

## 诊断结论

逐样本 Hybrid 分解确认，部分真实接触样本的解析模型 `analytic_contact_active=false`，预测接触概率接近零，因此旧 V11 响应头无法输出有效修正。解除门控后，有限样本上可以产生响应，但在独立接触模式上泛化失败。这说明主要问题还包括事件时刻、接触模式和动作扰动覆盖不足，继续增加同一批 256 次拟合不会形成可靠 receipt。

当前状态仍为：

```text
offline_validation_code_complete
native_direction_receipt_not_issued
contact_bptt_blocked
actor_training_not_started
```
