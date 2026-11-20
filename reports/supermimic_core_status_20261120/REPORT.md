# SuperMimic 核心闭环状态报告

日期：2026-11-20

分支：`feat/supermimic-hybrid-toi-v13-20261119`

本报告对应本轮代码提交 `7c1231b`。工作区中已有的任务合同、环境和评价脚本修改未纳入本提交。

## 已完成

### 真实 PhysX 数据

- 冻结策略归档包含 `800 × 128 = 102400` 条真实控制边界 transition。
- 事件数据集 `supermimic-contact-event-v1` 已生成 `1026` 条记录。
- 事件计数为 qualified contact 75、frame hit 37、crossed net 61、target landing 55、recovery 37、body ground 61、fault 16、control-boundary negative 684。
- 按完整 lane 切分后，train 775、validation 118、holdout 133。
- 该归档没有 PhysX 子步真值。negative 记录明确标记为 control-boundary negative，不能当作子步级 no-contact 证明。

### 事件窗口

现有完整 WM 窗口为击球前 4 个控制 tick 和击球后 64 个控制 tick，对应 40 ms 和 640 ms。

本轮根据事件分布新增可配置窗口，并生成了事件核心数据集：

- 事件核心：`pre=4, post=8`，覆盖 40 ms 接近过程和 80 ms 接触后分离过程。
- WM/想象：`pre=4, post=64`，保留 640 ms 自回归和恢复上下文。
- 接触子步轨迹建议覆盖 TOI 前后至少 8–12 ms。该要求需要新的带子步轨迹采集满足，旧冻结归档不具备这一信息。

### 原生响应样本

第一批 holdout probe 已形成 8 个接触 lane。

第二批 calibration probe 已完成 20/20 Isaac 分支，得到 31 个独立接触样本，来自多个 lane 和多个事件规格。响应范数中位数约 `0.00230 m/s`，最大值约 `0.33015 m/s`，说明样本仍存在明显的响应幅值不平衡。

当前正在进行更大扰动 `epsilon=0.005` 的高响应 probe。当前两个 nominal 分支已推进到 200 个控制 tick，GPU 采集仍在运行，策略更新为 0。

## 离线模型结果

### V13 原生响应拟合

使用 31 个 calibration 接触样本拟合 V13，256 次 GPU 更新。

- checkpoint：`/home/autodrive/LZQW/experiments/supermimic_native_response_fit_v13_20261120_full31/action_ball_candidate.pt`
- checkpoint SHA256：`2cd87506be7f6e341daa8ba0617b4077f2886703e8118ffb1122f419d36038b9`
- validation RMSE：`0.00115010`
- validation zero-response baseline：`0.00116222`
- validation 最小方向余弦：`-0.05840`
- final holdout RMSE：`0.01621445`
- final holdout zero-response baseline：`0.01623946`
- final holdout 方向余弦：`0.69767`、`0.24758`
- receipt：未签发

结论是验证集有轻微改善，但独立留出集的预测响应仍接近零，方向和幅值没有达到原生接触准入要求。

### 低维解析冲量校准

使用训练集选候选、开发验证集选择、最终留出集一次评估的三段式流程，选出的参数为：

- `gap_bias_m = -0.004`
- `normal_impulse_scale = 1.25`

结果：

- validation RMSE：`0.00114906`，baseline `0.00116222`
- validation 最小方向余弦：`-0.06111`
- final holdout RMSE：`0.01624112`，baseline `0.01623946`
- final holdout 最小方向余弦：`-0.25027`
- final holdout 最大相对误差：`1.03546`

该低维校准没有通过原生方向和幅值准入，不能开放 contact-aware Actor BPTT。

## 代码变更

本轮新增的可复现代码变更：

- `quadsmash/rl/supermimic_contact_event_dataset_v1.py`
  - 事件窗口支持显式 `pre_control_ticks` 和 `post_control_ticks`。
  - 保留默认 `4+64` 语义。
- `scripts/build_supermimic_contact_event_dataset.py`
  - 新增 `--pre-control-ticks` 和 `--post-control-ticks`。
- `tests/test_supermimic_contact_event_dataset_v1.py`
  - 新增短事件核心窗口测试。

## 测试

本轮针对事件数据集的定向测试结果：

```text
5 passed in 0.67s
```

之前版本保存的完整 CPU 测试记录为 253 passed。本轮没有重新宣称该旧记录覆盖新 native probe 或 GPU 运行时。

## 当前准入状态

```text
真实 PhysX transition archive              已完成
事件级 control-boundary 数据集             已完成
事件核心/完整 WM 窗口配置                   已完成
原生子步 probe                              正在采集
V13 离线拟合                                已完成但未通过
低维解析冲量校准                            已完成但未通过
wm-validation-receipt                       未签发
contact-validation-receipt                  未签发
Recovery/Approach Actor 更新                未启动
Critic 更新                                  未启动
大规模策略训练                              未启动
```

当前不能把离线拟合 RMSE 的轻微下降解释为真实击球能力改善。必须先完成高响应、多独立事件的子步级原生 probe，并重新验证接触召回、TOI、方向和幅值，再决定是否开放策略学习。
